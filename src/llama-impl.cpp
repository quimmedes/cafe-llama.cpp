#include "llama-impl.h"
#include "llama-mmap.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

struct llama_logger_state {
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_logger_state g_logger_state;

void llama_prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || n_rows == 0) {
        return;
    }

    GGML_ASSERT(ggml_is_matrix(tensor));

    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const auto * base = (const char *) tensor->data;

    std::vector<llama_memory_range> mr;
    mr.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        GGML_ASSERT(rows[i] >= 0 && rows[i] < tensor->ne[1]);
        mr.push_back({ base + (size_t) rows[i] * tensor->nb[1], row_bytes });
    }

    llama_prefetch(std::move(mr));
}

namespace {

// Threads that each read one row at a time. A page fault blocks its thread, so the number of threads
// is the number of disk reads in flight. Between gathers they touch the pages that prefetches asked for.
struct llama_row_pool {
    std::vector<std::thread> threads;

    std::mutex              run_mtx; // one job at a time
    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;

    const std::function<void(size_t)> * job = nullptr;
    size_t              n_items = 0;
    std::atomic<size_t> next{0};
    int                 n_busy  = 0;
    int                 n_touching = 0;
    uint64_t            gen     = 0;
    bool                stop    = false;

    std::deque<const char *> touch; // bytes whose pages a prefetch wants, oldest first

    explicit llama_row_pool(int n_threads) {
        for (int i = 0; i < n_threads; ++i) {
            threads.emplace_back([this] { worker(); });
        }
    }

    ~llama_row_pool() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    void worker() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(size_t)> * f = nullptr;
            const char * p = nullptr;
            size_t n = 0;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return stop || gen != seen || !touch.empty(); });
                if (stop) {
                    return;
                }
                // a gather goes first; a late wake-up after it finished finds job == nullptr
                if (gen != seen) {
                    seen = gen;
                    if (job != nullptr) {
                        f = job;
                        n = n_items;
                        n_busy++;
                    }
                }
                if (f == nullptr && !touch.empty()) {
                    p = touch.front();
                    touch.pop_front();
                    n_touching++;
                }
            }
            if (p != nullptr) {
                (void) *(const volatile char *) p;
                std::lock_guard<std::mutex> lk(mtx);
                if (--n_touching == 0) {
                    cv_done.notify_all();
                }
                continue;
            }
            if (f == nullptr) {
                continue;
            }
            for (size_t i; (i = next.fetch_add(1)) < n;) {
                (*f)(i);
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (--n_busy == 0) {
                    cv_done.notify_all();
                }
            }
        }
    }

    void run(size_t n, const std::function<void(size_t)> & f) {
        std::lock_guard<std::mutex> run_lk(run_mtx);
        {
            std::lock_guard<std::mutex> lk(mtx);
            job     = &f;
            n_items = n;
            next    = 0;
            gen++;
        }
        cv.notify_all();
        for (size_t i; (i = next.fetch_add(1)) < n;) {
            f(i);
        }
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&] { return n_busy == 0; });
        job = nullptr;
    }

    // drops the queued touches and waits for the ones being read, before the memory they point to goes away
    void cancel_touch() {
        std::unique_lock<std::mutex> lk(mtx);
        touch.clear();
        cv_done.wait(lk, [&] { return n_touching == 0; });
    }

    void add_touch(const char * const * ptrs, size_t n) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            touch.insert(touch.end(), ptrs, ptrs + n);
            // stale hints must not hold back the reads that are needed now
            while (touch.size() > 4096) {
                touch.pop_front();
            }
        }
        cv.notify_all();
    }
};

llama_row_pool & llama_get_row_pool() {
    // enough reads in flight for a SATA or NVMe queue
    static llama_row_pool pool(32);
    return pool;
}

} // namespace

static bool llama_rows_in_host_memory(const ggml_tensor * tensor) {
    return tensor && tensor->data && tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer) && ggml_is_matrix(tensor);
}

// the first and the last byte of each row, for a row that crosses a page boundary
static std::vector<const char *> llama_row_bytes(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    const auto * base = (const char *) tensor->data;
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    std::vector<const char *> ptrs;
    ptrs.reserve(2*n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        if (rows[i] < 0 || rows[i] >= tensor->ne[1]) {
            continue;
        }
        const char * p = base + (size_t) rows[i]*tensor->nb[1];
        ptrs.push_back(p);
        ptrs.push_back(p + row_bytes - 1);
    }
    return ptrs;
}

void llama_fault_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    if (!llama_rows_in_host_memory(tensor) || n_rows == 0) {
        return;
    }

    const std::vector<const char *> ptrs = llama_row_bytes(tensor, rows, n_rows);
    const std::function<void(size_t)> touch = [&](size_t i) {
        (void) *(const volatile char *) ptrs[i];
    };
    llama_get_row_pool().run(ptrs.size(), touch);
}

void llama_prefetch_rows_async(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    if (!llama_rows_in_host_memory(tensor) || n_rows == 0) {
        return;
    }

    const std::vector<const char *> ptrs = llama_row_bytes(tensor, rows, n_rows);
    llama_get_row_pool().add_touch(ptrs.data(), ptrs.size());
}

void llama_prefetch_rows_cancel() {
    llama_get_row_pool().cancel_touch();
}

time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

void llama_log_get(ggml_log_callback * log_callback, void ** user_data) {
    ggml_log_get(log_callback, user_data);
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
    g_logger_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void llama_clear_tensor_data(ggml_tensor * t, size_t offset, size_t size) {
    static const std::vector<uint8_t> zeros(1024*1024, 0);

    // not all backend buffers implement ggml_backend_tensor_memset(), so write zeros instead
    // TODO: make this a generic fallback in `ggml_backend_tensor_memset` when `set_tensor` is available
    for (size_t ofs = 0; ofs < size; ofs += zeros.size()) {
        ggml_backend_tensor_set(t, zeros.data(), offset + ofs, std::min(size - ofs, zeros.size()));
    }
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, ne.at(i));
    }
    return buf;
}

std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, size_t i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i, size_t max_len) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                const size_t arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::string res = "[";
                size_t j = 0;
                for (; j < arr_n && res.size() <= max_len; j++) {
                    if (j > 0) {
                        res += ", ";
                    }
                    if (arr_type == GGUF_TYPE_STRING) {
                        const char * s = gguf_get_arr_str(ctx_gguf, i, j);
                        const size_t need = res.size() + 2 * strlen(s) + 2;
                        if (res.capacity() < need) {
                            res.reserve(std::max(need, 2 * res.capacity()));
                        }
                        res += '"';
                        for (; *s; s++) {
                            if (*s == '\\' || *s == '"') {
                                res += '\\';
                            }
                            res += *s;
                        }
                        res += '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        res += "???";
                    } else {
                        res += gguf_data_to_str(arr_type, data, j);
                    }
                }
                if (j == arr_n) {
                    res += "]";
                }
                return res;
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}
