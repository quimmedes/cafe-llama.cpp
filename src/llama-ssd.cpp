#include "llama-ssd.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

#if !defined(O_DIRECT)
#define O_DIRECT 0
#endif
#if !defined(O_CLOEXEC)
#define O_CLOEXEC 0
#endif

static void * ssd_alloc_aligned(size_t align, size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, align);
#else
    void * p = nullptr;
    if (posix_memalign(&p, align, size) != 0) {
        return nullptr;
    }
    return p;
#endif
}

static void ssd_free_aligned(void * p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

llama_ssd_reader::~llama_ssd_reader() {
    close();
}

bool llama_ssd_reader::open(const std::string & path, int lanes, bool direct, size_t align, size_t bounce_cap) {
    close();

    if (lanes <= 0) {
        lanes = 1;
    }
    if (align == 0) {
        align = 4096;
    }
    align_ = align;
    path_  = path;

    for (int i = 0; i < lanes; i++) {
#if defined(_WIN32)
        // unsupported on this platform: fall back to buffered
        int fd = -1;
#else
        int fd = -1;
        if (direct) {
            fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        }
        if (fd == -1) {
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
#endif
        if (fd == -1) {
            LLAMA_LOG_WARN("%s: failed to open %s for streamed reads\n", __func__, path.c_str());
            close();
            return false;
        }
        fds_.push_back(fd);
        fds_buf_.push_back(fd);
        bounces_.push_back(nullptr);
        bounce_sz_.push_back(0);
    }

    direct_ = direct;
#if defined(_WIN32)
    direct_ = false;
#endif

    // file size
#if defined(_WIN32)
    struct _stat64 st;
    if (_fstat64(fds_[0], &st) != 0) {
        close();
        return false;
    }
    fsize_ = (uint64_t) st.st_size;
#else
    struct stat st;
    if (fstat(fds_[0], &st) != 0) {
        close();
        return false;
    }
    fsize_ = (uint64_t) st.st_size;
    if (direct) {
        // st_blksize is the alignment O_DIRECT wants
        align_ = (size_t) st.st_blksize;
    }
#endif

    aligned_ = direct_;

    // pre-allocate the bounce buffers so the hot path never allocates
    if (aligned_ && bounce_cap > 0) {
        const size_t sz = (bounce_cap + align_ - 1) & ~(align_ - 1);
        for (size_t i = 0; i < fds_.size(); i++) {
            bounces_[i] = ssd_alloc_aligned(align_, sz);
            if (!bounces_[i]) {
                close();
                return false;
            }
            bounce_sz_[i] = sz;
        }
    }

    LLAMA_LOG_INFO("%s: %s: %d lane(s), %s I/O, align %zu\n",
            __func__, path.c_str(), (int) fds_.size(), direct_ ? "direct" : "buffered", align_);

    return true;
}

void llama_ssd_reader::close() {
    for (size_t i = 0; i < bounces_.size(); i++) {
        if (bounces_[i]) {
            ssd_free_aligned(bounces_[i]);
        }
    }
    fds_.clear();
    fds_buf_.clear();
    bounces_.clear();
    bounce_sz_.clear();
    fsize_   = 0;
    direct_  = false;
    aligned_ = false;
}

long long llama_ssd_reader::read(int lane, void * dst, uint64_t off, uint64_t nbytes) {
    if (nbytes == 0) {
        return 0;
    }
    if (lane < 0 || lane >= (int) fds_.size()) {
        return -1;
    }

    const uint64_t end = (fsize_ && off + nbytes > fsize_) ? fsize_ : off + nbytes;
    if (end <= off) {
        return 0;
    }

#if defined(_WIN32)
    // buffered fallback: positioned read via _lseeki64 + _read
    {
        uint64_t done = 0;
        const int fd = fds_[lane];
        while (off + done < end) {
            _lseeki64(fd, (long long) (off + done), SEEK_SET);
            const size_t want = (size_t) (end - off - done);
            const int got = _read(fd, (char *) dst + done, want > 0x7fffffff ? 0x7fffffff : (unsigned) want);
            if (got <= 0) {
                return -1;
            }
            done += (uint64_t) got;
        }
        return (long long) done;
    }
#else
    if (!aligned_) {
        uint64_t done = 0;
        while (off + done < end) {
            const ssize_t got = pread(fds_[lane], (char *) dst + done, (size_t) (end - off - done), (off_t) (off + done));
            if (got <= 0) {
                return -1;
            }
            done += (uint64_t) got;
        }
        return (long long) done;
    }

    const uint64_t a0  = off & ~(uint64_t) (align_ - 1);
    const uint64_t a1  = (end + align_ - 1) & ~(uint64_t) (align_ - 1);
    const size_t   len = (size_t) (a1 - a0);

    if (bounce_sz_[lane] < len) {
        if (bounces_[lane]) {
            ssd_free_aligned(bounces_[lane]);
        }
        bounces_[lane] = nullptr;
        bounce_sz_[lane] = 0;
        bounces_[lane] = ssd_alloc_aligned(align_, len);
        if (!bounces_[lane]) {
            return -1;
        }
        bounce_sz_[lane] = len;
    }

    char * b = (char *) bounces_[lane];
    const uint64_t read_end = (fsize_ && a1 > fsize_) ? fsize_ : a1;
    const uint64_t bulk_end = read_end & ~(uint64_t) (align_ - 1);

    for (uint64_t a = a0; a < bulk_end; ) {
        const ssize_t got = pread(fds_[lane], b + (a - a0), (size_t) (bulk_end - a), (off_t) a);
        if (got <= 0) {
            return -1;
        }
        a += (uint64_t) got;
    }
    if (bulk_end < read_end) {
        // sub-alignment EOF tail via a buffered fd
        for (uint64_t a = bulk_end; a < read_end; ) {
            const ssize_t got = pread(fds_buf_[lane], b + (a - a0), (size_t) (read_end - a), (off_t) a);
            if (got <= 0) {
                return -1;
            }
            a += (uint64_t) got;
        }
    }

    memcpy(dst, b + (off - a0), (size_t) (end - off));
    return (long long) (read_end - a0);
#endif
}

static void ssd_touch_range(const void * ptr, size_t len) {
    if (!ptr || len == 0) {
        return;
    }
    const size_t page_sz = 4096;
    const volatile char * p = (const volatile char *) ptr;
    char sum = 0;
    for (size_t off = 0; off < len; off += page_sz) {
        sum += p[off];
    }
    (void) sum;
}

void llama_ssd_expert_meta::get_expert_slice(int e, void *& ptr, size_t & len, int tensor_idx) const {
    ptr = nullptr;
    len = 0;
    if (e < 0 || e >= n_expert) {
        return;
    }
    switch (tensor_idx) {
        case 0:
            if (gate_exps && gate_exps->data) {
                ptr = (char *) gate_exps->data + e * stride_gate;
                len = stride_gate;
            }
            break;
        case 1:
            if (up_exps && up_exps->data) {
                ptr = (char *) up_exps->data + e * stride_up;
                len = stride_up;
            }
            break;
        case 2:
            if (down_exps && down_exps->data) {
                ptr = (char *) down_exps->data + e * stride_down;
                len = stride_down;
            }
            break;
        case 3:
            if (gate_up_exps && gate_up_exps->data) {
                ptr = (char *) gate_up_exps->data + e * stride_gate_up;
                len = stride_gate_up;
            }
            break;
        default:
            break;
    }
}

void llama_ssd_expert_predictor::init(int n_streamed_layers, const std::vector<int> & n_experts,
                                      const std::vector<int> & n_experts_used, int hot_capacity) {
    decay_ = 0.95f;
    hot_capacity_ = std::max(1, hot_capacity);
    n_experts_ = n_experts;
    stats_.resize(n_streamed_layers);
    hot_sets_.resize(n_streamed_layers);
    predicted_next_.resize(n_streamed_layers);

    for (int il = 0; il < n_streamed_layers; ++il) {
        const int ne = n_experts[il];
        stats_[il].assign(ne, expert_stat{});
        int init_k = std::min(ne, hot_capacity_);
        hot_sets_[il].clear();
        for (int e = 0; e < init_k; ++e) {
            hot_sets_[il].push_back(e);
        }
        predicted_next_[il] = hot_sets_[il];
    }
    GGML_UNUSED(n_experts_used);
}

void llama_ssd_expert_predictor::record_usage(int il, const int32_t * ids, size_t n_ids, uint64_t step) {
    if (il < 0 || il >= (int) stats_.size() || n_ids == 0) {
        return;
    }
    const int ne = n_experts_[il];
    auto & layer_stats = stats_[il];

    for (int e = 0; e < ne; ++e) {
        layer_stats[e].hotness *= decay_;
    }

    for (size_t i = 0; i < n_ids; ++i) {
        int id = ids[i];
        if (id >= 0 && id < ne) {
            layer_stats[id].hotness += (1.0f - decay_);
            layer_stats[id].last_used_step = step;
            layer_stats[id].hit_count++;
        }
    }

    std::vector<int> ranking(ne);
    for (int e = 0; e < ne; ++e) {
        ranking[e] = e;
    }
    std::sort(ranking.begin(), ranking.end(), [&](int a, int b) {
        if (layer_stats[a].hotness != layer_stats[b].hotness) {
            return layer_stats[a].hotness > layer_stats[b].hotness;
        }
        return layer_stats[a].last_used_step > layer_stats[b].last_used_step;
    });

    int k = std::min(ne, hot_capacity_);
    hot_sets_[il].clear();
    for (int i = 0; i < k; ++i) {
        hot_sets_[il].push_back(ranking[i]);
    }

    std::vector<int32_t> next_pred = hot_sets_[il];
    for (size_t i = 0; i < n_ids; ++i) {
        int id = ids[i];
        if (id >= 0 && id < ne && !is_hot(il, id)) {
            next_pred.push_back(id);
        }
    }
    predicted_next_[il] = std::move(next_pred);
}

const std::vector<int32_t> & llama_ssd_expert_predictor::get_hot_set(int il) const {
    static const std::vector<int32_t> empty;
    return (il >= 0 && il < (int) hot_sets_.size()) ? hot_sets_[il] : empty;
}

const std::vector<int32_t> & llama_ssd_expert_predictor::get_predicted(int il) const {
    static const std::vector<int32_t> empty;
    return (il >= 0 && il < (int) predicted_next_.size()) ? predicted_next_[il] : empty;
}

bool llama_ssd_expert_predictor::is_hot(int il, int e) const {
    if (il < 0 || il >= (int) hot_sets_.size()) {
        return false;
    }
    const auto & hs = hot_sets_[il];
    return std::find(hs.begin(), hs.end(), e) != hs.end();
}

float llama_ssd_expert_predictor::get_hotness(int il, int e) const {
    if (il < 0 || il >= (int) stats_.size() || e < 0 || e >= (int) stats_[il].size()) {
        return 0.0f;
    }
    return stats_[il][e].hotness;
}

struct llama_ssd_expert_cache::impl {
    llama_ssd_expert_predictor predictor;
    std::vector<llama_ssd_expert_meta> layers;
    std::vector<std::vector<bool>> is_loaded;

    int hot_slots_per_layer = 4;
    size_t max_cache_bytes = 0;
    size_t resident_bytes = 0;
    bool predict_enabled = true;
    bool use_mlock = false;
    int io_threads = 0;

    uint64_t current_step = 0;

    std::thread worker;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<std::pair<int, int>> prefetch_queue;
    bool stop_worker = false;

    ~impl() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stop_worker = true;
                queue_cv.notify_all();
            }
            worker.join();
        }
        if (use_mlock) {
            for (size_t il = 0; il < layers.size(); ++il) {
                for (int e = 0; e < layers[il].n_expert; ++e) {
                    if (is_loaded[il][e]) {
                        for (int t = 0; t < 4; ++t) {
                            void * ptr = nullptr;
                            size_t len = 0;
                            layers[il].get_expert_slice(e, ptr, len, t);
                            if (ptr && len > 0) {
                                unlock_slice(ptr, len);
                            }
                        }
                    }
                }
            }
        }
    }

    void lock_slice(void * ptr, size_t len) {
        if (!ptr || len == 0) {
            return;
        }
#if defined(_WIN32)
        VirtualLock(ptr, len);
#elif defined(_POSIX_MEMLOCK_RANGE)
        mlock(ptr, len);
#endif
    }

    void unlock_slice(void * ptr, size_t len) {
        if (!ptr || len == 0) {
            return;
        }
#if defined(_WIN32)
        VirtualUnlock(ptr, len);
#elif defined(_POSIX_MEMLOCK_RANGE)
        munlock(ptr, len);
#endif
    }

    void advise_slice(void * ptr, size_t len, int advice) {
        if (!ptr || len == 0) {
            return;
        }
#if defined(__linux__) || defined(__APPLE__) || defined(_POSIX_MAPPED_FILES)
        posix_madvise(ptr, len, advice);
#else
        GGML_UNUSED(advice);
#endif
    }

    void hot_load(int il, int e) {
        if (il < 0 || il >= (int) layers.size()) {
            return;
        }
        const auto & meta = layers[il];
        if (e < 0 || e >= meta.n_expert || is_loaded[il][e]) {
            return;
        }

        for (int t = 0; t < 4; ++t) {
            void * ptr = nullptr;
            size_t len = 0;
            meta.get_expert_slice(e, ptr, len, t);
            if (ptr && len > 0) {
#if defined(POSIX_MADV_WILLNEED)
                advise_slice(ptr, len, POSIX_MADV_WILLNEED);
#endif
                ssd_touch_range(ptr, len);
                if (use_mlock) {
                    lock_slice(ptr, len);
                }
            }
        }
        is_loaded[il][e] = true;
        resident_bytes += meta.expert_bytes;
    }

    void evict(int il, int e) {
        if (il < 0 || il >= (int) layers.size()) {
            return;
        }
        const auto & meta = layers[il];
        if (e < 0 || e >= meta.n_expert || !is_loaded[il][e]) {
            return;
        }

        for (int t = 0; t < 4; ++t) {
            void * ptr = nullptr;
            size_t len = 0;
            meta.get_expert_slice(e, ptr, len, t);
            if (ptr && len > 0) {
                if (use_mlock) {
                    unlock_slice(ptr, len);
                }
#if defined(POSIX_MADV_DONTNEED)
                advise_slice(ptr, len, POSIX_MADV_DONTNEED);
#elif defined(POSIX_MADV_RANDOM)
                advise_slice(ptr, len, POSIX_MADV_RANDOM);
#endif
            }
        }
        is_loaded[il][e] = false;
        if (resident_bytes >= meta.expert_bytes) {
            resident_bytes -= meta.expert_bytes;
        } else {
            resident_bytes = 0;
        }
    }

    void sync_hot_set(int il) {
        if (il < 0 || il >= (int) layers.size()) {
            return;
        }
        const auto & hot = predictor.get_hot_set(il);
        std::vector<bool> desired(layers[il].n_expert, false);
        for (int id : hot) {
            if (id >= 0 && id < layers[il].n_expert) {
                desired[id] = true;
                hot_load(il, id);
            }
        }
        for (int e = 0; e < layers[il].n_expert; ++e) {
            if (!desired[e] && is_loaded[il][e]) {
                evict(il, e);
            }
        }
    }

    void prefetch(int il, const int32_t * ids, size_t n_ids) {
        if (il < 0 || il >= (int) layers.size() || n_ids == 0) {
            return;
        }
        if (io_threads > 0) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            for (size_t i = 0; i < n_ids; ++i) {
                prefetch_queue.push({il, ids[i]});
            }
            queue_cv.notify_one();
        } else {
            for (size_t i = 0; i < n_ids; ++i) {
                int id = ids[i];
                if (id >= 0 && id < layers[il].n_expert && !is_loaded[il][id]) {
                    for (int t = 0; t < 4; ++t) {
                        void * ptr = nullptr;
                        size_t len = 0;
                        layers[il].get_expert_slice(id, ptr, len, t);
                        if (ptr && len > 0) {
#if defined(POSIX_MADV_WILLNEED)
                            advise_slice(ptr, len, POSIX_MADV_WILLNEED);
#endif
                        }
                    }
                }
            }
        }
    }

    void worker_loop() {
        while (true) {
            std::pair<int, int> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&]() { return stop_worker || !prefetch_queue.empty(); });
                if (stop_worker && prefetch_queue.empty()) {
                    break;
                }
                task = prefetch_queue.front();
                prefetch_queue.pop();
            }
            hot_load(task.first, task.second);
        }
    }

    int find_streamed_layer(int model_il) const {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].il == model_il) {
                return (int) i;
            }
        }
        return -1;
    }

    bool init(const llama_model & model, const llama_model_params & params) {
        predict_enabled = params.ssd_predict;
        use_mlock = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        io_threads = params.ssd_io_threads;

        layers.clear();
        std::vector<int> n_experts;
        std::vector<int> n_experts_used;

        const int n_stream = params.ssd_n_streaming;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            if (n_stream >= 0 && (int) il >= n_stream) {
                break;
            }
            const auto & l = model.layers[il];
            if (!l.ffn_gate_exps && !l.ffn_gate_up_exps) {
                continue;
            }
            llama_ssd_expert_meta meta;
            meta.il = (int) il;
            meta.gate_exps = l.ffn_gate_exps;
            meta.up_exps = l.ffn_up_exps;
            meta.down_exps = l.ffn_down_exps;
            meta.gate_up_exps = l.ffn_gate_up_exps;

            meta.n_expert = meta.gate_exps ? (int) meta.gate_exps->ne[2]
                          : (meta.gate_up_exps ? (int) meta.gate_up_exps->ne[2] : 0);
            meta.n_expert_used = (int) model.hparams.n_expert_used_max();

            meta.stride_gate = meta.gate_exps ? meta.gate_exps->nb[2] : 0;
            meta.stride_up = meta.up_exps ? meta.up_exps->nb[2] : 0;
            meta.stride_down = meta.down_exps ? meta.down_exps->nb[2] : 0;
            meta.stride_gate_up = meta.gate_up_exps ? meta.gate_up_exps->nb[2] : 0;

            meta.expert_bytes = meta.stride_gate + meta.stride_up + meta.stride_down + meta.stride_gate_up;

            if (meta.n_expert > 0 && meta.expert_bytes > 0) {
                layers.push_back(meta);
                n_experts.push_back(meta.n_expert);
                n_experts_used.push_back(meta.n_expert_used);
            }
        }

        if (layers.empty()) {
            return false;
        }

        if (params.ssd_cache_slots > 0) {
            hot_slots_per_layer = params.ssd_cache_slots;
        } else if (params.ssd_cache_mb > 0) {
            max_cache_bytes = (size_t) params.ssd_cache_mb * 1024 * 1024;
            size_t bytes_per_layer = max_cache_bytes / layers.size();
            size_t avg_exp_bytes = layers[0].expert_bytes;
            hot_slots_per_layer = avg_exp_bytes > 0 ? (int) (bytes_per_layer / avg_exp_bytes) : 4;
        } else {
            int n_exp = layers[0].n_expert;
            int n_used = layers[0].n_expert_used;
            hot_slots_per_layer = std::min(n_exp, std::max(2 * n_used, n_exp / 4));
        }
        hot_slots_per_layer = std::max(1, hot_slots_per_layer);

        if (max_cache_bytes == 0) {
            for (const auto & meta : layers) {
                max_cache_bytes += meta.expert_bytes * hot_slots_per_layer;
            }
        }

        is_loaded.resize(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            is_loaded[il].assign(layers[il].n_expert, false);
        }

        predictor.init((int) layers.size(), n_experts, n_experts_used, hot_slots_per_layer);

        if (io_threads > 0) {
            worker = std::thread(&impl::worker_loop, this);
        }

        const int init_warm = std::min(hot_slots_per_layer, std::max(1, 2 * layers[0].n_expert_used));
        for (size_t il = 0; il < layers.size(); ++il) {
            for (int e = 0; e < init_warm; ++e) {
                hot_load((int) il, e);
            }
        }

        LLAMA_LOG_INFO("%s: SSD expert cache initialized: %d streamed layers, %d hot slots/layer (%zu MiB resident budget), prediction %s, %d I/O threads\n",
            __func__, (int) layers.size(), hot_slots_per_layer, max_cache_bytes / (1024 * 1024),
            predict_enabled ? "enabled" : "disabled", io_threads);

        return true;
    }

    void on_step_start(uint64_t step) {
        current_step = step;
        if (!predict_enabled || layers.empty()) {
            return;
        }
        const auto & pred = predictor.get_predicted(0);
        if (!pred.empty()) {
            prefetch(0, pred.data(), pred.size());
        }
    }

    void on_experts_used(int model_il, const int32_t * ids, size_t n_ids) {
        int idx = find_streamed_layer(model_il);
        if (idx < 0) {
            return;
        }

        predictor.record_usage(idx, ids, n_ids, current_step);
        sync_hot_set(idx);

        if (predict_enabled && idx + 1 < (int) layers.size()) {
            const auto & next_pred = predictor.get_predicted(idx + 1);
            if (!next_pred.empty()) {
                prefetch(idx + 1, next_pred.data(), next_pred.size());
            }
        }
    }

    void on_step_finish(uint64_t step) {
        current_step = step;
        if (!predict_enabled || layers.empty()) {
            return;
        }
        for (size_t i = 0; i < std::min<size_t>(2, layers.size()); ++i) {
            const auto & pred = predictor.get_predicted((int) i);
            if (!pred.empty()) {
                prefetch((int) i, pred.data(), pred.size());
            }
        }
    }
};

llama_ssd_expert_cache::llama_ssd_expert_cache() : pimpl_(std::make_unique<impl>()) {}
llama_ssd_expert_cache::~llama_ssd_expert_cache() = default;

bool llama_ssd_expert_cache::init(const llama_model & model, const llama_model_params & params) {
    return pimpl_->init(model, params);
}

void llama_ssd_expert_cache::on_step_start(uint64_t step) {
    pimpl_->on_step_start(step);
}

void llama_ssd_expert_cache::on_experts_used(int il, const int32_t * ids, size_t n_ids) {
    pimpl_->on_experts_used(il, ids, n_ids);
}

void llama_ssd_expert_cache::on_step_finish(uint64_t step) {
    pimpl_->on_step_finish(step);
}

void llama_ssd_expert_cache::hot_load_expert(int il, int e) {
    pimpl_->hot_load(il, e);
}

void llama_ssd_expert_cache::evict_expert(int il, int e) {
    pimpl_->evict(il, e);
}

void llama_ssd_expert_cache::prefetch_experts(int il, const int32_t * ids, size_t n_ids) {
    pimpl_->prefetch(il, ids, n_ids);
}

bool llama_ssd_expert_cache::is_loaded(int il, int e) const {
    int idx = pimpl_->find_streamed_layer(il);
    return idx >= 0 && pimpl_->is_loaded[idx][e];
}

int llama_ssd_expert_cache::hot_slots_per_layer() const {
    return pimpl_->hot_slots_per_layer;
}

size_t llama_ssd_expert_cache::total_budget_bytes() const {
    return pimpl_->max_cache_bytes;
}

size_t llama_ssd_expert_cache::resident_bytes() const {
    return pimpl_->resident_bytes;
}

bool llama_ssd_expert_cache::predict_enabled() const {
    return pimpl_->predict_enabled;
}
