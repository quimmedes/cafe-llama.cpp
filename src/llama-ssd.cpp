#include "llama-ssd.h"

#include "llama-impl.h"

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
