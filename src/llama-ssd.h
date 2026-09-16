#pragma once

// Parallel-lane positioned file reader for streamed expert slices.
//
// One reader owns N lane fds and N bounce buffers, so N threads can read
// distinct byte ranges of the same file concurrently without contending.
// Where the platform's direct mode rejects unaligned access (O_DIRECT,
// FILE_FLAG_NO_BUFFERING), a direct read rounds its window to the read
// alignment, pulls it into the lane's bounce buffer, and copies the requested
// interior out. The O_DIRECT request is verified once at open and silently
// falls back to buffered I/O when the platform or the storage refuses it, so
// a caller never has to reason about the storage; direct() reports what is
// actually in effect.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_ssd_reader {
    llama_ssd_reader() = default;
    ~llama_ssd_reader();

    llama_ssd_reader(const llama_ssd_reader &) = delete;
    llama_ssd_reader & operator=(const llama_ssd_reader &) = delete;

    // Open `path` with `lanes` independent fds and an aligned bounce per lane.
    bool open(const std::string & path, int lanes, bool direct, size_t align, size_t bounce_cap);
    void close();

    bool is_open() const { return !fds_.empty(); }
    bool direct() const { return direct_; }
    uint64_t file_size() const { return fsize_; }
    int lanes() const { return (int) fds_.size(); }

    // Read `nbytes` at file offset `off` into `dst`, on `lane` (0 <= lane < lanes()).
    // Returns the number of bytes actually pulled from the drive (the aligned
    // window where direct mode requires one), or -1 on error.
    long long read(int lane, void * dst, uint64_t off, uint64_t nbytes);

private:
    std::vector<int>    fds_;
    std::vector<int>    fds_buf_;   // buffered fd per lane, for the sub-alignment EOF tail
    std::vector<void *> bounces_;
    std::vector<size_t> bounce_sz_;
    size_t   align_    = 4096;
    uint64_t fsize_    = 0;
    bool     direct_   = false;     // cache bypass actually in effect
    bool     aligned_  = false;     // direct_ and the mode rejects unaligned reads
    std::string path_;
};
