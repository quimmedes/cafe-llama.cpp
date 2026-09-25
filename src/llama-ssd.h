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
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_model;
struct llama_model_params;

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

// Metadata for routed MoE experts of a streamed layer
struct llama_ssd_expert_meta {
    int il = -1;
    int n_expert = 0;
    int n_expert_used = 0;

    struct ggml_tensor * gate_exps    = nullptr;
    struct ggml_tensor * up_exps      = nullptr;
    struct ggml_tensor * down_exps    = nullptr;
    struct ggml_tensor * gate_up_exps = nullptr;

    size_t stride_gate    = 0;
    size_t stride_up      = 0;
    size_t stride_down    = 0;
    size_t stride_gate_up = 0;
    size_t expert_bytes   = 0;

    void get_expert_slice(int e, void *& ptr, size_t & len, int tensor_idx) const;
};

// Predictor for active and hot MoE experts
class llama_ssd_expert_predictor {
public:
    struct expert_stat {
        float hotness = 0.0f;
        uint64_t last_used_step = 0;
        uint32_t hit_count = 0;
    };

    void init(int n_streamed_layers, const std::vector<int> & n_experts, const std::vector<int> & n_experts_used, int hot_capacity);

    void record_usage(int il, const int32_t * ids, size_t n_ids, uint64_t step);

    const std::vector<int32_t> & get_hot_set(int il) const;
    const std::vector<int32_t> & get_predicted(int il) const;
    bool is_hot(int il, int e) const;
    float get_hotness(int il, int e) const;

private:
    float decay_ = 0.95f;
    int hot_capacity_ = 4;
    std::vector<int> n_experts_;
    std::vector<std::vector<expert_stat>> stats_;
    std::vector<std::vector<int32_t>> hot_sets_;
    std::vector<std::vector<int32_t>> predicted_next_;
};

// Hot-loading expert cache for -nssd
class llama_ssd_expert_cache {
public:
    llama_ssd_expert_cache();
    ~llama_ssd_expert_cache();

    llama_ssd_expert_cache(const llama_ssd_expert_cache &) = delete;
    llama_ssd_expert_cache & operator=(const llama_ssd_expert_cache &) = delete;

    bool init(const llama_model & model, const llama_model_params & params);

    void on_step_start(uint64_t step);
    void on_experts_used(int il, const int32_t * ids, size_t n_ids);
    void on_step_finish(uint64_t step);

    void hot_load_expert(int il, int e);
    void evict_expert(int il, int e);
    void prefetch_experts(int il, const int32_t * ids, size_t n_ids);

    bool is_loaded(int il, int e) const;
    int hot_slots_per_layer() const;
    size_t total_budget_bytes() const;
    size_t resident_bytes() const;
    bool predict_enabled() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
