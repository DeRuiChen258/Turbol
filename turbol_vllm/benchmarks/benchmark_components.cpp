// ============================================================================
// benchmark_components.cpp — micro-benchmark suite for TurboRL core components.
//
//   Measures steady-state throughput (ops/sec) for the hot paths of an RL /
//   RLHF training loop:
//     1. GPUEnvironment::Step      — vectorized env stepping
//     2. RingBuffer::PushBatch     — experience storage
//     3. RingBuffer::Sample        — uniform replay sampling
//     4. PolicyEngine::Forward     — actor forward + log-prob (libtorch)
//     5. PolicyEngine::Update      — actor/critic backward + Adam step
//     6. GAE                       — generalized advantage estimation (CUDA)
//
//   Usage:
//     ./benchmark_components [--num-envs N] [--obs-dim D] [--act-dim A]
//                            [--steps N] [--batch B] [--device cpu|cuda]
//
//   Every benchmark runs a warm-up phase, then a timed phase, and reports
//   mean latency and throughput. CPU is the default device so the suite runs
//   even without a GPU; CUDA paths are exercised when a device is present.
// ============================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "turbol/common.hpp"
#include "turbol/env/gpu_environment.hpp"
#include "turbol/policy/policy_engine.hpp"
#include "turbol/replay_buffer/ring_buffer.hpp"

#ifdef TURBORL_HAS_CUDA
// GAE kernel lives in src/cuda/gae.cu and is exposed as a plain C symbol.
extern "C" cudaError_t gae_compute(const float* rewards, const float* values,
                                   const bool* dones, float* advantages,
                                   float* returns, int T, int num_envs,
                                   float gamma, float lambda, cudaStream_t stream);
#endif

using namespace turborl;
using namespace turborl::env;
using namespace turborl::policy;
using namespace turborl::replay_buffer;

namespace {

// ---------------------------------------------------------------------------
// Small timing harness
// ---------------------------------------------------------------------------
struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0 = Clock::now();
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
};

// Run `fn` once for warm-up, then `iters` timed iterations. Returns the mean
// wall-clock time per iteration in milliseconds.
template <typename Fn>
double TimeLoop(Fn&& fn, int iters, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) fn();
    Timer timer;
    for (int i = 0; i < iters; ++i) fn();
    return timer.elapsed_ms() / static_cast<double>(iters);
}

void Report(const std::string& name, double ms_per_op, double ops,
            const std::string& unit) {
    std::printf("  %-28s %10.3f ms/op   %12.1f %s\n", name.c_str(),
                ms_per_op, ops, unit.c_str());
}

struct Options {
    int num_envs = 4096;
    int obs_dim = 16;
    int act_dim = 4;
    int steps = 2000;
    int batch = 256;
    int capacity = 1 << 20;  // 1M transitions
    Device device = Device::CPU();
};

void FillUniform(Tensor* t, float lo = -1.0f, float hi = 1.0f) {
    // Only safe on CPU tensors (host-side write).
    float* p = t->data<float>();
    const float span = hi - lo;
    const int64_t n = t->numel();
    for (int64_t i = 0; i < n; ++i) p[i] = lo + span * (static_cast<float>(i % 1000) / 999.0f);
}

void FillBool(Tensor* t, bool value = false) {
    // Host-side write; only safe on CPU tensors.
    bool* p = t->data<bool>();
    for (int64_t i = 0; i < t->numel(); ++i) p[i] = value;
}

// ---------------------------------------------------------------------------
// 1. Env step throughput
// ---------------------------------------------------------------------------
void BenchEnvStep(const Options& o) {
    GPUEnvironment env(o.num_envs, o.obs_dim, o.act_dim, o.device);
    env.Reset();

    Tensor action(Shape({o.num_envs, o.act_dim}), DataType::kFloat32, o.device);
    Tensor obs(Shape({o.num_envs, o.obs_dim}), DataType::kFloat32, o.device);
    Tensor reward(Shape({o.num_envs}), DataType::kFloat32, o.device);
    Tensor done(Shape({o.num_envs}), DataType::kFloat32, o.device);
    if (o.device.is_cpu()) FillUniform(&action);

    const int iters = o.steps;
    double ms = TimeLoop(
        [&]() { (void)env.Step(action, &obs, &reward, &done); }, iters);
    const double steps_per_s = 1000.0 / ms;
    const double env_steps_per_s = steps_per_s * o.num_envs;

    std::printf("\n[1/6] GPUEnvironment::Step (%d envs, %dx%d)\n", o.num_envs,
                o.obs_dim, o.act_dim);
    Report("env.Step", ms, steps_per_s, "steps/s");
    std::printf("        => aggregate throughput %12.1f env-steps/s\n",
                env_steps_per_s);
}

// ---------------------------------------------------------------------------
// 2 & 3. RingBuffer push / sample throughput
// ---------------------------------------------------------------------------
void BenchReplay(const Options& o) {
    RingBuffer buffer(o.capacity, o.obs_dim, o.act_dim, o.device);

    Tensor obs(Shape({o.batch, o.obs_dim}), DataType::kFloat32, o.device);
    Tensor act(Shape({o.batch, o.act_dim}), DataType::kFloat32, o.device);
    Tensor reward(Shape({o.batch}), DataType::kFloat32, o.device);
    Tensor done(Shape({o.batch}), DataType::kBool, o.device);
    if (o.device.is_cpu()) {
        FillUniform(&obs);
        FillUniform(&act);
        FillUniform(&reward);
        FillBool(&done);
    }

    // Warm the buffer so Sample() operates over a non-trivial valid range.
    for (int i = 0; i < 64; ++i)
        (void)buffer.PushBatch(obs, act, reward, done);

    const int push_iters = o.steps;
    double push_ms = TimeLoop(
        [&]() { (void)buffer.PushBatch(obs, act, reward, done); }, push_iters);

    Tensor s_obs, s_act, s_rew, s_done;
    const int sample_iters = o.steps;
    double sample_ms = TimeLoop(
        [&]() { (void)buffer.Sample(o.batch, &s_obs, &s_act, &s_rew, &s_done); },
        sample_iters);

    std::printf("\n[2/6] RingBuffer::PushBatch (%d transitions/batch)\n", o.batch);
    Report("PushBatch", push_ms, 1000.0 / push_ms, "batches/s");
    std::printf("        => aggregate throughput %12.1f transitions/s\n",
                (1000.0 / push_ms) * o.batch);

    std::printf("\n[3/6] RingBuffer::Sample (%d transitions/batch)\n", o.batch);
    Report("Sample", sample_ms, 1000.0 / sample_ms, "batches/s");
    std::printf("        => aggregate throughput %12.1f transitions/s\n",
                (1000.0 / sample_ms) * o.batch);
}

// ---------------------------------------------------------------------------
// 4 & 5. Policy forward / update throughput (libtorch)
// ---------------------------------------------------------------------------
void BenchPolicy(const Options& o) {
#ifndef TURBORL_LIBTORCH
    std::printf("\n[4/6] PolicyEngine: SKIPPED (libtorch disabled)\n");
    std::printf("[5/6] PolicyEngine: SKIPPED (libtorch disabled)\n");
    (void)o;
    return;
#else
    const int n = o.batch;
    PolicyEngine policy(o.obs_dim, o.act_dim, o.device);
    Status s = policy.Initialize();
    if (!s.ok()) {
        std::printf("\n[4/6] PolicyEngine: SKIPPED (%s)\n", s.message().c_str());
        std::printf("[5/6] PolicyEngine: SKIPPED\n");
        return;
    }

    Tensor obs(Shape({n, o.obs_dim}), DataType::kFloat32, o.device);
    if (o.device.is_cpu()) FillUniform(&obs);

    Tensor action, logprob;
    double fwd_ms = TimeLoop(
        [&]() { (void)policy.Forward(obs, &action, &logprob); }, o.steps);
    std::printf("\n[4/6] PolicyEngine::Forward (batch=%d)\n", n);
    Report("Forward", fwd_ms, 1000.0 / fwd_ms, "batches/s");
    std::printf("        => aggregate throughput %12.1f samples/s\n",
                (1000.0 / fwd_ms) * n);

    const int row = o.obs_dim + o.act_dim + 2;
    Tensor batch(Shape({n, row}), DataType::kFloat32, o.device);
    if (o.device.is_cpu()) FillUniform(&batch);

    double upd_ms = TimeLoop([&]() { (void)policy.Update(batch); }, o.steps);
    std::printf("\n[5/6] PolicyEngine::Update (batch=%d, row=%d)\n", n, row);
    Report("Update", upd_ms, 1000.0 / upd_ms, "batches/s");
    std::printf("        => aggregate throughput %12.1f samples/s\n",
                (1000.0 / upd_ms) * n);
#endif
}

// ---------------------------------------------------------------------------
// 6. GAE throughput (CUDA)
// ---------------------------------------------------------------------------
void BenchGAE(const Options& o) {
#ifdef TURBORL_HAS_CUDA
    if (GetCudaDeviceCount() == 0) {
        std::printf("\n[6/6] GAE: SKIPPED (no CUDA device)\n");
        return;
    }
    const int T = 256;          // trajectory length
    const int num_envs = o.num_envs;

    const size_t rf = sizeof(float) * T * num_envs;
    const size_t vf = sizeof(float) * (T + 1) * num_envs;
    const size_t bf = sizeof(bool) * T * num_envs;

    float *rewards, *values, *adv, *ret;
    bool* dones;
    cudaMalloc(&rewards, rf);
    cudaMalloc(&values, vf);
    cudaMalloc(&dones, bf);
    cudaMalloc(&adv, rf);
    cudaMalloc(&ret, rf);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch a warm-up pass so lazy init / cache effects don't pollute timing.
    for (int i = 0; i < 3; ++i)
        (void)gae_compute(rewards, values, dones, adv, ret, T, num_envs,
                          0.99f, 0.95f, stream);
    cudaDeviceSynchronize();

    // Time `iters` back-to-back launches and synchronize once at the end so the
    // measured wall-clock includes real GPU execution (amortized per call),
    // rather than just host-side launch overhead.
    const int iters = 200;
    Timer timer;
    for (int i = 0; i < iters; ++i)
        (void)gae_compute(rewards, values, dones, adv, ret, T, num_envs,
                          0.99f, 0.95f, stream);
    cudaDeviceSynchronize();
    const double ms = timer.elapsed_ms() / static_cast<double>(iters);

    const double steps = static_cast<double>(T) * num_envs;
    std::printf("\n[6/6] GAE (T=%d, envs=%d)\n", T, num_envs);
    Report("gae_compute", ms, 1000.0 / ms, "calls/s");
    std::printf("        => aggregate throughput %12.1f env-steps/s\n",
                (1000.0 / ms) * steps);

    cudaStreamDestroy(stream);
    cudaFree(rewards);
    cudaFree(values);
    cudaFree(dones);
    cudaFree(adv);
    cudaFree(ret);
#else
    (void)o;
    std::printf("\n[6/6] GAE: SKIPPED (CUDA disabled)\n");
#endif
}

void ParseArgs(int argc, char** argv, Options* o) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) {
            return i + 1 < argc && std::strcmp(argv[i], flag) == 0;
        };
        if (next("--num-envs")) o->num_envs = std::atoi(argv[++i]);
        else if (next("--obs-dim")) o->obs_dim = std::atoi(argv[++i]);
        else if (next("--act-dim")) o->act_dim = std::atoi(argv[++i]);
        else if (next("--steps")) o->steps = std::atoi(argv[++i]);
        else if (next("--batch")) o->batch = std::atoi(argv[++i]);
        else if (next("--device")) {
            if (std::strcmp(argv[++i], "cuda") == 0) o->device = Device::CUDA(0);
            else o->device = Device::CPU();
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    ParseArgs(argc, argv, &o);

    std::printf("=== TurboRL Component Micro-Benchmarks ===\n");
    std::printf("  device    : %s\n", o.device.ToString().c_str());
    std::printf("  num_envs  : %d\n", o.num_envs);
    std::printf("  obs/act   : %d / %d\n", o.obs_dim, o.act_dim);
    std::printf("  steps     : %d\n", o.steps);
    std::printf("  batch     : %d\n", o.batch);

    BenchEnvStep(o);
    BenchReplay(o);
    BenchPolicy(o);
    BenchGAE(o);

    std::printf("\nBenchmark completed.\n");
    return 0;
}
