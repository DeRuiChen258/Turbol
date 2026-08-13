// pybind11 C++ bindings — exposes the core C++ modules to Python.
//
// Exposes:
//   - Tensor (typed buffer view)
//   - GPUEnvironment
//   - RingBuffer
//   - PolicyEngine
//   - Config
//   - RolloutEngine
//   - RewardEngine
//   - DistributedTrainer
//
// Zero-copy principle: where a Python torch.Tensor and a turborl::Tensor share
// the same underlying buffer, we pass it via torch::from_blob without copying.
// The capsule (PyCapsule) mechanism ensures correct lifetime management.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "turbol/common.hpp"
#include "turbol/env/gpu_environment.hpp"
#include "turbol/replay_buffer/ring_buffer.hpp"
#include "turbol/policy/policy_engine.hpp"
#include "turbol/core/config.hpp"
#include "turbol/rollout/rollout_engine.hpp"
#include "turbol/reward/reward_engine.hpp"
#include "turbol/distributed/distributed_trainer.hpp"
#include "turbol/profiler/profiler.hpp"
#include "turbol/profiler/tensorboard_writer.hpp"

#ifdef TURBORL_LIBTORCH
#include <torch/torch.h>
#endif

namespace py = pybind11;
using namespace turborl;

// ============================================================================
// Helpers
// ============================================================================

// Parse a device string ("cpu", "cuda", "cuda:0", ...) into a Device.
static Device ParseDevice(const std::string& device_str) {
    if (device_str.find("cuda") != std::string::npos) {
        int id = 0;
        auto pos = device_str.find(':');
        if (pos != std::string::npos)
            id = std::stoi(device_str.substr(pos + 1));
        return Device::CUDA(id);
    }
    return Device::CPU();
}

// Convert a numpy array or a torch.Tensor into a Tensor (copies to device).
// (Reserved for future numpy/torch ingestion paths.)

// Create a numpy array view over a Tensor (no copy).
static py::array TensorToNumpy(const Tensor& t) {
    std::vector<ssize_t> shape;
    for (auto d : t.shape().dims) shape.push_back(static_cast<ssize_t>(d));

    std::string format;
    switch (t.dtype()) {
        case DataType::kFloat32: format = py::format_descriptor<float>::format();  break;
        case DataType::kFloat16: format = "e"; break;
        case DataType::kInt32:   format = py::format_descriptor<int32_t>::format(); break;
        case DataType::kInt64:   format = py::format_descriptor<int64_t>::format(); break;
        case DataType::kBool:    format = "?"; break;
        default:                 format = py::format_descriptor<float>::format();  break;
    }

    const ssize_t itemsize = static_cast<ssize_t>(GetDataTypeSize(t.dtype()));
    // Empty strides → pybind11 assumes C-contiguous layout.
    return py::array(py::buffer_info(const_cast<void*>(t.data()), itemsize, format,
                                     static_cast<ssize_t>(shape.size()), shape,
                                     std::vector<ssize_t>{}));
}

// Convert torch::Tensor → turborl::Tensor (copies if needed).
// (Reserved for future torch ingestion paths.)

// Create a torch::Tensor view over a turborl::Tensor (zero-copy if on CUDA).
static torch::Tensor TensorToTorch(const Tensor& t) {
#ifdef TURBORL_LIBTORCH
    c10::ScalarType st;
    switch (t.dtype()) {
        case DataType::kFloat32: st = c10::kFloat;  break;
        case DataType::kFloat16: st = c10::kHalf;   break;
        case DataType::kInt32:   st = c10::kInt;    break;
        case DataType::kInt64:   st = c10::kLong;   break;
        default:                  st = c10::kFloat;  break;
    }
    c10::DeviceType dt = t.is_cuda() ? c10::kCUDA : c10::kCPU;
    std::vector<int64_t> sizes;
    for (auto d : t.shape().dims) sizes.push_back(d);
    return torch::from_blob(const_cast<void*>(t.data()), sizes, st).to(dt).clone();
#else
    throw std::runtime_error("libtorch not available");
#endif
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(turbol_core, m) {
    m.doc() = "TurboRL core — C++ native bindings";

    // ---- Tensor ----
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init([](const std::vector<int64_t>& shape,
                         const std::string& dtype_str,
                         const std::string& device_str) {
            DataType dt = DataType::kFloat32;
            if (dtype_str == "float32") dt = DataType::kFloat32;
            else if (dtype_str == "float16") dt = DataType::kFloat16;
            else if (dtype_str == "int32")   dt = DataType::kInt32;
            else if (dtype_str == "int64")   dt = DataType::kInt64;
            Device d = Device::CPU();
            if (device_str.find("cuda") != std::string::npos) {
                int id = 0;
                auto pos = device_str.find(':');
                if (pos != std::string::npos)
                    id = std::stoi(device_str.substr(pos + 1));
                d = Device::CUDA(id);
            }
            return Tensor(Shape(shape), dt, d);
        }), py::arg("shape"), py::arg("dtype") = "float32",
           py::arg("device") = "cpu")
        .def("numel",         &Tensor::numel)
        .def("shape",         &Tensor::shape)
        .def("dtype",         &Tensor::dtype)
        .def("device",        &Tensor::device)
        .def("is_cuda",       &Tensor::is_cuda)
        .def("clone",         &Tensor::Clone)
        .def("to_numpy",      &TensorToNumpy)
        .def("to_torch",      &TensorToTorch)
        .def("to_device",     &Tensor::ToDevice);

    // ---- GPUEnvironment ----
    py::class_<env::GPUEnvironment>(m, "GPUEnvironment")
        .def(py::init([](int num_envs, int obs_dim, int act_dim,
                         const std::string& device) {
                 return env::GPUEnvironment(num_envs, obs_dim, act_dim,
                                            ParseDevice(device));
             }),
             py::arg("num_envs") = 1, py::arg("obs_dim") = 4,
             py::arg("act_dim") = 2,  py::arg("device") = "cpu")
        .def("reset",        &env::GPUEnvironment::Reset)
        .def("step",         [](env::GPUEnvironment& self,
                               const Tensor& action,
                               Tensor* obs, Tensor* reward, Tensor* done) {
            return self.Step(action, obs, reward, done);
        }, py::arg("action"), py::arg("obs"), py::arg("reward"),
           py::arg("done"), py::return_value_policy::reference_internal)
        .def("observe",      &env::GPUEnvironment::Observe)
        .def("num_envs",     &env::GPUEnvironment::NumEnvs)
        .def("obs_shape",    &env::GPUEnvironment::ObsShape)
        .def("act_shape",    &env::GPUEnvironment::ActionShape);

    // ---- RingBuffer ----
    py::class_<replay_buffer::RingBuffer>(m, "RingBuffer")
        .def(py::init([](int capacity, int obs_dim, int act_dim,
                         const std::string& device) {
                 return replay_buffer::RingBuffer(capacity, obs_dim, act_dim,
                                                  ParseDevice(device));
             }),
             py::arg("capacity"), py::arg("obs_dim"),
             py::arg("act_dim"),  py::arg("device") = "cpu")
        .def("push",        &replay_buffer::RingBuffer::Push)
        .def("push_batch",  &replay_buffer::RingBuffer::PushBatch)
        .def("sample",      [](replay_buffer::RingBuffer& self, int bs,
                               Tensor* o, Tensor* a, Tensor* r, Tensor* d) {
            return self.Sample(bs, o, a, r, d);
        }, py::arg("batch_size"), py::arg("obs"), py::arg("act"),
           py::arg("reward"), py::arg("done"), py::return_value_policy::reference_internal)
        .def("size",        &replay_buffer::RingBuffer::Size)
        .def("capacity",    &replay_buffer::RingBuffer::Capacity)
        .def("clear",       &replay_buffer::RingBuffer::Clear);

    // ---- PolicyEngine ----
    // NOTE: PolicyEngine is non-movable (holds torch::optim::Adam), so the init
    // factory must return a raw pointer rather than by value.
    py::class_<policy::PolicyEngine>(m, "PolicyEngine")
        .def(py::init([](int obs_dim, int act_dim, const std::string& device) {
                 return new policy::PolicyEngine(obs_dim, act_dim,
                                                 ParseDevice(device));
             }),
             py::arg("obs_dim") = 4, py::arg("act_dim") = 2,
             py::arg("device") = "cpu")
        .def("initialize",  &policy::PolicyEngine::Initialize)
        .def("forward",     [](policy::PolicyEngine& self, const Tensor& obs,
                               Tensor* action, Tensor* logprob) {
            return self.Forward(obs, action, logprob);
        }, py::arg("observation"), py::arg("action"), py::arg("logprob"),
           py::return_value_policy::reference_internal)
        .def("get_value",   [](policy::PolicyEngine& self, const Tensor& obs,
                                Tensor* value) {
            return self.GetValue(obs, value);
        }, py::arg("observation"), py::arg("value"),
           py::return_value_policy::reference_internal)
        .def("update",      &policy::PolicyEngine::Update);

    // ---- Config ----
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def("load",        &Config::Load)
        .def("save",        &Config::Save)
        .def("get_float",   [](const Config& self, const std::string& k, float def) {
            return self.Get<float>(k, def);
        }, py::arg("key"), py::arg("default"))
        .def("get_int",     [](const Config& self, const std::string& k, int def) {
            return self.Get<int>(k, def);
        }, py::arg("key"), py::arg("default"))
        .def("get_bool",    [](const Config& self, const std::string& k, bool def) {
            return self.Get<bool>(k, def);
        }, py::arg("key"), py::arg("default"))
        .def("get_str",     [](const Config& self, const std::string& k,
                                const std::string& def) {
            return self.Get<std::string>(k, def);
        }, py::arg("key"), py::arg("default"))
        .def("set",         [](Config& self, const std::string& k, float v) {
            self.Set(k, v);
        }, py::arg("key"), py::arg("value"))
        .def("set",         [](Config& self, const std::string& k, int v) {
            self.Set(k, v);
        }, py::arg("key"), py::arg("value"))
        .def("set",         [](Config& self, const std::string& k, bool v) {
            self.Set(k, v);
        }, py::arg("key"), py::arg("value"))
        .def("set",         [](Config& self, const std::string& k,
                                const std::string& v) {
            self.Set(k, v);
        }, py::arg("key"), py::arg("value"))
        .def("has",         &Config::Has);

    // ---- RolloutEngine ----
    py::class_<rollout::RolloutConfig>(m, "RolloutConfig")
        .def(py::init<>())
        .def_readwrite("inference_backend", &rollout::RolloutConfig::inference_backend)
        .def_readwrite("model_path",        &rollout::RolloutConfig::model_path)
        .def_readwrite("max_new_tokens",    &rollout::RolloutConfig::max_new_tokens)
        .def_readwrite("temperature",        &rollout::RolloutConfig::temperature)
        .def_readwrite("top_p",              &rollout::RolloutConfig::top_p)
        .def_readwrite("top_k",              &rollout::RolloutConfig::top_k);

    py::class_<rollout::RolloutEngine>(m, "RolloutEngine")
        .def(py::init<const rollout::RolloutConfig&>(),
             py::arg("config"))
        .def("initialize",   &rollout::RolloutEngine::Initialize)
        .def("shutdown",    &rollout::RolloutEngine::Shutdown)
        .def("generate",    [](rollout::RolloutEngine& self,
                               const std::string& prompt) {
            std::string out;
            auto s = self.Generate(prompt, &out);
            if (!s.ok()) throw std::runtime_error(s.message());
            return out;
        }, py::arg("prompt"))
        .def("generate_batch", [](rollout::RolloutEngine& self,
                                  const std::vector<std::string>& prompts) {
            std::vector<std::string> outs;
            auto s = self.GenerateBatch(prompts, &outs);
            if (!s.ok()) throw std::runtime_error(s.message());
            return outs;
        }, py::arg("prompts"))
        .def("is_vllm_available", &rollout::RolloutEngine::IsVLLMAvailable);

    // ---- RewardEngine ----
    py::class_<reward::RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("kl_coef",      &reward::RewardConfig::kl_coef)
        .def_readwrite("kl_target",     &reward::RewardConfig::kl_target)
        .def_readwrite("adaptive_kl",   &reward::RewardConfig::adaptive_kl)
        .def_readwrite("normalize",     &reward::RewardConfig::normalize)
        .def_readwrite("reward_clip",   &reward::RewardConfig::reward_clip)
        .def_readwrite("format_bonus",  &reward::RewardConfig::format_bonus)
        .def_readwrite("length_penalty",&reward::RewardConfig::length_penalty)
        .def_readwrite("target_length", &reward::RewardConfig::target_length)
        .def_readwrite("device",        &reward::RewardConfig::device);

    py::class_<reward::RewardResult>(m, "RewardResult")
        .def_readwrite("total",        &reward::RewardResult::total)
        .def_readwrite("model_scores",  &reward::RewardResult::model_scores)
        .def_readwrite("kl_penalties", &reward::RewardResult::kl_penalties)
        .def_readwrite("rule_scores",  &reward::RewardResult::rule_scores);

    py::class_<reward::RewardEngine>(m, "RewardEngine")
        .def(py::init<const reward::RewardConfig&>(), py::arg("config"))
        .def("initialize",    &reward::RewardEngine::Initialize)
        .def("score_batch",   [](reward::RewardEngine& self,
                                 const std::vector<std::string>& responses,
                                 bool has_ref, Tensor ref_logprobs) {
            reward::RewardResult result;
            std::optional<Tensor> ref = has_ref
                ? std::optional<Tensor>(std::move(ref_logprobs)) : std::nullopt;
            auto s = self.ScoreBatch(responses, ref, &result);
            if (!s.ok()) throw std::runtime_error(s.message());
            return result;
        }, py::arg("responses"), py::arg("has_ref") = false,
           py::arg("ref_logprobs") = Tensor(), py::return_value_policy::move)
        .def("score_tokens",   &reward::RewardEngine::ScoreTokens)
        .def("update_adaptive_kl", &reward::RewardEngine::UpdateAdaptiveKL)
        .def("current_kl_coef",    &reward::RewardEngine::CurrentKLCoef);

    // ---- DistributedTrainer ----
    py::class_<distributed::DistributedConfig>(m, "DistributedConfig")
        .def(py::init<>())
        .def_readwrite("rank",        &distributed::DistributedConfig::rank)
        .def_readwrite("world_size",  &distributed::DistributedConfig::world_size)
        .def_readwrite("master_addr", &distributed::DistributedConfig::master_addr)
        .def_readwrite("master_port", &distributed::DistributedConfig::master_port)
        .def_readwrite("use_nccl",    &distributed::DistributedConfig::use_nccl);

    py::class_<distributed::DistributedTrainer>(m, "DistributedTrainer")
        .def(py::init<const distributed::DistributedConfig&>(),
             py::arg("config"))
        .def("initialize",  &distributed::DistributedTrainer::Initialize)
        .def("shutdown",    &distributed::DistributedTrainer::Shutdown)
        .def("all_reduce",  &distributed::DistributedTrainer::AllReduce)
        .def("broadcast",   &distributed::DistributedTrainer::Broadcast)
        .def("get_rank",    &distributed::DistributedTrainer::GetRank)
        .def("get_world_size", &distributed::DistributedTrainer::GetWorldSize);

    // ---- Profiler ----

    // ProfilerGuard — RAII context manager that auto-ends a span.
    // Usage in Python:
    //   with ProfilerGuard(profiler, "forward", "compute", 0):
    //       model.forward(x)
    //   # span ends automatically here
    py::class_<profiler::Profiler>(m, "Profiler");  // forward declaration

    py::class_<profiler::SpanID>(m, "SpanID")
        .def(py::init<>())
        .def_readwrite("id", &profiler::SpanID::id);

    py::class_<profiler::Profiler::Guard>(m, "ProfilerGuard")
        .def(py::init<profiler::Profiler*, const std::string&,
                       const std::string&, int>(),
             py::arg("profiler"), py::arg("name"),
             py::arg("category") = "compute", py::arg("device_id") = -1)
        .def("__enter__", [](profiler::Profiler::Guard& g) -> profiler::Profiler::Guard& { return g; })
        .def("__exit__", [](profiler::Profiler::Guard& g,
                             const py::object&, const py::object&, const py::object&) {
            // Guard destructor auto-calls EndSpan
        });

    py::class_<profiler::ProfilerConfig>(m, "ProfilerConfig")
        .def(py::init<>())
        .def_readwrite("enable_nvtx",        &profiler::ProfilerConfig::enable_nvtx)
        .def_readwrite("enable_cuda_timing",  &profiler::ProfilerConfig::enable_cuda_timing)
        .def_readwrite("enable_memory",       &profiler::ProfilerConfig::enable_memory)
        .def_readwrite("enable_cpu",          &profiler::ProfilerConfig::enable_cpu)
        .def_readwrite("cuda_device_id",      &profiler::ProfilerConfig::cuda_device_id)
        .def_readwrite("export_dir",          &profiler::ProfilerConfig::export_dir);

    py::class_<profiler::Profiler>(m, "Profiler")
        .def(py::init<const profiler::ProfilerConfig&>(),
             py::arg("config") = profiler::ProfilerConfig{})
        .def("enable",      &profiler::Profiler::Enable)
        .def("disable",     &profiler::Profiler::Disable)
        .def("is_enabled",  &profiler::Profiler::IsEnabled)
        .def("begin_span",  [](profiler::Profiler& self, const std::string& name,
                                const std::string& category, int device_id) {
            return self.BeginSpan(name, category, device_id);
        }, py::arg("name"), py::arg("category") = "compute",
           py::arg("device_id") = -1)
        .def("end_span",    &profiler::Profiler::EndSpan)
        // Context-manager shortcut: with profiler.span("name"): ...
        .def("span",        [](profiler::Profiler* self, const std::string& name,
                                const std::string& category, int device_id) {
            return profiler::Profiler::Guard(self, name, category, device_id);
        }, py::arg("name"), py::arg("category") = "compute",
           py::arg("device_id") = -1,
           "Returns a context-manager Guard that auto-ends the span on __exit__.")
        .def("begin_event", &profiler::Profiler::BeginEvent)
        .def("end_event",   &profiler::Profiler::EndEvent)
        .def("mark",        &profiler::Profiler::Mark)
        .def("export_chrome_trace", &profiler::Profiler::ExportChromeTrace,
             py::arg("path"))
        .def("export_json",  &profiler::Profiler::ExportJSON, py::arg("path"))
        .def("print_summary", &profiler::Profiler::PrintSummary)
        .def("get_all_spans", &profiler::Profiler::GetAllSpans)
        .def("get_stats",    &profiler::Profiler::GetStats);

    // AggregatedStats
    py::class_<profiler::AggregatedStats>(m, "AggregatedStats")
        .def_readonly("name",     &profiler::AggregatedStats::name)
        .def_readonly("count",    &profiler::AggregatedStats::count)
        .def_readonly("total_ns", &profiler::AggregatedStats::total_ns)
        .def_readonly("min_ns",   &profiler::AggregatedStats::min_ns)
        .def_readonly("max_ns",   &profiler::AggregatedStats::max_ns)
        .def("avg_ns",            &profiler::AggregatedStats::avg_ns);

    // TensorBoard writer — AddHistogram accepts a numpy array directly
    py::class_<profiler::TensorBoardWriter>(m, "TensorBoardWriter")
        .def(py::init<const std::string&>(), py::arg("log_dir"))
        .def("add_scalar",
             [](profiler::TensorBoardWriter& self, const std::string& tag,
                float value, int step) { self.AddScalar(tag, value, step); },
             py::arg("tag"), py::arg("value"), py::arg("step"))
        .def("add_histogram",
             [](profiler::TensorBoardWriter& self, const std::string& tag,
                py::array_t<float> values, int step) {
                 auto buf = values.request();
                 self.AddHistogram(tag,
                                   static_cast<const float*>(buf.ptr),
                                   static_cast<int>(buf.size), step);
             }, py::arg("tag"), py::arg("values"), py::arg("step"))
        .def("add_text",
             [](profiler::TensorBoardWriter& self, const std::string& tag,
                const std::string& text, int step) { self.AddText(tag, text, step); },
             py::arg("tag"), py::arg("text"), py::arg("step"))
        .def("flush", &profiler::TensorBoardWriter::Flush);

    // Prometheus writer
    py::class_<profiler::Labels>(m, "Labels")
        .def(py::init<>())
        .def("__setitem__", [](profiler::Labels& self, const std::string& k,
                                const std::string& v) {
            self.kv[k] = v;
        })
        .def("__getitem__", [](const profiler::Labels& self, const std::string& k) {
            return self.kv.at(k);
        });

    py::class_<profiler::PrometheusWriter>(m, "PrometheusWriter")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def("write",        &profiler::PrometheusWriter::Write,
             py::arg("name"), py::arg("value"), py::arg("labels") = profiler::Labels{})
        .def("write_counter",&profiler::PrometheusWriter::WriteCounter,
             py::arg("name"), py::arg("value"), py::arg("labels") = profiler::Labels{})
        .def("write_gauge",  &profiler::PrometheusWriter::WriteGauge,
             py::arg("name"), py::arg("value"), py::arg("labels") = profiler::Labels{})
        .def("flush",        &profiler::PrometheusWriter::Flush);

    // ---- Top-level helpers ----
    m.def("get_version", []() { return "1.0.0"; });
    m.def("get_device_count", []() {
#ifdef TURBORL_HAS_CUDA
        return GetCudaDeviceCount();
#else
        return 0;
#endif
    });
    m.def("get_memory_info", [](int device_id) {
#ifdef TURBORL_HAS_CUDA
        auto info = GetCudaMemoryInfo(device_id);
        return py::dict(py::arg("total") = static_cast<long long>(info.total),
                        py::arg("free")  = static_cast<long long>(info.free),
                        py::arg("used")  = static_cast<long long>(info.used()),
                        py::arg("utilization") = info.utilization());
#else
        return py::dict(py::arg("total") = 0L, py::arg("free") = 0L,
                        py::arg("used") = 0L,   py::arg("utilization") = 0.0);
#endif
    }, py::arg("device_id") = 0);
}
