#include "cuda_driver_min.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using morphserve::vmm::CudaDriver;

namespace {

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

Stats summarize(std::vector<double> vals) {
    Stats s;
    if (vals.empty()) return s;
    std::sort(vals.begin(), vals.end());
    s.min = vals.front();
    s.max = vals.back();
    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
    auto pct = [&](double p) {
        double idx = p * static_cast<double>(vals.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        double frac = idx - static_cast<double>(lo);
        return vals[lo] * (1.0 - frac) + vals[hi] * frac;
    };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

std::string stats_json(const Stats& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "{\"mean\":" << s.mean
        << ",\"min\":" << s.min
        << ",\"p50\":" << s.p50
        << ",\"p95\":" << s.p95
        << ",\"p99\":" << s.p99
        << ",\"max\":" << s.max << "}";
    return oss.str();
}

std::string escape_json(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

struct SweepEntry {
    size_t handle_mib = 0;
    size_t layer_handles = 0;
    size_t spike_handles = 0;
    double pool_create_ms = 0.0;
    Stats layer_remap_ms;
    Stats spike_acquire_ms;
    Stats spike_unmap_only_ms;
    Stats spike_release_ms;
    bool pattern_ok = true;
};

SweepEntry run_granularity_sweep(
    CudaDriver& drv,
    const CUmemAllocationProp& prop,
    const CUmemAccessDesc& access,
    CUstream stream,
    size_t handle_bytes,
    size_t layer_bytes,
    size_t spike_bytes,
    int layer_iters,
    int spike_iters) {
    SweepEntry entry;
    entry.handle_mib = handle_bytes / (1ULL << 20);
    size_t arena_bytes = std::max(layer_bytes, spike_bytes);
    arena_bytes = ((arena_bytes + handle_bytes - 1) / handle_bytes) * handle_bytes;
    layer_bytes = ((layer_bytes + handle_bytes - 1) / handle_bytes) * handle_bytes;
    spike_bytes = ((spike_bytes + handle_bytes - 1) / handle_bytes) * handle_bytes;

    size_t total_handles = arena_bytes / handle_bytes;
    entry.layer_handles = layer_bytes / handle_bytes;
    entry.spike_handles = spike_bytes / handle_bytes;

    CUdeviceptr kv_va = 0, transient_va = 0, expert_va = 0;
    drv.check(drv.cuMemAddressReserve(&kv_va, arena_bytes, handle_bytes, 0, 0), "sweep reserve kv");
    drv.check(drv.cuMemAddressReserve(&transient_va, arena_bytes, handle_bytes, 0, 0), "sweep reserve transient");
    drv.check(drv.cuMemAddressReserve(&expert_va, arena_bytes, handle_bytes, 0, 0), "sweep reserve expert");

    std::vector<CUmemGenericAllocationHandle> handles(total_handles, 0);
    auto tc0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < total_handles; ++i) {
        drv.check(drv.cuMemCreate(&handles[i], handle_bytes, &prop, 0), "sweep cuMemCreate");
    }
    auto tc1 = std::chrono::steady_clock::now();
    entry.pool_create_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

    for (size_t i = 0; i < total_handles; ++i) {
        drv.check(drv.cuMemMap(expert_va + i * handle_bytes, handle_bytes, 0, handles[i], 0), "sweep init map");
    }
    drv.check(drv.cuMemSetAccess(expert_va, arena_bytes, &access, 1), "sweep init access");

    // Layer remap (512 MiB)
    std::vector<double> layer_ms;
    layer_ms.reserve(layer_iters);
    for (int iter = -3; iter < layer_iters; ++iter) {
        auto t0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(expert_va, layer_bytes), "sweep layer unmap");
        for (size_t i = 0; i < entry.layer_handles; ++i) {
            drv.check(drv.cuMemMap(kv_va + i * handle_bytes, handle_bytes, 0, handles[i], 0), "sweep layer map");
        }
        drv.check(drv.cuMemSetAccess(kv_va, layer_bytes, &access, 1), "sweep layer access");
        auto t1 = std::chrono::steady_clock::now();

        drv.check(drv.cuMemUnmap(kv_va, layer_bytes), "sweep layer restore unmap");
        for (size_t i = 0; i < entry.layer_handles; ++i) {
            drv.check(drv.cuMemMap(expert_va + i * handle_bytes, handle_bytes, 0, handles[i], 0), "sweep layer restore map");
        }
        drv.check(drv.cuMemSetAccess(expert_va, layer_bytes, &access, 1), "sweep layer restore access");

        if (iter >= 0) {
            layer_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    entry.layer_remap_ms = summarize(layer_ms);

    // 4 GiB VAE spike acquire + bulk unmap + restore
    std::vector<double> acq_ms, unmap_ms, rel_ms;
    acq_ms.reserve(spike_iters);
    unmap_ms.reserve(spike_iters);
    rel_ms.reserve(spike_iters);
    for (int iter = -2; iter < spike_iters; ++iter) {
        auto t0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(expert_va, spike_bytes), "sweep spike drop expert");
        for (size_t i = 0; i < entry.spike_handles; ++i) {
            drv.check(drv.cuMemMap(transient_va + i * handle_bytes, handle_bytes, 0, handles[i], 0), "sweep spike map");
        }
        drv.check(drv.cuMemSetAccess(transient_va, spike_bytes, &access, 1), "sweep spike access");
        auto t1 = std::chrono::steady_clock::now();

        if (iter == 0) {
            drv.check(drv.cuMemsetD8Async(transient_va, 0x5A, 4096, stream), "sweep memset head");
            drv.check(drv.cuMemsetD8Async(transient_va + spike_bytes - 4096, 0xA5, 4096, stream), "sweep memset tail");
            drv.check(drv.cuStreamSynchronize(stream), "sweep sync verify");
            uint8_t head[64] = {}, tail[64] = {};
            drv.check(drv.cuMemcpyDtoH(head, transient_va, sizeof(head)), "sweep dtoh head");
            drv.check(drv.cuMemcpyDtoH(tail, transient_va + spike_bytes - 64, sizeof(tail)), "sweep dtoh tail");
            for (uint8_t b : head) if (b != 0x5A) entry.pattern_ok = false;
            for (uint8_t b : tail) if (b != 0xA5) entry.pattern_ok = false;
        }

        auto t2 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(transient_va, spike_bytes), "sweep spike bulk unmap");
        auto t_unmap = std::chrono::steady_clock::now();
        for (size_t i = 0; i < entry.spike_handles; ++i) {
            drv.check(drv.cuMemMap(expert_va + i * handle_bytes, handle_bytes, 0, handles[i], 0), "sweep spike restore");
        }
        drv.check(drv.cuMemSetAccess(expert_va, spike_bytes, &access, 1), "sweep spike restore access");
        auto t3 = std::chrono::steady_clock::now();

        if (iter >= 0) {
            acq_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            unmap_ms.push_back(std::chrono::duration<double, std::milli>(t_unmap - t2).count());
            rel_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
        }
    }
    entry.spike_acquire_ms = summarize(acq_ms);
    entry.spike_unmap_only_ms = summarize(unmap_ms);
    entry.spike_release_ms = summarize(rel_ms);

    drv.check(drv.cuMemUnmap(expert_va, arena_bytes), "sweep cleanup unmap");
    for (auto h : handles) drv.check(drv.cuMemRelease(h), "sweep release");
    drv.check(drv.cuMemAddressFree(kv_va, arena_bytes), "sweep free kv");
    drv.check(drv.cuMemAddressFree(transient_va, arena_bytes), "sweep free transient");
    drv.check(drv.cuMemAddressFree(expert_va, arena_bytes), "sweep free expert");
    return entry;
}

}  // namespace

int main(int argc, char** argv) {
    int device_idx = 0;
    size_t spike_gib = 4;
    size_t layer_mib = 512;
    int single_iters = 500;
    int layer_iters = 20;
    int spike_iters = 10;
    int h2d_iters = 10;
    std::string json_out;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device_idx = std::stoi(argv[++i]);
        else if (arg == "--spike-gib" && i + 1 < argc) spike_gib = std::stoull(argv[++i]);
        else if (arg == "--layer-mib" && i + 1 < argc) layer_mib = std::stoull(argv[++i]);
        else if (arg == "--single-iters" && i + 1 < argc) single_iters = std::stoi(argv[++i]);
        else if (arg == "--layer-iters" && i + 1 < argc) layer_iters = std::stoi(argv[++i]);
        else if (arg == "--spike-iters" && i + 1 < argc) spike_iters = std::stoi(argv[++i]);
        else if (arg == "--h2d-iters" && i + 1 < argc) h2d_iters = std::stoi(argv[++i]);
        else if (arg == "--json-out" && i + 1 < argc) json_out = argv[++i];
    }

    try {
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuInit(0), "cuInit");

        int driver_version = 0;
        drv.check(drv.cuDriverGetVersion(&driver_version), "cuDriverGetVersion");

        int dev_count = 0;
        drv.check(drv.cuDeviceGetCount(&dev_count), "cuDeviceGetCount");
        if (device_idx < 0 || device_idx >= dev_count) {
            throw std::runtime_error("invalid device index");
        }

        CUdevice dev = 0;
        drv.check(drv.cuDeviceGet(&dev, device_idx), "cuDeviceGet");
        char dev_name[256] = {};
        drv.check(drv.cuDeviceGetName(dev_name, sizeof(dev_name), dev), "cuDeviceGetName");

        int vmm_supported = 0;
        drv.check(drv.cuDeviceGetAttribute(
                      &vmm_supported,
                      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                      dev),
                  "cuDeviceGetAttribute(VMM)");

        CUcontext ctx = nullptr;
        drv.check(drv.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        drv.check(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

        size_t free_before = 0, total_mem = 0;
        drv.check(drv.cuMemGetInfo(&free_before, &total_mem), "cuMemGetInfo(before)");

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_idx;

        size_t min_gran = 0, rec_gran = 0;
        drv.check(drv.cuMemGetAllocationGranularity(
                      &min_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                  "cuMemGetAllocationGranularity(min)");
        drv.check(drv.cuMemGetAllocationGranularity(
                      &rec_gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                  "cuMemGetAllocationGranularity(rec)");

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_idx;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUstream stream = nullptr;
        drv.check(drv.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        // --- Benchmark 1: Single 2MB KV page remap ---
        CUdeviceptr kv_va = 0, exp_va = 0;
        drv.check(drv.cuMemAddressReserve(&kv_va, min_gran, min_gran, 0, 0), "single reserve kv");
        drv.check(drv.cuMemAddressReserve(&exp_va, min_gran, min_gran, 0, 0), "single reserve exp");
        CUmemGenericAllocationHandle one_page = 0;
        drv.check(drv.cuMemCreate(&one_page, min_gran, &prop, 0), "single create");
        drv.check(drv.cuMemMap(exp_va, min_gran, 0, one_page, 0), "single init map");
        drv.check(drv.cuMemSetAccess(exp_va, min_gran, &access, 1), "single init access");

        std::vector<double> single_remap_us;
        single_remap_us.reserve(single_iters);
        for (int iter = -20; iter < single_iters; ++iter) {
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(exp_va, min_gran), "single unmap exp");
            drv.check(drv.cuMemMap(kv_va, min_gran, 0, one_page, 0), "single map kv");
            drv.check(drv.cuMemSetAccess(kv_va, min_gran, &access, 1), "single access kv");
            auto t1 = std::chrono::steady_clock::now();

            drv.check(drv.cuMemUnmap(kv_va, min_gran), "single unmap kv");
            drv.check(drv.cuMemMap(exp_va, min_gran, 0, one_page, 0), "single map exp");
            drv.check(drv.cuMemSetAccess(exp_va, min_gran, &access, 1), "single access exp");

            if (iter >= 0) {
                single_remap_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
        drv.check(drv.cuMemUnmap(exp_va, min_gran), "single cleanup unmap");
        drv.check(drv.cuMemRelease(one_page), "single release");
        drv.check(drv.cuMemAddressFree(kv_va, min_gran), "single free kv");
        drv.check(drv.cuMemAddressFree(exp_va, min_gran), "single free exp");

        // --- Benchmark 2: Two-Tier Super-Handle Granularity Sweep (2, 32, 64, 128, 256 MiB) ---
        const size_t layer_bytes = layer_mib * (1ULL << 20);
        const size_t spike_bytes = spike_gib * (1ULL << 30);
        const std::vector<size_t> sweep_mibs = {2, 32, 64, 128, 256};
        std::vector<SweepEntry> sweep_results;
        bool all_patterns_ok = true;
        for (size_t hmib : sweep_mibs) {
            size_t hbytes = hmib * (1ULL << 20);
            if (hbytes < min_gran || (hbytes % min_gran) != 0) continue;
            SweepEntry e = run_granularity_sweep(
                drv, prop, access, stream, hbytes, layer_bytes, spike_bytes, layer_iters, spike_iters);
            if (!e.pattern_ok) all_patterns_ok = false;
            sweep_results.push_back(e);
        }

        // --- Benchmark 3: Async H2D Streaming into freshly 128 MiB Super-Handle mapped 512 MiB Expert layer ---
        const size_t super_bytes = 128ULL << 20;
        const size_t h2d_handles_n = (layer_bytes + super_bytes - 1) / super_bytes;
        const size_t h2d_bytes = h2d_handles_n * super_bytes;
        CUdeviceptr h2d_va = 0;
        drv.check(drv.cuMemAddressReserve(&h2d_va, h2d_bytes, super_bytes, 0, 0), "h2d reserve");
        std::vector<CUmemGenericAllocationHandle> h2d_handles(h2d_handles_n, 0);
        for (size_t i = 0; i < h2d_handles_n; ++i) {
            drv.check(drv.cuMemCreate(&h2d_handles[i], super_bytes, &prop, 0), "h2d create");
            drv.check(drv.cuMemMap(h2d_va + i * super_bytes, super_bytes, 0, h2d_handles[i], 0), "h2d init map");
        }
        drv.check(drv.cuMemSetAccess(h2d_va, h2d_bytes, &access, 1), "h2d init access");

        void* host_pinned = nullptr;
        drv.check(drv.cuMemHostAlloc(&host_pinned, h2d_bytes, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
        auto* host_u32 = reinterpret_cast<uint32_t*>(host_pinned);
        size_t num_u32 = h2d_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_u32; i += 1024) {
            host_u32[i] = static_cast<uint32_t>(0x13579BDFU ^ static_cast<uint32_t>(i));
        }

        CUevent ev_start = nullptr, ev_end = nullptr;
        drv.check(drv.cuEventCreate(&ev_start, CU_EVENT_DEFAULT), "ev_start");
        drv.check(drv.cuEventCreate(&ev_end, CU_EVENT_DEFAULT), "ev_end");

        std::vector<double> h2d_gbps, h2d_remap_ms, h2d_total_ms;
        bool h2d_bit_exact = true;
        for (int iter = -2; iter < h2d_iters; ++iter) {
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(h2d_va, h2d_bytes), "h2d unmap");
            for (size_t i = 0; i < h2d_handles_n; ++i) {
                drv.check(drv.cuMemMap(h2d_va + i * super_bytes, super_bytes, 0, h2d_handles[i], 0), "h2d map");
            }
            drv.check(drv.cuMemSetAccess(h2d_va, h2d_bytes, &access, 1), "h2d access");
            auto t_mapped = std::chrono::steady_clock::now();

            drv.check(drv.cuEventRecord(ev_start, stream), "ev_start");
            drv.check(drv.cuMemcpyHtoDAsync(h2d_va, host_pinned, h2d_bytes, stream), "HtoDAsync");
            drv.check(drv.cuEventRecord(ev_end, stream), "ev_end");
            drv.check(drv.cuEventSynchronize(ev_end), "ev_sync");
            auto t1 = std::chrono::steady_clock::now();

            float copy_ms = 0.0f;
            drv.check(drv.cuEventElapsedTime(&copy_ms, ev_start, ev_end), "ev_elapsed");
            if (iter == 0) {
                uint32_t check_val = 0;
                for (size_t off = 0; off < h2d_bytes; off += (32ULL << 20)) {
                    size_t idx = off / sizeof(uint32_t);
                    drv.check(drv.cuMemcpyDtoH(&check_val, h2d_va + off, sizeof(check_val)), "verify h2d");
                    if (check_val != (0x13579BDFU ^ static_cast<uint32_t>(idx))) {
                        h2d_bit_exact = false;
                        break;
                    }
                }
            }
            if (iter >= 0) {
                double gbps = (static_cast<double>(h2d_bytes) / 1e9) / (static_cast<double>(copy_ms) / 1e3);
                h2d_gbps.push_back(gbps);
                h2d_remap_ms.push_back(std::chrono::duration<double, std::milli>(t_mapped - t0).count());
                h2d_total_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }

        drv.check(drv.cuEventDestroy(ev_start), "destroy ev_start");
        drv.check(drv.cuEventDestroy(ev_end), "destroy ev_end");
        drv.check(drv.cuMemFreeHost(host_pinned), "free host");
        drv.check(drv.cuMemUnmap(h2d_va, h2d_bytes), "h2d cleanup unmap");
        for (auto h : h2d_handles) drv.check(drv.cuMemRelease(h), "h2d release");
        drv.check(drv.cuMemAddressFree(h2d_va, h2d_bytes), "h2d free va");
        drv.check(drv.cuStreamDestroy(stream), "destroy stream");
        drv.check(drv.cuCtxSynchronize(), "cuCtxSynchronize");

        size_t free_after = 0, total_after = 0;
        drv.check(drv.cuMemGetInfo(&free_after, &total_after), "cuMemGetInfo(after)");
        long long residual_delta_bytes =
            static_cast<long long>(free_before) - static_cast<long long>(free_after);

        std::ostringstream out;
        out << "{\n"
            << "  \"status\": \"ok\",\n"
            << "  \"compiled_with_cuda_h\": " << (MORPHSERVE_HAS_CUDA_H ? "true" : "false") << ",\n"
            << "  \"driver_version\": " << driver_version << ",\n"
            << "  \"device_index\": " << device_idx << ",\n"
            << "  \"device_name\": \"" << escape_json(dev_name) << "\",\n"
            << "  \"vmm_supported\": " << (vmm_supported ? "true" : "false") << ",\n"
            << "  \"total_memory_bytes\": " << total_mem << ",\n"
            << "  \"residual_memory_delta_bytes\": " << residual_delta_bytes << ",\n"
            << "  \"min_granularity_bytes\": " << min_gran << ",\n"
            << "  \"recommended_granularity_bytes\": " << rec_gran << ",\n"
            << "  \"single_2mb_remap_us\": " << stats_json(summarize(single_remap_us)) << ",\n"
            << "  \"granularity_sweep\": [\n";
        for (size_t i = 0; i < sweep_results.size(); ++i) {
            const auto& e = sweep_results[i];
            out << "    {\n"
                << "      \"handle_mib\": " << e.handle_mib << ",\n"
                << "      \"layer_handles\": " << e.layer_handles << ",\n"
                << "      \"spike_handles\": " << e.spike_handles << ",\n"
                << "      \"pool_create_ms\": " << std::fixed << std::setprecision(3) << e.pool_create_ms << ",\n"
                << "      \"layer_512mib_remap_ms\": " << stats_json(e.layer_remap_ms) << ",\n"
                << "      \"vae_4gib_acquire_ms\": " << stats_json(e.spike_acquire_ms) << ",\n"
                << "      \"vae_4gib_bulk_unmap_only_ms\": " << stats_json(e.spike_unmap_only_ms) << ",\n"
                << "      \"vae_4gib_release_and_restore_ms\": " << stats_json(e.spike_release_ms) << ",\n"
                << "      \"pattern_verified\": " << (e.pattern_ok ? "true" : "false") << "\n"
                << "    }" << (i + 1 < sweep_results.size() ? "," : "") << "\n";
        }
        out << "  ],\n"
            << "  \"h2d_super_handle_mib\": 128,\n"
            << "  \"h2d_stream_mib\": " << (h2d_bytes / (1ULL << 20)) << ",\n"
            << "  \"h2d_stream_gbps\": " << stats_json(summarize(h2d_gbps)) << ",\n"
            << "  \"h2d_super_remap_ms\": " << stats_json(summarize(h2d_remap_ms)) << ",\n"
            << "  \"h2d_remap_plus_copy_ms\": " << stats_json(summarize(h2d_total_ms)) << ",\n"
            << "  \"h2d_bit_exact\": " << (h2d_bit_exact ? "true" : "false") << "\n"
            << "}\n";

        const std::string payload = out.str();
        if (!json_out.empty()) {
            std::ofstream ofs(json_out);
            ofs << payload;
        }
        std::cout << payload;
        drv.cuDevicePrimaryCtxRelease(dev);
        return (all_patterns_ok && h2d_bit_exact) ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "{\"status\":\"error\",\"error\":\"" << escape_json(e.what()) << "\"}\n";
        return 1;
    }
}
