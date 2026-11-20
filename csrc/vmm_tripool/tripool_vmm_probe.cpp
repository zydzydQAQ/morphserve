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

}  // namespace

int main(int argc, char** argv) {
    int device_idx = 0;
    size_t spike_gib = 4;
    size_t layer_mib = 512;
    int single_iters = 500;
    int layer_iters = 30;
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

        const size_t gran = min_gran;
        size_t spike_bytes = ((spike_gib * (1ULL << 30)) + gran - 1) / gran * gran;
        size_t layer_bytes = ((layer_mib * (1ULL << 20)) + gran - 1) / gran * gran;
        size_t total_pages = std::max(spike_bytes, layer_bytes) / gran;
        size_t layer_pages = layer_bytes / gran;
        size_t spike_pages = spike_bytes / gran;

        // Reserve 3 disjoint VA arenas: Persistent KV, Transient Scratchpad, Opportunistic Expert Cache
        size_t arena_bytes = total_pages * gran;
        CUdeviceptr kv_va = 0, transient_va = 0, expert_va = 0;
        drv.check(drv.cuMemAddressReserve(&kv_va, arena_bytes, gran, 0, 0), "reserve(kv_va)");
        drv.check(drv.cuMemAddressReserve(&transient_va, arena_bytes, gran, 0, 0), "reserve(transient_va)");
        drv.check(drv.cuMemAddressReserve(&expert_va, arena_bytes, gran, 0, 0), "reserve(expert_va)");

        // Pre-allocate physical page pool (2MB handles)
        std::vector<CUmemGenericAllocationHandle> handles(total_pages, 0);
        auto t_create0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < total_pages; ++i) {
            drv.check(drv.cuMemCreate(&handles[i], gran, &prop, 0), "cuMemCreate");
        }
        auto t_create1 = std::chrono::steady_clock::now();
        double pool_create_ms =
            std::chrono::duration<double, std::milli>(t_create1 - t_create0).count();

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_idx;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Initially map all pages into Expert Cache VA
        for (size_t i = 0; i < total_pages; ++i) {
            drv.check(drv.cuMemMap(expert_va + i * gran, gran, 0, handles[i], 0), "init map expert");
        }
        drv.check(drv.cuMemSetAccess(expert_va, arena_bytes, &access, 1), "init setAccess expert");

        // --- Benchmark 1: Single 2MB page remap (Expert VA -> KV VA -> Expert VA) ---
        std::vector<double> single_remap_us;
        single_remap_us.reserve(single_iters);
        for (int iter = -20; iter < single_iters; ++iter) {
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(expert_va, gran), "single unmap expert");
            drv.check(drv.cuMemMap(kv_va, gran, 0, handles[0], 0), "single map kv");
            drv.check(drv.cuMemSetAccess(kv_va, gran, &access, 1), "single setAccess kv");
            auto t1 = std::chrono::steady_clock::now();

            drv.check(drv.cuMemUnmap(kv_va, gran), "single unmap kv");
            drv.check(drv.cuMemMap(expert_va, gran, 0, handles[0], 0), "single map expert");
            drv.check(drv.cuMemSetAccess(expert_va, gran, &access, 1), "single setAccess expert");

            if (iter >= 0) {
                single_remap_us.push_back(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }

        // --- Benchmark 2: Single-layer Expert (512 MiB) drop-clean + remap to KV VA ---
        std::vector<double> layer_remap_ms;
        layer_remap_ms.reserve(layer_iters);
        for (int iter = -5; iter < layer_iters; ++iter) {
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(expert_va, layer_bytes), "layer unmap expert");
            for (size_t i = 0; i < layer_pages; ++i) {
                drv.check(drv.cuMemMap(kv_va + i * gran, gran, 0, handles[i], 0), "layer map kv");
            }
            drv.check(drv.cuMemSetAccess(kv_va, layer_bytes, &access, 1), "layer setAccess kv");
            auto t1 = std::chrono::steady_clock::now();

            drv.check(drv.cuMemUnmap(kv_va, layer_bytes), "layer unmap kv");
            for (size_t i = 0; i < layer_pages; ++i) {
                drv.check(drv.cuMemMap(expert_va + i * gran, gran, 0, handles[i], 0), "layer restore expert");
            }
            drv.check(drv.cuMemSetAccess(expert_va, layer_bytes, &access, 1), "layer restore access");

            if (iter >= 0) {
                layer_remap_ms.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }

        // --- Benchmark 3: Full 4 GiB VAE Transient Spike burst (drop Expert -> Transient VA -> touch -> release) ---
        CUstream stream = nullptr;
        drv.check(drv.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        std::vector<double> spike_acquire_ms;
        std::vector<double> spike_release_ms;
        spike_acquire_ms.reserve(spike_iters);
        spike_release_ms.reserve(spike_iters);
        bool spike_pattern_ok = true;

        for (int iter = -2; iter < spike_iters; ++iter) {
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(expert_va, spike_bytes), "spike drop expert");
            for (size_t i = 0; i < spike_pages; ++i) {
                drv.check(drv.cuMemMap(transient_va + i * gran, gran, 0, handles[i], 0), "spike map transient");
            }
            drv.check(drv.cuMemSetAccess(transient_va, spike_bytes, &access, 1), "spike setAccess transient");
            auto t1 = std::chrono::steady_clock::now();

            if (iter == 0) {
                // Verify real GPU memory access on first & last page of the 4 GiB transient arena
                drv.check(drv.cuMemsetD8Async(transient_va, 0x5A, 4096, stream), "memset head");
                drv.check(drv.cuMemsetD8Async(transient_va + spike_bytes - 4096, 0xA5, 4096, stream), "memset tail");
                drv.check(drv.cuStreamSynchronize(stream), "sync verify");
                uint8_t head[64] = {}, tail[64] = {};
                drv.check(drv.cuMemcpyDtoH(head, transient_va, sizeof(head)), "dtoh head");
                drv.check(drv.cuMemcpyDtoH(tail, transient_va + spike_bytes - 64, sizeof(tail)), "dtoh tail");
                for (uint8_t b : head) if (b != 0x5A) spike_pattern_ok = false;
                for (uint8_t b : tail) if (b != 0xA5) spike_pattern_ok = false;
            }

            auto t2 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(transient_va, spike_bytes), "spike unmap transient");
            for (size_t i = 0; i < spike_pages; ++i) {
                drv.check(drv.cuMemMap(expert_va + i * gran, gran, 0, handles[i], 0), "spike restore expert");
            }
            drv.check(drv.cuMemSetAccess(expert_va, spike_bytes, &access, 1), "spike restore access");
            auto t3 = std::chrono::steady_clock::now();

            if (iter >= 0) {
                spike_acquire_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                spike_release_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
            }
        }

        // --- Benchmark 4: Async H2D Streaming into freshly cuMemMap'd 512 MiB Expert VA ---
        void* host_pinned = nullptr;
        drv.check(drv.cuMemHostAlloc(&host_pinned, layer_bytes, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
        auto* host_u32 = reinterpret_cast<uint32_t*>(host_pinned);
        size_t num_u32 = layer_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_u32; i += 1024) {
            host_u32[i] = static_cast<uint32_t>(0x13579BDFU ^ static_cast<uint32_t>(i));
        }

        CUevent ev_start = nullptr, ev_end = nullptr;
        drv.check(drv.cuEventCreate(&ev_start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
        drv.check(drv.cuEventCreate(&ev_end, CU_EVENT_DEFAULT), "cuEventCreate(end)");

        std::vector<double> h2d_gbps;
        std::vector<double> h2d_total_ms;
        h2d_gbps.reserve(h2d_iters);
        h2d_total_ms.reserve(h2d_iters);
        bool h2d_bit_exact = true;

        for (int iter = -2; iter < h2d_iters; ++iter) {
            // Unmap + freshly map layer VA, then stream weights asynchronously from pinned host DRAM
            auto t0 = std::chrono::steady_clock::now();
            drv.check(drv.cuMemUnmap(expert_va, layer_bytes), "h2d unmap");
            for (size_t i = 0; i < layer_pages; ++i) {
                drv.check(drv.cuMemMap(expert_va + i * gran, gran, 0, handles[i], 0), "h2d remap");
            }
            drv.check(drv.cuMemSetAccess(expert_va, layer_bytes, &access, 1), "h2d setAccess");
            drv.check(drv.cuEventRecord(ev_start, stream), "ev_start");
            drv.check(drv.cuMemcpyHtoDAsync(expert_va, host_pinned, layer_bytes, stream), "HtoDAsync");
            drv.check(drv.cuEventRecord(ev_end, stream), "ev_end");
            drv.check(drv.cuEventSynchronize(ev_end), "ev_sync");
            auto t1 = std::chrono::steady_clock::now();

            float copy_ms = 0.0f;
            drv.check(drv.cuEventElapsedTime(&copy_ms, ev_start, ev_end), "ev_elapsed");
            if (iter == 0) {
                // Spot-check bit-exact copy across page boundaries
                uint32_t check_val = 0;
                for (size_t p = 0; p < layer_pages; p += 16) {
                    size_t byte_off = p * gran;
                    size_t idx = byte_off / sizeof(uint32_t);
                    drv.check(drv.cuMemcpyDtoH(&check_val, expert_va + byte_off, sizeof(check_val)), "verify h2d");
                    if (check_val != (0x13579BDFU ^ static_cast<uint32_t>(idx))) {
                        h2d_bit_exact = false;
                        break;
                    }
                }
            }
            if (iter >= 0) {
                double gbps = (static_cast<double>(layer_bytes) / 1e9) / (static_cast<double>(copy_ms) / 1e3);
                h2d_gbps.push_back(gbps);
                h2d_total_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }

        // Cleanup & verify zero leaked HBM
        drv.check(drv.cuEventDestroy(ev_start), "cuEventDestroy");
        drv.check(drv.cuEventDestroy(ev_end), "cuEventDestroy");
        drv.check(drv.cuStreamDestroy(stream), "cuStreamDestroy");
        drv.check(drv.cuMemFreeHost(host_pinned), "cuMemFreeHost");

        drv.check(drv.cuMemUnmap(expert_va, arena_bytes), "cleanup unmap expert");
        for (auto h : handles) {
            drv.check(drv.cuMemRelease(h), "cuMemRelease");
        }
        drv.check(drv.cuMemAddressFree(kv_va, arena_bytes), "free kv_va");
        drv.check(drv.cuMemAddressFree(transient_va, arena_bytes), "free transient_va");
        drv.check(drv.cuMemAddressFree(expert_va, arena_bytes), "free expert_va");
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
            << "  \"free_before_bytes\": " << free_before << ",\n"
            << "  \"free_after_bytes\": " << free_after << ",\n"
            << "  \"residual_memory_delta_bytes\": " << residual_delta_bytes << ",\n"
            << "  \"min_granularity_bytes\": " << min_gran << ",\n"
            << "  \"recommended_granularity_bytes\": " << rec_gran << ",\n"
            << "  \"pool_pages\": " << total_pages << ",\n"
            << "  \"pool_bytes\": " << arena_bytes << ",\n"
            << "  \"pool_create_ms\": " << std::fixed << std::setprecision(3) << pool_create_ms << ",\n"
            << "  \"single_2mb_remap_us\": " << stats_json(summarize(single_remap_us)) << ",\n"
            << "  \"layer_remap_mib\": " << (layer_bytes / (1ULL << 20)) << ",\n"
            << "  \"layer_remap_ms\": " << stats_json(summarize(layer_remap_ms)) << ",\n"
            << "  \"vae_spike_gib\": " << (spike_bytes / (1ULL << 30)) << ",\n"
            << "  \"vae_spike_acquire_ms\": " << stats_json(summarize(spike_acquire_ms)) << ",\n"
            << "  \"vae_spike_release_ms\": " << stats_json(summarize(spike_release_ms)) << ",\n"
            << "  \"vae_spike_pattern_verified\": " << (spike_pattern_ok ? "true" : "false") << ",\n"
            << "  \"h2d_stream_mib\": " << (layer_bytes / (1ULL << 20)) << ",\n"
            << "  \"h2d_stream_gbps\": " << stats_json(summarize(h2d_gbps)) << ",\n"
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
        return (spike_pattern_ok && h2d_bit_exact) ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "{\"status\":\"error\",\"error\":\"" << escape_json(e.what()) << "\"}\n";
        return 1;
    }
}
