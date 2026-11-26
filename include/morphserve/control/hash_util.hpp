#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace morphserve::control {

inline double round_1(double x) noexcept {
    return std::round(x * 10.0) / 10.0;
}

inline double round_2(double x) noexcept {
    return std::round(x * 100.0) / 100.0;
}

inline double round_3(double x) noexcept {
    return std::round(x * 1000.0) / 1000.0;
}

inline double round_4(double x) noexcept {
    return std::round(x * 10000.0) / 10000.0;
}

inline double round_6(double x) noexcept {
    return std::round(x * 1000000.0) / 1000000.0;
}

inline double round_n(double val, int digits) noexcept {
    const double p = std::pow(10.0, digits);
    return std::round(val * p) / p;
}

// Pure C++17 cryptographic/deterministic hash helpers (zero external OpenSSL dependency).
std::string sha256_hex_bytes(const uint8_t* data, size_t len);
std::string sha256_hex_file(const std::string& path);
uint32_t md5_prefix_u32(const std::string& text);

// Shared summary statistics helpers used across control plane and simulator.
double percentile_raw(const std::vector<double>& vals, double p);
double mean_raw(const std::vector<double>& vals);
double percentile_rounded(const std::vector<double>& vals, double p);

// Arithmetic-intensity gate for two-pass FlashPrefix-CFG + fused LSE merge.
bool should_use_cascade_cfg_split(int q_len, int l_shared, int num_shared_branches) noexcept;

}  // namespace morphserve::control
