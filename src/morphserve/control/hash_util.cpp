#include "morphserve/control/hash_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace morphserve::control {

namespace {

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

inline uint32_t rotr32(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32U - n));
}

inline uint32_t rotl32(uint32_t x, uint32_t n) noexcept {
    return (x << n) | (x >> (32U - n));
}

void sha256_block(std::array<uint32_t, 8>& h, const uint8_t block[64]) noexcept {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24U) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16U) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8U) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7U) ^ rotr32(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
        const uint32_t s1 = rotr32(w[i - 2], 17U) ^ rotr32(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t hh = h[7];
    for (size_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6U) ^ rotr32(e, 11U) ^ rotr32(e, 25U);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = hh + s1 + ch + kSha256K[i] + w[i];
        const uint32_t s0 = rotr32(a, 2U) ^ rotr32(a, 13U) ^ rotr32(a, 22U);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

constexpr std::array<uint32_t, 64> kMd5K = {
    0xd76aa478U, 0xe8c7b756U, 0x242070dbU, 0xc1bdceeeU, 0xf57c0fafU, 0x4787c62aU, 0xa8304613U, 0xfd469501U,
    0x698098d8U, 0x8b44f7afU, 0xffff5bb1U, 0x895cd7beU, 0x6b901122U, 0xfd987193U, 0xa679438eU, 0x49b40821U,
    0xf61e2562U, 0xc040b340U, 0x265e5a51U, 0xe9b6c7aaU, 0xd62f105dU, 0x02441453U, 0xd8a1e681U, 0xe7d3fbc8U,
    0x21e1cde6U, 0xc33707d6U, 0xf4d50d87U, 0x455a14edU, 0xa9e3e905U, 0xfcefa3f8U, 0x676f02d9U, 0x8d2a4c8aU,
    0xfffa3942U, 0x8771f681U, 0x6d9d6122U, 0xfde5380cU, 0xa4beea44U, 0x4bdecfa9U, 0xf6bb4b60U, 0xbebfbc70U,
    0x289b7ec6U, 0xeaa127faU, 0xd4ef3085U, 0x04881d05U, 0xd9d4d039U, 0xe6db99e5U, 0x1fa27cf8U, 0xc4ac5665U,
    0xf4292244U, 0x432aff97U, 0xab9423a7U, 0xfc93a039U, 0x655b59c3U, 0x8f0ccc92U, 0xffeff47dU, 0x85845dd1U,
    0x6fa87e4fU, 0xfe2ce6e0U, 0xa3014314U, 0x4e0811a1U, 0xf7537e82U, 0xbd3af235U, 0x2ad7d2bbU, 0xeb86d391U};

constexpr std::array<uint32_t, 64> kMd5S = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

}  // namespace

std::string sha256_hex_bytes(const uint8_t* data, size_t len) {
    std::array<uint32_t, 8> h = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    size_t off = 0;
    while (off + 64 <= len) {
        sha256_block(h, data + off);
        off += 64;
    }
    uint8_t tail[128] = {};
    const size_t rem = len - off;
    if (rem > 0) {
        std::memcpy(tail, data + off, rem);
    }
    tail[rem] = 0x80U;
    const size_t pad_blocks = (rem < 56) ? 1 : 2;
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (size_t i = 0; i < 8; ++i) {
        tail[pad_blocks * 64 - 1 - i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xffULL);
    }
    sha256_block(h, tail);
    if (pad_blocks == 2) {
        sha256_block(h, tail + 64);
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t v : h) {
        oss << std::setw(8) << v;
    }
    return oss.str();
}

std::string sha256_hex_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("sha256_hex_file failed to open: " + path);
    }
    std::array<uint32_t, 8> h = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    uint8_t buf[65536];
    uint8_t rem_buf[64];
    size_t rem_len = 0;
    uint64_t total_len = 0;

    while (in) {
        in.read(reinterpret_cast<char*>(buf), sizeof(buf));
        const std::streamsize got = in.gcount();
        if (got <= 0) {
            break;
        }
        total_len += static_cast<uint64_t>(got);
        size_t pos = 0;
        const size_t n = static_cast<size_t>(got);
        if (rem_len > 0) {
            const size_t fill = std::min(64 - rem_len, n);
            std::memcpy(rem_buf + rem_len, buf, fill);
            rem_len += fill;
            pos += fill;
            if (rem_len == 64) {
                sha256_block(h, rem_buf);
                rem_len = 0;
            }
        }
        while (pos + 64 <= n) {
            sha256_block(h, buf + pos);
            pos += 64;
        }
        if (pos < n) {
            rem_len = n - pos;
            std::memcpy(rem_buf, buf + pos, rem_len);
        }
    }

    uint8_t tail[128] = {};
    if (rem_len > 0) {
        std::memcpy(tail, rem_buf, rem_len);
    }
    tail[rem_len] = 0x80U;
    const size_t pad_blocks = (rem_len < 56) ? 1 : 2;
    const uint64_t bit_len = total_len * 8ULL;
    for (size_t i = 0; i < 8; ++i) {
        tail[pad_blocks * 64 - 1 - i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xffULL);
    }
    sha256_block(h, tail);
    if (pad_blocks == 2) {
        sha256_block(h, tail + 64);
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t v : h) {
        oss << std::setw(8) << v;
    }
    return oss.str();
}

uint32_t md5_prefix_u32(const std::string& text) {
    const size_t len = text.size();
    const size_t pad_blocks = ((len % 64) < 56) ? (len / 64 + 1) : (len / 64 + 2);
    std::vector<uint8_t> msg(pad_blocks * 64, 0);
    if (len > 0) {
        std::memcpy(msg.data(), text.data(), len);
    }
    msg[len] = 0x80U;
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (size_t i = 0; i < 8; ++i) {
        msg[msg.size() - 8 + i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xffULL);
    }

    uint32_t a0 = 0x67452301U;
    uint32_t b0 = 0xefcdab89U;
    uint32_t c0 = 0x98badcfeU;
    uint32_t d0 = 0x10325476U;

    for (size_t blk = 0; blk < pad_blocks; ++blk) {
        const uint8_t* p = msg.data() + blk * 64;
        uint32_t m[16];
        for (size_t i = 0; i < 16; ++i) {
            m[i] = static_cast<uint32_t>(p[i * 4]) |
                   (static_cast<uint32_t>(p[i * 4 + 1]) << 8U) |
                   (static_cast<uint32_t>(p[i * 4 + 2]) << 16U) |
                   (static_cast<uint32_t>(p[i * 4 + 3]) << 24U);
        }
        uint32_t a = a0;
        uint32_t b = b0;
        uint32_t c = c0;
        uint32_t d = d0;
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t f = 0;
            uint32_t g = 0;
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5U * i + 1U) % 16U;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3U * i + 5U) % 16U;
            } else {
                f = c ^ (b | (~d));
                g = (7U * i) % 16U;
            }
            f = f + a + kMd5K[i] + m[g];
            a = d;
            d = c;
            c = b;
            b = b + rotl32(f, kMd5S[i]);
        }
        a0 += a;
        b0 += b;
        c0 += c;
        d0 += d;
    }

    // `hashlib.md5(...).hexdigest()[:8]` parses `digest[0..3]` (little-endian `a0` bytes) in big-endian order.
    const uint8_t b0_byte = static_cast<uint8_t>(a0 & 0xffU);
    const uint8_t b1_byte = static_cast<uint8_t>((a0 >> 8U) & 0xffU);
    const uint8_t b2_byte = static_cast<uint8_t>((a0 >> 16U) & 0xffU);
    const uint8_t b3_byte = static_cast<uint8_t>((a0 >> 24U) & 0xffU);
    return (static_cast<uint32_t>(b0_byte) << 24U) |
           (static_cast<uint32_t>(b1_byte) << 16U) |
           (static_cast<uint32_t>(b2_byte) << 8U) |
           static_cast<uint32_t>(b3_byte);
}

double percentile_raw(const std::vector<double>& vals, double p) {
    if (vals.empty()) {
        return 0.0;
    }
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());
    const double idx = static_cast<double>(s.size() - 1) * (p / 100.0);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(s.size() - 1, static_cast<size_t>(std::ceil(idx)));
    if (lo == hi) {
        return s[lo];
    }
    const double frac = idx - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

double mean_raw(const std::vector<double>& vals) {
    if (vals.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    return sum / static_cast<double>(vals.size());
}

double percentile_rounded(const std::vector<double>& vals, double p) {
    return round_2(percentile_raw(vals, p));
}

bool should_use_cascade_cfg_split(int q_len, int l_shared, int num_shared_branches) noexcept {
    if (q_len <= 0 || num_shared_branches < 2 || l_shared < 256) {
        return false;
    }
    if (q_len <= 2048) {
        return l_shared >= std::max(256, static_cast<int>(1.5 * static_cast<double>(q_len)));
    }
    return l_shared >= 4 * q_len;
}

}  // namespace morphserve::control
