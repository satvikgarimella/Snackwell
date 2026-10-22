// Independent integer binary32 oracle. Five limbs hold every exact aligned
// sum of two finite FP32 operands (at most 278 bits); no host FP or RTL is used.
#include <algorithm>
#include <array>
#include <cstdint>

using Wide = std::array<uint64_t, 5>;

static Wide shifted(uint64_t value, int shift) {
    Wide result{};
    const int limb = shift / 64, offset = shift % 64;
    result[limb] = value << offset;
    if (offset && limb + 1 < 5) result[limb + 1] = value >> (64 - offset);
    return result;
}

static int compare(const Wide& a, const Wide& b) {
    for (int i = 4; i >= 0; --i) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}

static Wide combine(const Wide& a, const Wide& b, bool subtract) {
    Wide result{};
    uint64_t carry = 0;
    for (int i = 0; i < 5; ++i) {
        if (subtract) {
            const uint64_t partial = a[i] - b[i];
            result[i] = partial - carry;
            carry = (a[i] < b[i]) || (partial < carry);
        } else {
            const uint64_t partial = a[i] + b[i];
            result[i] = partial + carry;
            carry = (partial < a[i]) || (result[i] < partial);
        }
    }
    return result;
}

static bool bit(const Wide& a, int index) {
    return index >= 0 && index < 320 && ((a[index / 64] >> (index % 64)) & 1);
}

static bool below(const Wide& a, int end) {
    for (int i = 0; i < 5 && i * 64 < end; ++i) {
        const int count = std::min(64, end - i * 64);
        const uint64_t mask = count == 64 ? UINT64_MAX : (uint64_t(1) << count) - 1;
        if (a[i] & mask) return true;
    }
    return false;
}

static uint32_t pack(const Wide& magnitude, int power, uint32_t sign, uint32_t zero_sign) {
    int length = 0;
    for (int i = 4; i >= 0; --i) {
        if (magnitude[i]) {
            length = i * 64 + 64 - __builtin_clzll(magnitude[i]);
            break;
        }
    }
    if (!length) return zero_sign << 31;
    int quantum = std::max(length - 1 + power - 23, -149);
    const int shift = quantum - power;
    uint64_t significand;
    if (shift > 0) {
        const int limb = shift / 64, offset = shift % 64;
        significand = limb < 5 ? magnitude[limb] >> offset : 0;
        if (offset && limb + 1 < 5) significand |= magnitude[limb + 1] << (64 - offset);
        if (bit(magnitude, shift - 1) && (below(magnitude, shift - 1) || (significand & 1)))
            ++significand;
    } else {
        significand = magnitude[0] << -shift;
    }
    if (significand >= (1u << 24)) {
        significand >>= 1;
        ++quantum;
    }
    if (significand < (1u << 23)) return (sign << 31) | uint32_t(significand);
    const int exponent = quantum + 150;
    if (exponent >= 255) return (sign << 31) | 0x7f800000u;
    return (sign << 31) | (uint32_t(exponent) << 23) | (significand & 0x7fffffu);
}

extern "C" uint32_t snackwell_fp32(uint32_t opcode, uint32_t a, uint32_t b) {
    if (opcode == 0x0e) b ^= 0x80000000u;
    const uint32_t am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if (am > 0x7f800000u || bm > 0x7f800000u) return 0x7fc00000u;
    const uint32_t sign = (a ^ b) >> 31;
    if (opcode == 0x0c) {
        if ((am == 0x7f800000u && bm == 0) || (bm == 0x7f800000u && am == 0))
            return 0x7fc00000u;
        if (am == 0x7f800000u || bm == 0x7f800000u) return (sign << 31) | 0x7f800000u;
    } else {
        if (am == 0x7f800000u && bm == 0x7f800000u && sign) return 0x7fc00000u;
        if (am == 0x7f800000u) return a;
        if (bm == 0x7f800000u) return b;
    }
    const int ae = (a >> 23) & 255, be = (b >> 23) & 255;
    const uint64_t na = (a & 0x7fffffu) | (ae ? 1u << 23 : 0);
    const uint64_t nb = (b & 0x7fffffu) | (be ? 1u << 23 : 0);
    const int ea = ae ? ae - 150 : -149, eb = be ? be - 150 : -149;
    if (opcode == 0x0c) return pack(shifted(na * nb, 0), ea + eb, sign, sign);
    const int common = std::min(ea, eb);
    const Wide left = shifted(na, ea - common), right = shifted(nb, eb - common);
    const uint32_t zero_sign = (a & b) >> 31;
    if (!sign) return pack(combine(left, right, false), common, a >> 31, zero_sign);
    if (compare(left, right) >= 0)
        return pack(combine(left, right, true), common, a >> 31, zero_sign);
    return pack(combine(right, left, true), common, b >> 31, zero_sign);
}

extern "C" void snackwell_tensor_product(
    const uint32_t* a, const uint32_t* b, const uint32_t* initial, uint32_t* out,
    int m, int n, int k, int transpose_b
) {
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            uint32_t total = initial ? initial[r * n + c] : 0;
            for (int base = 0; base < k; base += 4) {
                uint32_t partial = 0;
                for (int j = base; j < base + 4; ++j) {
                    const uint32_t right = b[transpose_b ? c * k + j : j * n + c];
                    partial = snackwell_fp32(0x0d, partial, snackwell_fp32(0x0c, a[r * k + j], right));
                }
                total = snackwell_fp32(0x0d, total, partial);
            }
            out[r * n + c] = total;
        }
    }
}
