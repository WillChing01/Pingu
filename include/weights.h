#ifndef WEIGHTS_H_INCLUDED
#define WEIGHTS_H_INCLUDED

#include <array>

extern const short _binary_weights_nnue_perspective_w0_short_45056_32_bin_start[];
const std::array<std::array<short, 32>, 45056>& perspective_w0 =
    *reinterpret_cast<const std::array<std::array<short, 32>, 45056>*>(
        _binary_weights_nnue_perspective_w0_short_45056_32_bin_start);

extern const short _binary_weights_nnue_perspective_b0_short_32_bin_start[];
const std::array<short, 32>& perspective_b0 =
    *reinterpret_cast<const std::array<short, 32>*>(_binary_weights_nnue_perspective_b0_short_32_bin_start);

extern const char _binary_weights_nnue_stacks_w0_char_4_64_bin_start[];
const std::array<std::array<char, 64>, 4>& stacks_w0 =
    *reinterpret_cast<const std::array<std::array<char, 64>, 4>*>(_binary_weights_nnue_stacks_w0_char_4_64_bin_start);

extern const int _binary_weights_nnue_stacks_b0_int_4_bin_start[];
const std::array<int, 4>& stacks_b0 =
    *reinterpret_cast<const std::array<int, 4>*>(_binary_weights_nnue_stacks_b0_int_4_bin_start);

#endif // WEIGHTS_H_INCLUDED
