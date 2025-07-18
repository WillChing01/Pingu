#ifndef WEIGHTS_H_INCLUDED
#define WEIGHTS_H_INCLUDED

#include <array>

extern const short _binary_weights_nnue_perspective_b0_short_32_bin_start[];
const std::array<short, 32>& perspective_b0_32 =
    *reinterpret_cast<const std::array<short, 32>*>(_binary_weights_nnue_perspective_b0_short_32_bin_start);

extern const short _binary_weights_nnue_perspective_w0_short_45056_32_bin_start[];
const std::array<short, 1441792>& perspective_w0_45056_32 =
    *reinterpret_cast<const std::array<short, 1441792>*>(_binary_weights_nnue_perspective_w0_short_45056_32_bin_start);

extern const int _binary_weights_nnue_stacks_b0_int_4_bin_start[];
const std::array<int, 4>& stacks_b0_4 =
    *reinterpret_cast<const std::array<int, 4>*>(_binary_weights_nnue_stacks_b0_int_4_bin_start);

extern const char _binary_weights_nnue_stacks_w0_char_4_64_bin_start[];
const std::array<char, 256>& stacks_w0_4_64 =
    *reinterpret_cast<const std::array<char, 256>*>(_binary_weights_nnue_stacks_w0_char_4_64_bin_start);

extern const float _binary_weights_time_block_in_b_float_24_bin_start[];
const std::array<float, 24>& block_in_b_24 =
    *reinterpret_cast<const std::array<float, 24>*>(_binary_weights_time_block_in_b_float_24_bin_start);

extern const float _binary_weights_time_block_in_w_float_24_24_3_3_bin_start[];
const std::array<float, 5184>& block_in_w_24_24_3_3 =
    *reinterpret_cast<const std::array<float, 5184>*>(_binary_weights_time_block_in_w_float_24_24_3_3_bin_start);

extern const float _binary_weights_time_block_out_b_float_24_bin_start[];
const std::array<float, 24>& block_out_b_24 =
    *reinterpret_cast<const std::array<float, 24>*>(_binary_weights_time_block_out_b_float_24_bin_start);

extern const float _binary_weights_time_block_out_w_float_24_24_3_3_bin_start[];
const std::array<float, 5184>& block_out_w_24_24_3_3 =
    *reinterpret_cast<const std::array<float, 5184>*>(_binary_weights_time_block_out_w_float_24_24_3_3_bin_start);

extern const float _binary_weights_time_head_b_float_1_bin_start[];
const std::array<float, 1>& head_b_1 =
    *reinterpret_cast<const std::array<float, 1>*>(_binary_weights_time_head_b_float_1_bin_start);

extern const float _binary_weights_time_head_b_float_64_bin_start[];
const std::array<float, 64>& head_b_64 =
    *reinterpret_cast<const std::array<float, 64>*>(_binary_weights_time_head_b_float_64_bin_start);

extern const float _binary_weights_time_head_w_float_1_64_bin_start[];
const std::array<float, 64>& head_w_1_64 =
    *reinterpret_cast<const std::array<float, 64>*>(_binary_weights_time_head_w_float_1_64_bin_start);

extern const float _binary_weights_time_head_w_float_64_100_bin_start[];
const std::array<float, 6400>& head_w_64_100 =
    *reinterpret_cast<const std::array<float, 6400>*>(_binary_weights_time_head_w_float_64_100_bin_start);

extern const float _binary_weights_time_initial_b_float_24_bin_start[];
const std::array<float, 24>& initial_b_24 =
    *reinterpret_cast<const std::array<float, 24>*>(_binary_weights_time_initial_b_float_24_bin_start);

extern const float _binary_weights_time_initial_w_float_24_14_3_3_bin_start[];
const std::array<float, 3024>& initial_w_24_14_3_3 =
    *reinterpret_cast<const std::array<float, 3024>*>(_binary_weights_time_initial_w_float_24_14_3_3_bin_start);

#endif // WEIGHTS_H_INCLUDED
