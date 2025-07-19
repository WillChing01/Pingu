#ifndef TIME_NETWORK_H_INCLUDED
#define TIME_NETWORK_H_INCLUDED

#include <algorithm>
#include <array>
#include "activations.h"
#include "cnn.h"
#include "linear.h"
#include "weights.h"

struct Scalar {
    float scaledEval;
    float scaledPly;
    float scaledIncrement;
    float scaledOpponentTime;
};

template <typename T, const size_t C, const size_t H, const size_t W, const size_t S>
class AvgPoolFlatten {
  public:
    std::array<T, C * S * S> output;

    AvgPoolFlatten() {}

    void forward(const std::array<T, C * H * W>& input) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < S; ++oh) {
                size_t hStart = (oh * H) / S;
                size_t hEnd = ((oh + 1) * H) / S;

                for (size_t ow = 0; ow < S; ++ow) {
                    size_t wStart = (ow * W) / S;
                    size_t wEnd = ((ow + 1) * W) / S;

                    T sum = T(0);
                    size_t count = (hEnd - hStart) * (wEnd - wStart);

                    for (size_t h = hStart; h < hEnd; ++h) {
                        for (size_t w = wStart; w < wEnd; ++w) {
                            sum += input[c * H * W + h * W + w];
                        }
                    }

                    output[c * S * S + oh * S + ow] = count ? sum / T(count) : T(0);
                }
            }
        }
    }
};

class Head {
  private:
    std::array<float, 100> input;

    linear::Linear<float, 64, 100, _binary_weights_time_head_w_float_64_100_bin_start,
                   _binary_weights_time_head_b_float_64_bin_start, activations::ReLU>
        initial;
    linear::Linear<float, 1, 64, _binary_weights_time_head_w_float_1_64_bin_start,
                   _binary_weights_time_head_b_float_1_bin_start, activations::Sigmoid>
        out;

  public:
    Head() {}

    float forward(const Scalar& scalar, const std::array<float, 96>& board) {
        input[0] = scalar.scaledEval;
        input[1] = scalar.scaledPly;
        input[2] = scalar.scaledIncrement;
        input[3] = scalar.scaledOpponentTime;
        std::copy(board.begin(), board.end(), input.begin() + 4);

        initial.forward(input);
        out.forward(initial.output);

        return out.output[0];
    }
};

class TimeNetwork {
  private:
    cnn::Conv2D<float, 24, 14, 3, 8, 8, _binary_weights_time_initial_b_float_24_bin_start,
                _binary_weights_time_initial_w_float_24_14_3_3_bin_start, activations::ReLU>
        initial;

    cnn::ResidualBlock<float, 24, 3, 8, 8, _binary_weights_time_block_in_w_float_24_24_3_3_bin_start,
                       _binary_weights_time_block_in_b_float_24_bin_start,
                       _binary_weights_time_block_out_w_float_24_24_3_3_bin_start,
                       _binary_weights_time_block_out_b_float_24_bin_start>
        block;

    AvgPoolFlatten<float, 24, 8, 8, 2> pool;

    Head head;

  public:
    TimeNetwork() {}

    float forward(const Scalar& scalar, const std::array<float, 14 * 8 * 8>& board) {
        initial.forward(board);
        block.forward(initial.output);
        pool.forward(block.output);
        return head.forward(scalar, pool.output);
    }
};

#endif // TIME_NETWORK_H_INCLUDED
