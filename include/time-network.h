#ifndef TIME_NETWORK_H_INCLUDED
#define TIME_NETWORK_H_INCLUDED

#include <algorithm>
#include <array>

#include "activations.h"
#include "bitboard.h"
#include "cnn.h"
#include "constants.h"
#include "linear.h"
#include "thread.h"
#include "weights.h"
#include "util.h"

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
    Thread* thread;

    Scalar scalar;
    std::array<float, 14 * 8 * 8> board;

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

    void refreshScalar(U32 timeLeft, U32 increment, U32 opponentTime) {
        thread->prepareSearch(MAXDEPTH, std::numeric_limits<double>::infinity(), false);
        int qSearch = thread->qSearch(1, -MATE_SCORE, MATE_SCORE);
        globalNodeCount = 0;

        scalar.scaledEval = 1.f / (1.f + std::exp(-qSearch / 400.f));
        scalar.scaledPly = std::min(thread->b.hashHistory.size() / 100.f, 1.f);
        scalar.scaledIncrement = std::min((float)increment / (float)timeLeft, 1.f);
        scalar.scaledOpponentTime = std::min(0.5f * (float)opponentTime / (float)timeLeft, 1.f);
    }

    void refreshBoard() {
        bool inCheck = util::isInCheck(thread->b.side, thread->b.pieces, thread->b.occupied);
        std::fill(board.begin(), board.begin() + 64 * 12, 0);
        for (size_t i = 0; i < 12; ++i) {
            U64 x = thread->b.pieces[i];
            while (x) {
                board[64 * i + popLSB(x)] = 1;
            }
        }
        std::fill(board.begin() + 64 * 12, board.begin() + 64 * 13, thread->b.side);
        std::fill(board.begin() + 64 * 13, board.begin() + 64 * 14, inCheck);
    }

  public:
    TimeNetwork(Thread* t) : thread(t) {}

    float forward(U32 timeLeft, U32 increment, U32 opponentTime) {
        refreshScalar(timeLeft, increment, opponentTime);
        refreshBoard();

        initial.forward(board);
        block.forward(initial.output);
        pool.forward(block.output);
        float fraction = head.forward(scalar, pool.output);

        return fraction;
    }
};

#endif // TIME_NETWORK_H_INCLUDED
