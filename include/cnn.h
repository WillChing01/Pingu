#ifndef CNN_H_INCLUDED
#define CNN_H_INCLUDED

#include "activations.h"

namespace cnn {
    template <typename T, const size_t OC, const size_t IC, const size_t K, const size_t H, const size_t W,
              const T* _weights, const T* _biases, T (*Activation)(T x)>
    class Conv2D {
      private:
        const int D = K / 2;
        const std::array<T, OC * IC * K * K>& weights =
            *reinterpret_cast<const std::array<T, OC * IC * K * K>*>(_weights);
        const std::array<T, OC>& biases = *reinterpret_cast<const std::array<T, OC>*>(_biases);

      public:
        std::array<T, OC * H * W> output;

        Conv2D() {}

        void forward(const std::array<T, IC * H * W>& input) {
            for (size_t outChannel = 0; outChannel < OC; ++outChannel) {
                for (int y = 0; y < (int)H; ++y) {
                    for (int x = 0; x < (int)W; ++x) {
                        T sum = biases[outChannel];
                        for (size_t inChannel = 0; inChannel < IC; ++inChannel) {
                            for (int ky = std::max(0, D - y); ky < std::min((int)K, (int)H + D - y); ++ky) {
                                int yIn = y + ky - D;
                                for (int kx = std::max(0, D - x); kx < std::min((int)K, (int)W + D - x); ++kx) {
                                    int xIn = x + kx - D;
                                    sum += input[inChannel * H * W + yIn * W + xIn] *
                                           weights[outChannel * IC * K * K + inChannel * K * K + ky * K + kx];
                                }
                            }
                        }
                        output[outChannel * H * W + y * W + x] = Activation(sum);
                    }
                }
            }
        }
    };

    template <typename T, const size_t C, const size_t K, const size_t W, const size_t H, const T* _weightsIn,
              const T* _biasesIn, const T* _weightsOut, const T* _biasesOut>
    class ResidualBlock {
      private:
        Conv2D<T, C, C, K, H, W, _weightsIn, _biasesIn, activations::ReLU> inLayer;
        Conv2D<T, C, C, K, H, W, _weightsOut, _biasesOut, activations::Linear> outLayer;

      public:
        std::array<T, C * H * W> output;

        ResidualBlock() {}

        void forward(const std::array<T, C * H * W>& input) {
            inLayer.forward(input);
            outLayer.forward(inLayer.output);

            for (size_t i = 0; i < C * H * W; ++i) {
                output[i] = activations::ReLU(input[i] + outLayer.output[i]);
            }
        }
    };
} // namespace cnn

#endif // CNN_H_INCLUDED
