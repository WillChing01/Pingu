#ifndef LINEAR_H_INCLUDED
#define LINEAR_H_INCLUDED

namespace linear {
    template <typename T, const size_t O, const size_t I>
    using Weights = std::array<T, O * I>;

    template <typename T, const size_t O>
    using Biases = std::array<T, O>;

    template <typename T, const size_t O, const size_t I, const Weights<T, O, I>& weights, const Biases<T, O>& biases,
              T (*Activation)(T x)>
    class Linear {
      public:
        std::array<T, O> output;

        Linear() {}

        void forward(const std::array<T, I>& input) {
            for (size_t i = 0; i < O; ++i) {
                output[i] = biases[i];
                for (size_t j = 0; j < I; ++j) {
                    output[i] += input[j] * weights[i * I + j];
                }
                output[i] = Activation(output[i]);
            }
        }
    };
} // namespace linear

#endif // LINEAR_H_INCLUDED
