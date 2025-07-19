#ifndef LINEAR_H_INCLUDED
#define LINEAR_H_INCLUDED

namespace linear {
    template <typename T, const size_t O, const size_t I, const T* _weights, const T* _biases, T (*Activation)(T x)>
    class Linear {
      private:
        const std::array<T, O * I>& weights = *reinterpret_cast<const std::array<T, O * I>*>(_weights);
        const std::array<T, O>& biases = *reinterpret_cast<const std::array<T, O>*>(_biases);

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
