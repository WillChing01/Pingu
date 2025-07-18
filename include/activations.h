#ifndef ACTIVATIONS_H_INCLUDED
#define ACTIVATIONS_H_INCLUDED

namespace activations {
    template <typename T>
    using Activation = T (*)(T x);

    template <typename T>
    inline T Linear(T x) {
        return x;
    }

    template <typename T>
    inline T ReLU(T x) {
        return std::min(std::max(x, T(0)), T(1));
    }

    template <typename T>
    inline T Sigmoid(T x) {
        return T(1.) / (T(1.) + std::exp(-x));
    }
} // namespace activations

#endif // ACTIVATIONS_H_INCLUDED
