#ifndef ACTIVATIONS_H_INCLUDED
#define ACTIVATIONS_H_INCLUDED

namespace activations {
    template <typename T>
    inline T Linear(T x) {
        return x;
    }

    template <typename T>
    inline T ReLU(T x) {
        return std::max(T(0), x);
    }

    template <typename T>
    inline T Sigmoid(T x) {
        return T(1.) / (T(1.) + std::exp(-x));
    }
} // namespace activations

#endif // ACTIVATIONS_H_INCLUDED
