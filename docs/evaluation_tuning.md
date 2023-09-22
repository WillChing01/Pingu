# Evaluation Tuning

This page is intended as a summary of the tuning process for Pingu's hand-crafted evaluation.

Full credit goes to Andrew Grant's [tuning guide](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf) - the equations here are quoted verbatim from his guide.

Static evaluation of a position can be expressed as

$$
E = \rho_{mg} \underline{L}_{mg} \cdot \left( \underline{C}_w - \underline{C}_b \right)
  + \rho_{eg} \underline{L}_{eg} \cdot \left( \underline{C}_w - \underline{C}_b \right)
$$

We define the error $\epsilon$ when evaluating a dataset consisting of $N$ positions as

$$
\epsilon = \frac{1}{N} \sum_{i=1}^{N} \left( R_i - \sigma \left( E_i \right) \right) ^ 2
$$

where $ \sigma \left(E \right) $ is the sigmoid function

$$
\sigma \left( E \right) = \frac{1}{1 +  e ^{-KE}}
$$

for an arbitrary constant $ K $.

For a simple linear evaluation, we can express the derivative of the error as

$$
\frac{\partial \epsilon}{\partial\underline{L}_{mg}} =
    - \alpha \sum_{j=1}^{N} \left( R_j - \sigma \left( E_j \right) \right) \cdot
    \sigma \left( E_j \right) \cdot
    \left( 1 - \sigma \left( E_j \right) \right) \cdot
    \rho_{mg,j} \left ( \underline{C}_w - \underline{C}_b \right)
$$

$$
\frac{\partial \epsilon}{\partial\underline{L}_{eg}} =
    - \alpha \sum_{j=1}^{N} \left( R_j - \sigma \left( E_j \right) \right) \cdot
    \sigma \left( E_j \right) \cdot
    \left( 1 - \sigma \left( E_j \right) \right) \cdot
    \rho_{eg,j} \left ( \underline{C}_w - \underline{C}_b \right)
$$

for the middlegame and endgame weights respectively. All constants have been absorbed into an arbitrary term $ \alpha > 0 $ which can be scaled to adjust the learning rate of gradient descent.
