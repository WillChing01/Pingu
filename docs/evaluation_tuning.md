# Evaluation Tuning

## Summary

This page is intended as a summary of the tuning process for Pingu's hand-crafted evaluation.

Full credit goes to Andrew Grant's [tuning guide](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf) - the equations here are quoted verbatim from his guide.

> __N.B. Following the introduction of NNUE to Pingu, this tuning process is no longer used and can only be found in Pingu 2.0.0.__

## Contents

Pingu's HCE (hand-crafted evaluation) consists of a tapered evaluation - the evaluation varies linearly depending on how many pieces are on the board. As such, each evaluation term (material, pst, etc.) has two values - one for the middlegame and one for the endgame.

Game phases $` \rho `$ are defined for the middlegame, $` mg `$, and the endgame, $` eg `$, and they specify how close we are to the endgame.

The phases must obey $` \rho_{mg} + \rho_{eg} = 1 `$

Static evaluation of a position can be expressed as

```math
E = \rho_{mg} \underline{L}_{mg} \cdot \left( \underline{C}_{w} - \underline{C}_{b} \right)
  + \rho_{eg} \underline{L}_{eg} \cdot \left( \underline{C}_{w} - \underline{C}_{b} \right)
```

$` C_{w} `$ and $` C_{b} `$ are the 'feature' vectors for white and black respectively. In Pingu's HCE the feature vector encodes material, piece placement and rook/bishop mobility.

$` L `$ are the evaluation weights (material values, mobility values etc.), and the dot product $` L \cdot C `$ yields the evaluation for middlegame/endgame.

We define the error $` \epsilon `$ when evaluating a dataset consisting of $` N `$ positions as

```math
\epsilon = \frac{1}{N} \sum_{i=1}^{N} \left( R_{i} - \sigma \left( E_{i} \right) \right) ^ 2
```

where $` R_{i} `$ is the result (1 - white win, 0.5 - draw, 0 - black win) of the game $` i `$ from which the position originated, and $` \sigma \left(E \right) `$ is the sigmoid function

```math
\sigma \left( E \right) = \frac{1}{1 +  e^{-KE}}
```

for an arbitrary constant $` K `$.

In Pingu we have $` K = 0.00475 `$. This was chosen to minimize the error of the original unoptimized HCE, and it hasn't been changed since.

For a simple linear evaluation, we can express the derivative of the error as

```math
\frac{\partial \epsilon}{\partial\underline{L}_{mg}} =
    - \alpha \sum_{j=1}^{N} \left( R_{j} - \sigma \left( E_{j} \right) \right) \cdot
    \sigma \left( E_{j} \right) \cdot
    \left( 1 - \sigma \left( E_{j} \right) \right) \cdot
    \rho_{mg,j} \left ( \underline{C}_{w,j} - \underline{C}_{b,j} \right)
```

```math
\frac{\partial \epsilon}{\partial\underline{L}_{eg}} =
    - \alpha \sum_{j=1}^{N} \left( R_{j} - \sigma \left( E_{j} \right) \right) \cdot
    \sigma \left( E_{j} \right) \cdot
    \left( 1 - \sigma \left( E_{j} \right) \right) \cdot
    \rho_{eg,j} \left ( \underline{C}_{w,j} - \underline{C}_{b,j} \right)
```

for the middlegame and endgame weights respectively. All constants have been absorbed into an arbitrary term $` \alpha > 0 `$ which can be scaled to adjust the learning rate of gradient descent.
