# NNUE 3

## Summary

This document contains information about Pingu's NNUE. It is the sequel to [nnue_2.md](/docs/nnue_2.md) and it describes the process of training the NNUE for [Pingu 5.0.0](https://github.com/WillChing01/Pingu/releases/tag/v5.0.0).

The transition from Pingu 4.0.0 to Pingu 5.0.0 involved training two successive models of increasing complexity, leading to cumulative gains of ~170 elo at short time controls. In addition to the changes in model architecture, several large improvements were made to the machine learning pipeline, streamlining the process of data generation, preprocessing, and training.

## Contents

- [Summary](#summary)
- [Model Architecture](#model-architecture)
  - [Initial Network](#initial-network)
  - [Final Network](#final-network)
- [Data Generation](#data-generation)
- [Data Processing](#data-processing)

## Model Architecture

Pingu 5.0.0 introduced a Half-KA architecture to the engine's NNUE.

The input-layer consists of two perspective networks (our perspective, and our opponent's perspective). Each perspective network has an input layer consisting of 45056 binary input features.

```math
\text{our\_input\_layer}[64 \cdot 11 \cdot 64 \cdot i + 64 \cdot j + k] =
\begin{cases}
1, & \text{if our king is on square } i \text{ and piece } j \text{ is on square } k \\
0, & \text{otherwise}
\end{cases}
\\\\[1em]
\text{their\_input\_layer}[64 \cdot 11 \cdot 64 \cdot i + 64 \cdot j + k] =
\begin{cases}
1, & \text{if their king is on square } i \text{ and piece } j \text{ is on square } k \\
0, & \text{otherwise}
\end{cases}
```

Half-KA gives the network separate weights for each king square. This is useful because evaluation often depends heavily on king position. The input layer is extremely sparse, having a maximum of 31 active features in an input layer of 45046. This allows for fast recomputation when the friendly (from that perspective) king's position changes. In addition, the one-hot encoding allows for fast incremental updates of the first hidden layer when another piece moves.

The architecture of the final model used in Pingu 5.0.0 is shown below. Each layer is fully connected. The initial network was similar except it only had one set of weights before the final output.

<div align="center">
    <img src="img/nnue_architecture.png"/>
</div>
