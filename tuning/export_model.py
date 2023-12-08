import numpy as np
import torch
from nnue_trainer import NeuralNetwork

INPUT_COUNT = 768
L1_COUNT = 256
L2_COUNT = 32
OUTPUT_COUNT = 1

model_file = "saved_model.pth"

def main():
    model = NeuralNetwork(INPUT_COUNT, L1_COUNT, L2_COUNT, OUTPUT_COUNT)
    model.load_state_dict(torch.load(model_file))

    print("Successfully loaded saved model.")

    for layer in model.net:
        if type(layer) == torch.nn.Linear:
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()
            np.savetxt(
                "_nnue_weight_"+str(weight.shape).replace(' ', '')+".txt",
                weight,
                fmt = "%.18f",
                delimiter = ','
            )
            np.savetxt(
                "_nnue_bias_"+str(bias.shape).replace(' ', '')+".txt",
                bias,
                fmt = "%.18f",
                delimiter = ','
            )

    print("Successfully exported weights and biases.")

if __name__ == "__main__":
    main()
