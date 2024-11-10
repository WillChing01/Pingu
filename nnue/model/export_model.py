import numpy as np
import torch
from nnue.trainer.nnue_trainer import NeuralNetwork

INPUT_COUNT = 768
L1_COUNT = 64
L2_COUNT = 8
OUTPUT_COUNT = 1

model_file = "saved_model.pth"

def main():
    model = NeuralNetwork(INPUT_COUNT, L1_COUNT, L2_COUNT, OUTPUT_COUNT)
    model.load_state_dict(torch.load(model_file))

    print("Successfully loaded saved model.")

    with open("../include/weights.h", "w") as f:
        f.write("#ifndef WEIGHTS_H_INCLUDED\n")
        f.write("#define WEIGHTS_H_INCLUDED\n")
        f.write("\n")
        f.write("#include <array>\n")
        f.write("\n")
        for layer in model.net:
            if type(layer) == torch.nn.Linear:
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy()

                weight_name = '_'.join([str(i) for i in weight.shape])
                bias_name = '_'.join([str(i) for i in bias.shape])

                f.write("const std::array<std::array<float, "+str(weight.shape[1])+">, "+str(weight.shape[0])+"> weights_"+weight_name+" =\n")
                f.write("{{\n")
                for i in range(weight.shape[0]):
                    f.write("\t{")
                    for j in range(weight.shape[1]):
                        f.write("{:.10f}".format(weight[i][j]))
                        if j < weight.shape[1] - 1: f.write(", ")
                    f.write("},\n")
                f.write("}};\n")
                f.write("\n")
                f.write("const std::array<float, "+str(bias.shape[0])+"> bias_"+bias_name+" =\n")
                f.write("{{\n")
                for i in range(bias.shape[0]):
                    f.write("\t")
                    f.write("{:.10f}".format(bias[i]))
                    f.write(",\n")
                f.write("}};\n")
                f.write("\n")
        f.write("#endif // WEIGHTS_H_INCLUDED\n")

    print("Successfully exported weights and biases.")

if __name__ == "__main__":
    main()
