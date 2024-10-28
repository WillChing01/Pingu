import numpy as np
import torch
import nnue.trainer.nnue_trainer as nnue_trainer
from nnue.parsed.parse_data import fenToSparse

model_file = "saved_model.pth"

def main():
    model = nnue_trainer.NeuralNetwork(768, 64, 8, 1)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    print("Successfully loaded saved model.")

    while True:
        fen = input("fen: ")
        sparse = fenToSparse(fen.split(' ')[0])
        x = np.zeros(768)
        x[sparse] = 1
        print(model.forward(torch.Tensor(x)))

if __name__ == "__main__":
    main()
