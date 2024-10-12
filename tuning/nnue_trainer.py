import numpy as np
import random
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import DATASET_DTYPE

"""
Dataset and dataloader
"""

class ChessDataset(Dataset):
    def __init__(self, data_file):
        self.shape = np.array(data_file.split(".")[0].split("_")[-2:], dtype = np.int32)
        dataset = np.memmap(data_file, mode = "r", dtype = DATASET_DTYPE, shape = (self.shape[0], self.shape[1]))
        self.inputs, self.evaluations, self.results = torch.tensor_split(torch.tensor(dataset, dtype = torch.short), (32, 33), dim = 1)
        self.results = self.results.float() * 0.5
        dataset._mmap.close()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.evaluations[idx], self.results[idx]

"""
Model definition

(45056 -> 64 -> cReLU(64)) x 2 -> 1
"""

class ClippedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)
    
class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.mul(x, self.factor)

class PerspectiveNetwork(nn.Module):
    def __init__(self, input_count, output_count):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_count, output_count),
            ClippedReLU()
        )

    def forward(self, x):
        return self.net(x)

class NeuralNetwork(nn.Module):
    def __init__(self, input_count, l1_count, output_count):
        super().__init__()

        self.perspectiveNets = [
            PerspectiveNetwork(input_count, l1_count),
            PerspectiveNetwork(input_count, l1_count),
        ]

        self.net = nn.Sequential(
            nn.Linear(2 * l1_count, output_count),
            Scale(512)
        )

        self.perspectiveNets[0].apply(self.init_weights)
        self.perspectiveNets[1].apply(self.init_weights)
        self.net.apply(self.init_weights)

    def init_weights(self, x):
        if type(x) == nn.Linear:
            torch.nn.init.kaiming_normal_(x.weight, nonlinearity='relu')
            torch.nn.init.zeros_(x.bias)

    def forward(self, x, y):
        return self.net(torch.cat((self.perspectiveNets[0](x), self.perspectiveNets[1](y))))

"""
Training loop
"""

def custom_loss(output, targetEval, targetResult):
    K = 0.00475
    GAMMA = 0.5
    output_scaled = torch.sigmoid(K*output)
    target_scaled = GAMMA * torch.sigmoid(K*targetEval) + (1. - GAMMA) * targetResult
    return torch.mean((output_scaled - target_scaled)**2)

def training_loop(dataloader, model, loss_fn, optimizer, device, input_count):
    """
        Optimize model parameters.
    """

    model.train()
    size = len(dataloader.dataset)
    batch_size = 0
    training_loss = 0

    for batch, (x, y, z) in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = x.size(0)
        x = x.to(device, non_blocking = True)
        inputs = torch.zeros(batch_size, input_count, device = device).scatter_(dim = -1, index = x.long(), value = 1)

        output = model(inputs)
        loss = loss_fn(output, y.to(device, non_blocking = True), z.to(device, non_blocking = True))

        training_loss += loss.item() * batch_size / size

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print(f"Training Loss: {training_loss:>8f}")
    return training_loss

def validation_loop(dataloader, model, loss_fn, device, input_count):
    """
        Run model with validation data.
    """

    model.eval()
    size = len(dataloader.dataset)
    batch_size = 0
    validation_loss = 0

    with torch.no_grad():
        for batch, (x, y, z) in enumerate(dataloader):
            batch_size = x.size(0)
            x = x.to(device, non_blocking = True)
            inputs = torch.zeros(batch_size, input_count, device = device).scatter_(dim = -1, index = x.long(), value = 1)

            output = model(inputs)
            loss = loss_fn(output, y.to(device, non_blocking = True), z.to(device, non_blocking = True))

            validation_loss += loss.item() * batch_size / size

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print(f"Validation Loss: {validation_loss:>8f}")
    return validation_loss

def main():
    EPOCHS = 10000

    BATCH_SIZE = 1024
    NUM_WORKERS = 6

    SAVED_MODEL_FILE = ""

    INPUT_COUNT = 768
    L1_COUNT = 64
    L2_COUNT = 8
    OUTPUT_COUNT = 1

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device", device)

    model = NeuralNetwork(INPUT_COUNT, L1_COUNT, L2_COUNT, OUTPUT_COUNT).to(device)

    try:
        print("Loading saved model...")
        model.load_state_dict(torch.load(SAVED_MODEL_FILE))
        start_epoch = int(SAVED_MODEL_FILE.split("_")[1])
    except:
        print("Failed to load saved model.")
        start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters())

    training_dir = os.getcwd() + "/training/"
    validation_dir = os.getcwd() + "/validation/"

    training_chunks = []
    validation_chunks = []

    for (root, dir, files) in os.walk(training_dir):
        training_chunks = files
        break
    for (root, dir, files) in os.walk(validation_dir):
        validation_chunks = files
        break

    num_training = 0
    num_validation = 0

    for chunk_file in training_chunks:
        num_training += int(chunk_file.split(".")[0].split("_")[-2])
    for chunk_file in validation_chunks:
        num_validation += int(chunk_file.split(".")[0].split("_")[-2])

    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")

        print("Training...")
        training_loss = 0

        print("Shuffling chunks...")
        random.shuffle(training_chunks)

        for i in range(len(training_chunks)):
            print(f"Training chunk {i+1} / {len(training_chunks)} : {training_chunks[i]}")
            chunk_dataset = ChessDataset(training_dir + training_chunks[i])
            dataloader = DataLoader(chunk_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)

            chunk_loss = training_loop(dataloader, model, custom_loss, optimizer, device, INPUT_COUNT)
            training_loss += chunk_loss * len(chunk_dataset) / num_training

        print("Validating...")
        validation_loss = 0

        for i in range(len(validation_chunks)):
            print(f"Validation chunk {i+1} / {len(validation_chunks)} : {validation_chunks[i]}")
            chunk_dataset = ChessDataset(validation_dir + validation_chunks[i])
            dataloader = DataLoader(chunk_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = True)

            chunk_loss = validation_loop(dataloader, model, custom_loss, device, INPUT_COUNT)
            validation_loss += chunk_loss * len(chunk_dataset) / num_validation

        print(f"Training loss : {training_loss}")
        print(f"Validation loss : {validation_loss}")

        # epoch complete - save model.
        print("Saving model...")
        torch.save(model.state_dict(), ('epoch_'+str(epoch+1)+'_tloss_'+str(round(training_loss, 6))+'_vloss_'+str(round(validation_loss, 6))).replace('.',',')+'.pth')

    print("Done!")

if __name__ == '__main__':
    main()
