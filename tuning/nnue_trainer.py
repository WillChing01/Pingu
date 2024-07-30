import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

INPUT_DTYPE = np.short
LABEL_DTYPE = np.float32

INPUT_TENSOR_DTYPE = torch.short
LABEL_TENSOR_DTYPE = torch.float32

"""
Dataset and dataloader
"""

class ChessDataset(Dataset):
    def __init__(self, inputs_file, inputs_shape, labels_file, labels_shape, indices_file):
        self.inputs_file = inputs_file
        self.inputs_shape = inputs_shape
        self.labels_file = labels_file
        self.labels_shape = labels_shape
        self.indices = np.load(indices_file, allow_pickle = True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sparse_inputs = np.memmap(self.inputs_file, mode = "r", dtype = INPUT_DTYPE, shape = self.inputs_shape)
        labels = np.memmap(self.labels_file, mode = "r", dtype = LABEL_DTYPE, shape = self.labels_shape)
        index = self.indices[idx]

        x = torch.tensor(sparse_inputs[index], dtype = INPUT_TENSOR_DTYPE)
        y = torch.tensor([labels[index][0]], dtype = LABEL_TENSOR_DTYPE)
        z = torch.tensor([labels[index][1]], dtype = LABEL_TENSOR_DTYPE)

        sparse_inputs._mmap.close()
        labels._mmap.close()

        return x, y, z

class ChessChunkDataset(Dataset):
    def __init__(self, inputs_file, labels_file, start_index, finish_index, input_count):
        self.len = finish_index - start_index
        inputs_memmap = np.memmap(inputs_file, mode = "r", dtype = np.short, shape = (input_count, 32))
        labels_memmap = np.memmap(labels_file, mode = "r", dtype = np.short, shape = (input_count, 1))
        self.inputs = torch.from_numpy(inputs_memmap[start_index:finish_index])
        self.labels = torch.from_numpy(labels_memmap[start_index:finish_index])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.inputs[idx].long(), self.labels[idx].float()

"""
Model definition

Structure: fully connected layers

(Input)    (Hidden)    (Hidden)    (Output)
  768   ->    64    ->     8    ->    1
       cReLu       cReLu      Linear
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

class NeuralNetwork(nn.Module):
    def __init__(self, input_count, l1_count, l2_count, output_count):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_count, l1_count),
            ClippedReLU(),
            nn.Linear(l1_count, l2_count),
            ClippedReLU(),
            nn.Linear(l2_count, output_count),
            Scale(512)
        )
        print("Created network.")

        self.net.apply(self.init_weights)
        print("Initialized weights and biases.")

    def init_weights(self, x):
        if type(x) == nn.Linear:
            torch.nn.init.kaiming_normal_(x.weight, nonlinearity='relu')
            torch.nn.init.zeros_(x.bias)

    def forward(self, x):
        return self.net(x)

"""
Training loop
"""

def custom_loss(output, targetEval, targetResult):
    K = 0.00475
    GAMMA = 0.8
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
        inputs = torch.zeros(batch_size, input_count, device = device)
        inputs = inputs.scatter_(dim = -1, index = x.long(), value = 1)

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
            inputs = torch.zeros(batch_size, input_count, device = device)
            inputs = inputs.scatter_(dim = -1, index = x.long(), value = 1)

            output = model(inputs)
            loss = loss_fn(output, y.to(device, non_blocking = True), z.to(device, non_blocking = True))

            validation_loss += loss.item() * batch_size / size

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print(f"Validation Loss: {validation_loss:>8f}")
    return validation_loss

def main():

    LEARNING_RATE = 0.001
    EPOCHS = 10000

    BATCH_SIZE = 1024
    NUM_WORKERS = 4

    SAVED_MODEL_FILE = "saved_model.pth"

    INPUT_COUNT = 768
    L1_COUNT = 64
    L2_COUNT = 8
    OUTPUT_COUNT = 1

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device", device)

    inputs_file = ""
    inputs_shape = (0, 32)

    labels_file = ""
    labels_shape = (0, 2)

    training_indices_file = "training_indices.npy"
    validation_indices_file = "validation_indices.npy"

    model = NeuralNetwork(INPUT_COUNT, L1_COUNT, L2_COUNT, OUTPUT_COUNT).to(device)

    try:
        print("Loading saved model...")
        model.load_state_dict(torch.load(SAVED_MODEL_FILE))
    except:
        print("Failed to load saved model.")

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    training_dataset = ChessDataset(inputs_file, inputs_shape, labels_file, labels_shape, training_indices_file)
    training_dataloader = DataLoader(training_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)

    validation_dataset = ChessDataset(inputs_file, inputs_shape, labels_file, labels_shape, validation_indices_file)
    validation_dataloader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")

        training_loss = training_loop(training_dataloader, model, custom_loss, optimizer, device, INPUT_COUNT)
        validation_loss = validation_loop(validation_dataloader, model, custom_loss, device, INPUT_COUNT)

        # epoch complete - save model.
        print("Saving model...")
        torch.save(model.state_dict(), ('epoch_'+str(epoch+1)+'_tloss_'+str(round(training_loss, 6))+'_vloss_'+str(round(validation_loss, 6))).replace('.',',')+'.pth')

    print("Done!")

if __name__ == '__main__':
    main()
