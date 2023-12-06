import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""

Dataset and dataloader

"""

class ChessDataset(Dataset):
    def __init__(self, file, input_count):
        self.input_count = input_count
        raw_data = []
        with open(file, "rb") as f:
            raw_data = pickle.load(f)
        self.indices = np.zeros((len(raw_data), 32), dtype = np.ushort)
        self.eval = np.zeros((len(raw_data), 1), dtype = np.short)

        for i in range(len(raw_data)):
            for j in range(len(raw_data[i][0])):
                self.indices[i][j] = raw_data[i][0][j]
            for j in range(len(raw_data[i][0]), 32):
                self.indices[i][j] = self.indices[i][0]
            self.eval[i][0] = raw_data[i][1]

        del raw_data

    def __len__(self):
        return len(self.eval)

    def __getitem__(self, idx):
        features = np.zeros(self.input_count, dtype = np.ubyte)
        features[self.indices[idx]] = 1
        return torch.Tensor(features), torch.Tensor(self.eval[idx])

"""

Model definition

Structure: fully connected layers

(Input)    (Hidden)    (Hidden)    (Output)
  768   ->   256    ->    32    ->    1
       ReLu        ReLu       Linear

"""

class NeuralNetwork(nn.Module):
    def __init__(self, input_count, l1_count, l2_count, output_count):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_count, l1_count),
            nn.ReLU(),
            nn.Linear(l1_count, l2_count),
            nn.ReLU(),
            nn.Linear(l2_count, output_count),
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

def custom_loss(output, target):
    K = 0.00475
    output_scaled = torch.sigmoid(K*output)
    target_scaled = torch.sigmoid(K*target)
    return torch.mean((output_scaled - target_scaled)**2)

def training_loop(dataloader, model, loss_fn, optimizer, device):
    """
        Optimize model parameters.
    """

    size = len(dataloader.dataset)

    model.train()
    for batch, (x, y) in enumerate(dataloader):
        output = model(x.to(device, non_blocking = True))
        loss = loss_fn(output, y.to(device, non_blocking = True))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation_loop(dataloader, model, loss_fn, device):
    """
        Run model with validation data.
    """

    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            output = model(x.to(device, non_blocking = True))
            validation_loss += loss_fn(output, y.to(device, non_blocking = True)).item()

    validation_loss /= num_batches
    print(f"Validation Loss: {validation_loss:>8f}")
    return validation_loss

def main():

    LEARNING_RATE = 0.001
    EPOCHS = 10000

    BATCH_SIZE = 1024
    NUM_WORKERS = 4

    SAVED_MODEL_FILE = "saved_model.pth"

    INPUT_COUNT = 768
    L1_COUNT = 256
    L2_COUNT = 32
    OUTPUT_COUNT = 1

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device", device)

    training_file = "training_data.pickle"
    validation_file = "validation_data.pickle"

    training_data = ChessDataset(training_file, INPUT_COUNT)
    validation_data = ChessDataset(validation_file, INPUT_COUNT)

    training_dataloader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)
    validation_dataloader = DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)

    model = NeuralNetwork(INPUT_COUNT, L1_COUNT, L2_COUNT, OUTPUT_COUNT).to(device)

    try:
        print("Loading saved model...")
        model.load_state_dict(torch.load(SAVED_MODEL_FILE))
    except:
        print("Failed to load saved model.")

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loop(training_dataloader, model, custom_loss, optimizer, device)
        validation_loss = validation_loop(validation_dataloader, model, custom_loss, device)
        print("Saving model...")
        torch.save(model.state_dict(), 'epoch_'+str(epoch)+'_loss_'+str(round(validation_loss, 10)).replace('.',',')+'.pth')

    print("Done!")

if __name__ == '__main__':
    main()
