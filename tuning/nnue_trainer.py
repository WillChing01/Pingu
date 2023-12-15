import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""

Dataset and dataloader

"""

class ChessDataset(Dataset):
    def __init__(self, inputs_file, labels_file, dataset_size, input_count):
        self.inputs_file = inputs_file
        self.labels_file = labels_file
        self.len = dataset_size
        self.input_count = input_count

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inputs = np.memmap(self.inputs_file, mode = "r", dtype = np.short, shape = (self.len, 32))
        labels = np.memmap(self.labels_file, mode = "r", dtype = np.short, shape = (self.len, 1))
        return torch.LongTensor(inputs[idx]), torch.Tensor(labels[idx])

"""

Model definition

Structure: fully connected layers

(Input)    (Hidden)    (Hidden)    (Output)
  768   ->   256    ->    32    ->    1
       ReLu        ReLu       Linear

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

def custom_loss(output, target):
    K = 0.00475
    output_scaled = torch.sigmoid(K*output)
    target_scaled = torch.sigmoid(K*target)
    return torch.mean((output_scaled - target_scaled)**2)

def training_loop(dataloader, model, loss_fn, optimizer, device, input_count):
    """
        Optimize model parameters.
    """

    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    training_loss = 0

    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()

        x = x.to(device, non_blocking = True)
        inputs = torch.zeros(x.size(0), input_count, device = device)
        inputs = inputs.scatter_(dim = -1, index = x, value = 1)

        # output = model(x.to(device, non_blocking = True))

        output = model(inputs)
        loss = loss_fn(output, y.to(device, non_blocking = True))

        training_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_loss /= num_batches
    return training_loss

def validation_loop(dataloader, model, loss_fn, device, input_count):
    """
        Run model with validation data.
    """

    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking = True)
            inputs = torch.zeros(x.size(0), input_count, device = device)
            inputs = inputs.scatter_(dim = -1, index = x, value = 1)

            # output = model(x.to(device, non_blocking = True))

            output = model(inputs)
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

    training_inputs = "training_input_88882000_32.dat"
    training_labels = "training_labels_88882000_1.dat"

    validation_inputs = "validation_input_4678000_32.dat"
    validation_labels = "validation_labels_4678000_1.dat"

    training_data = ChessDataset(training_inputs, training_labels, 88882000, INPUT_COUNT)
    validation_data = ChessDataset(validation_inputs, validation_labels, 4678000, INPUT_COUNT)

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
        training_loss = training_loop(training_dataloader, model, custom_loss, optimizer, device, INPUT_COUNT)
        validation_loss = validation_loop(validation_dataloader, model, custom_loss, device, INPUT_COUNT)
        print("Saving model...")
        torch.save(model.state_dict(), ('epoch_'+str(epoch+1)+'_tloss_'+str(round(training_loss, 6))+'_vloss_'+str(round(validation_loss, 6))).replace('.',',')+'.pth')

    print("Done!")

if __name__ == '__main__':
    main()
