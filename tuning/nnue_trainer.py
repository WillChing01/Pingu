import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device", device)

"""

Dataset and dataloader

"""

BATCH_SIZE = 1024

training_file = "training_data.pickle"
validation_file = "validation_data.pickle"

class ChessDataset(Dataset):
    def __init__(self, file):
        self.data = []
        with open(file, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert fen string into numpy tensor.
        features = torch.zeros(INPUT_COUNT)

        for x in self.data[idx][0]:
            features[x] = 1

        label = torch.tensor([self.data[idx][1]])

        return features, label

training_data = ChessDataset(training_file)
validation_data = ChessDataset(validation_file)

training_dataloader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle = True)

"""

Model definition

Structure: fully connected layers

(Input)    (Hidden)    (Hidden)    (Output)
  768   ->   256    ->    32    ->    1
       ReLu        ReLu       Linear

"""

SAVED_MODEL_FILE = "saved_model.pth"

INPUT_COUNT = 768
L1_COUNT = 256
L2_COUNT = 32
OUTPUT_COUNT = 1

K = 0.00475

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_COUNT, L1_COUNT),
            nn.ReLU(),
            nn.Linear(L1_COUNT, L2_COUNT),
            nn.ReLU(),
            nn.Linear(L2_COUNT, OUTPUT_COUNT),
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

model = NeuralNetwork().to(device)

try:
    print("Loading saved model...")
    model.load_state_dict(torch.load(SAVED_MODEL_FILE))
except:
    print("Failed to load saved model.")

"""

Training loop

"""

LEARNING_RATE = 0.001
EPOCHS = 100

def custom_loss(output, target):
    output_scaled = torch.sigmoid(K*output)
    target_scaled = torch.sigmoid(K*target)
    return torch.mean((output_scaled - target_scaled)**2)

def training_loop(dataloader, model, loss_fn, optimizer):
    """
        Optimize model parameters.
    """

    size = len(dataloader.dataset)

    model.train()
    for batch, (x, y) in enumerate(dataloader):
        output = model(x.to(device))
        loss = loss_fn(output, y.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation_loop(dataloader, model, loss_fn):
    """
        Run model with validation data.
    """

    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            output = model(x.to(device))
            validation_loss += loss_fn(output, y.to(device)).item()

    validation_loss /= num_batches
    print(f"Validation Loss: {validation_loss:>8f}")

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    training_loop(training_dataloader, model, custom_loss, optimizer)
    validation_loop(validation_dataloader, model, custom_loss)
    print("Saving model...")
    torch.save(model.state_dict(), 'epoch_'+str(epoch)+'.pth')

print("Done!")
