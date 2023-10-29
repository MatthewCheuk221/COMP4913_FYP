import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

path = "Model7.pth"
# Define neural network 
class Network(nn.Module):
    
    def __init__(self, input, output):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.layer5 = nn.Linear(100, 100)
        self.layer6 = nn.Linear(100, 100)
        self.layer7 = nn.Linear(100, output)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = F.relu(self.layer4(x3))
        x5 = F.relu(self.layer5(x4))
        x6 = F.relu(self.layer6(x5))
        x7 = self.layer7(x6)
        return x7


def train(model, num_epochs, train_loader, val_loader, optimizer, loss_fn):

    print("Start training")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0

        # Training Loop 
        for data in train_loader:
            inputs, outputs = data

            optimizer.zero_grad()
            train_loss = loss_fn(model(inputs), outputs)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item() 

        train_loss_value = running_train_loss / len(train_loader)

        validate_loss_value = test(model, val_loader, loss_fn)

        if validate_loss_value < float('inf'):
            torch.save(model.state_dict(), "./" + path)
            lowest_mse = validate_loss_value

        print(
            f'Completed training batch {epoch}. Training loss: {train_loss_value:.5f} Validation loss: {validate_loss_value:.5f}')


def test(model, loader, loss_fn):
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        model.eval()
        for data in loader:
            inputs, outputs = data
            loss = loss_fn(model(inputs), outputs)
            running_loss += loss.item() * outputs.size(0)
            total += outputs.size(0)

    return running_loss / total


def predict(model, loader):
    predicted = []
    with torch.no_grad():
        model.eval()
        for inputs in loader:
            predicted.append(model(inputs[0]))
    return torch.vstack(predicted).numpy()


def main():

    # Define input and output and create TensorDataset
    X = torch.FloatTensor(dataset.iloc[:, :-3].to_numpy())
    Y = torch.FloatTensor(dataset.iloc[:, -3:].to_numpy())
    data = TensorDataset(X, Y)

    # Split train, validate and test sets
    validate_set = int(len(X) * 0.2)
    test_set = int(len(X) * 0.3)
    train_set = len(X) - test_set - validate_set
    train_set, validate_set, test_set = random_split(data, [train_set, validate_set, test_set])
    # Read the data in batches and put into memory
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # Define model
    model = Network(X.size(1), Y.size(1))
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    epochs = 200
    train(model, epochs, train_loader, validate_loader, optimizer, loss_function)
    print('Finish training')
    # Load the model saved
    model = Network(X.size(1), Y.size(1))
    model.load_state_dict(torch.load(path))
    test_loss = test(model, test_loader, loss_function)
    print(f"Test loss: {test_loss:.5f}")

    # Prediction
    predict_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    predicted = predict(model, predict_loader)
    x_list = []
    y_list = []
    for x, y in predict_loader:
        x_list.append(x.numpy())
        y_list.append(y.numpy())
    df = pd.DataFrame(np.concatenate([np.vstack(x_list), np.vstack(y_list), predicted], axis=1))
    df.to_excel("Prediction7.xlsx", sheet_name='Sheet 1', index=False)

if __name__ == "__main__":
    #dataset = pd.read_csv('rfid1.csv')
    #dataset = pd.read_csv('rfid2.csv')
    #dataset = pd.read_csv('rfid3.csv')
    #dataset = pd.read_csv('rfid1&2.csv')
    #dataset = pd.read_csv('rfid1&3.csv')
    #dataset = pd.read_csv('rfid2&3.csv')
    dataset = pd.read_csv('rfid.csv')
    main()
