import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = pd.read_csv('P4/iris.csv')

# Separar las características (X) y las etiquetas (y)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

# Convertir las etiquetas a valores numéricos
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(-1, 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
y = onehot_encoded

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un conjunto de datos personalizado para PyTorch
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    
# Crear conjuntos de datos y dataloaders
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

print(len(train_dataset))
print(train_dataset[0])

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
train_dataloader2 = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Definir la arquitectura del modelo
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(4, 200)  
        self.fc2 = nn.Linear(200, 2000)
        self.fc3 = nn.Linear(2000, 200)
        self.fc4 = nn.Linear(200, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        #x = self.sigmoid(x)
        return x

# Crear una instancia de la red neuronal
net = NeuralNetwork()
#criterion = nn.MSELoss()
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Entrenar el modelo
net.train()
num_epochs = 200
for epoch in range(num_epochs):
    for X, y in train_dataloader:
        X = X.type(torch.float32)
        y = y.type(torch.float32)
        y_pred = net(X)
        #loss = criterion(y_pred[:,0], y)
        loss = criterion(y_pred, torch.argmax(y, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch + 1}, loss = {loss.item():.4f}')

net.eval()

X_train_torch2, y_train_torch2 = next(iter(train_dataloader2))
X_train_torch2 = X_train_torch2.type(torch.float32)
y_train_torch2 = y_train_torch2.type(torch.float32)
y_train_pred_torch2 = net(X_train_torch2)
#loss_train2 = criterion(y_train_pred_torch2[:, 0], y_train_torch2).item()
loss_train2 = criterion(y_train_pred_torch2, torch.argmax(y_train_torch2, dim=1)).item()
print(f'CEL en entrenamiento = {loss_train2:.4f}')

X_test_torch, y_test_torch = next(iter(test_dataloader))
X_test_torch = X_test_torch.type(torch.float32)
y_test_torch = y_test_torch.type(torch.float32)
y_test_pred_torch = net(X_test_torch)
#loss_test = criterion(y_test_pred_torch[:, 0], y_test_torch).item()
loss_test = criterion(y_test_pred_torch, torch.argmax(y_test_torch, dim=1)).item()
print(f'CEL en datos de prueba = {loss_test:.4f}')


