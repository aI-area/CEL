import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style("whitegrid")

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions.squeeze()

def train_test_split_tasks(data, num_tasks):
    task_size = len(data) // num_tasks
    task_data = []
    for task in range(num_tasks):
        task_data.append(data[task * task_size: (task + 1) * task_size])
    return task_data

def load_data(task_data, test_size=0.2):
    X = task_data.drop(columns=['value', 'Date']).values
    y = task_data['value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True), DataLoader(MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Parameters
input_dim = 7
hidden_dim = 32
output_dim = 1
batch_size = 32
epochs = 100

# Load data
df = pd.read_csv('data_file.csv')
num_tasks = 10
all_tasks = train_test_split_tasks(df, num_tasks)
all_data = [(load_data(task)) for task in all_tasks]

# Initialize model
model = LSTM(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Evaluation.......")

stored_performances = []
final_performances = []
forgetting_r2 = []
forgetting_rmse = []


# Training loop
for task, (train_data, test_data) in enumerate(all_data):

    for epoch in range(epochs):
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate performance on the current task
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2 = r2_score(actuals, predictions)
    rmse = math.sqrt(mean_squared_error(actuals, predictions))
    print(f'R-squared for task {task + 1}: {r2}')


    # Store the performance for later comparison
    stored_performances.append((r2, rmse))

print("Reevaluation.......")
# Re-evaluate the model on all tasks
for task, (train_data, test_data) in enumerate(all_data):
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_final = r2_score(actuals, predictions)
    rmse_final = math.sqrt(mean_squared_error(actuals, predictions))
    final_performances.append((r2_final, rmse_final))
    print(f'R-squared for task {task + 1}: {r2_final}')
print("Forgetting.......")
# Compute forgetting
for task in range(num_tasks):
    forgetting_r2.append(stored_performances[task][0] - final_performances[task][0])
    forgetting_rmse.append(stored_performances[task][1] - final_performances[task][1])
    print(f'Forgetting for R-squared of task {task + 1}: {forgetting_r2[-1]}')


memory_stability_r2 = 1 - np.mean(forgetting_r2)

memory_stability_rmse = 1 - np.mean(forgetting_rmse)

print(f'Memory stability for R-squared: {memory_stability_r2}')
