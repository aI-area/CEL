import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import math
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    def __init__(self, data, targets, seq_length):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.targets[index+self.seq_length-1]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_length + 1

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, dim_model, num_heads, num_layers, output_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        self.encoder = nn.Linear(input_dim, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        transformer_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)
        self.decoder = nn.Linear(dim_model, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[-1]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Parameters
input_dim = 7
seq_length = 5
dim_model = 512
num_heads = 8
num_layers = 3
output_dim = 1
batch_size = 9
epochs = 100


df = pd.read_csv('africa_mpox_data.csv')
num_tasks = 10

def train_test_split_tasks(data, num_tasks):
    task_size = len(data) // num_tasks
    task_data = []
    for task in range(num_tasks):
        task_data.append(data[task * task_size: (task + 1) * task_size])
    return task_data


def load_data(task_data, seq_length, test_size=0.2):
    X = task_data.drop(columns=['value', 'Date']).values
    y = task_data['value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataset = MyDataset(X_train, y_train, seq_length)
    test_dataset = MyDataset(X_test, y_test, seq_length)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False)

all_tasks = train_test_split_tasks(df, num_tasks)
for i, task in enumerate(all_tasks):
    print(f"Task {i+1}: Number of samples = {len(task)}")
all_data = [load_data(task, seq_length) for task in all_tasks]

# Define the model
model = TimeSeriesTransformer(input_dim, dim_model, num_heads, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

all_predictions = []
all_actuals = []
stored_performances = []
final_performances = []

train_data, test_data = all_data[0]
for inputs, labels in train_data:
    print(f"Batch input shape: {inputs.shape}, Batch label shape: {labels.shape}")
    break  # Just check the first batch


# Training
for task, (train_data, test_data) in enumerate(all_data):
    print(f"Training on task {task + 1}")

    # Training phase
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"Completed training for task {task + 1}")

    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_data:
            if inputs.size(0) > 0:
                outputs = model(inputs)
                if outputs.nelement() == 0:
                    print("Model did not generate any outputs.")
                    continue
                if outputs.shape[0] != labels.shape[0]:
                    outputs = outputs[:labels.shape[0]]
                predictions.extend(outputs.numpy().flatten().tolist())
                actuals.extend(labels.numpy().flatten().tolist())

    print(f"Actuals length: {len(actuals)}, Predictions length: {len(predictions)}")

    if len(actuals) > 0 and len(predictions) > 0:
        r2 = r2_score(actuals, predictions)
        rmse = math.sqrt(mean_squared_error(actuals, predictions))
        print(f'R-squared for task {task + 1}: {r2}')
        print(f'RMSE for task {task + 1}: {rmse}')
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
        stored_performances.append((r2, rmse))
    else:
        print(f"Insufficient data for evaluation in task {task + 1}")

# Re-evaluate the model on all tasks
for task, (train_data, test_data) in enumerate(all_data):
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            if outputs.nelement() == 0:
                print("Model did not generate any outputs.")
                continue
            if outputs.shape[0] != labels.shape[0]:
                outputs = outputs[:labels.shape[0]]
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_final = r2_score(actuals, predictions)
    rmse_final = math.sqrt(mean_squared_error(actuals, predictions))
    final_performances.append((r2_final, rmse_final))

# Calculate forgetting and memory stability
forgetting_r2 = [stored_performances[i][0] - final_performances[i][0] for i in range(num_tasks)]
forgetting_rmse = [stored_performances[i][1] - final_performances[i][1] for i in range(num_tasks)]

memory_stability_r2 = 1 - np.mean(forgetting_r2)
memory_stability_rmse = 1 - np.mean(forgetting_rmse)

print(f'Memory stability for R-squared: {memory_stability_r2}')
print(f'Memory stability for RMSE: {memory_stability_rmse}')