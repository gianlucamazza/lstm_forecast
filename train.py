import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(_model: nn.Module, _train_loader: DataLoader, _val_loader: DataLoader, num_epochs: int = 100) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_model = None
    best_val_loss = float('inf')

    _model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for X_batch, y_batch in _train_loader:
            x_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = _model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        _model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in _val_loader:
                x_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = _model(x_val_batch)
                val_loss += criterion(val_outputs, y_val_batch.unsqueeze(1)).item()

        train_loss /= len(_train_loader)
        val_loss /= len(_val_loader)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _model.state_dict()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        _model.train()

    # Save the best model
    if best_model is not None:
        torch.save(best_model, 'model.pth')


def evaluate_model(_ticker: str, _model: nn.Module, _x: np.ndarray, _y: np.ndarray, _scaler: StandardScaler) -> None:
    _model.eval()
    with torch.no_grad():
        predictions = _model(torch.tensor(_x, dtype=torch.float32).to(device)).cpu().numpy()
        predictions_reshaped = np.zeros((_y.shape[0], 3))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]
        y_true = _scaler.inverse_transform(np.concatenate([_y.reshape(-1, 1), np.zeros((_y.shape[0], 2))], axis=1))[:,
                 0]

    plt.figure(figsize=(14, 7))
    plt.title(f'{_ticker} Stock Price Prediction')
    plt.plot(y_true, label='True Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.show()
