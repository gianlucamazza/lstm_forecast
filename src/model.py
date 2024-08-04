from typing import List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger

logger = setup_logger("model_logger", "logs/model.log")
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class PricePredictor(nn.Module):
    """
    A PyTorch neural network module for predicting prices using an LSTM model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        fc_output_size: int,
    ) -> None:
        """
        Initializes the PricePredictor model.
        """
        super(PricePredictor, self).__init__()
        logger.info(
            f"Initializing PricePredictor with input size: {input_size}, hidden size: {hidden_size}, "
            f"num layers: {num_layers}, dropout: {dropout}"
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, fc_output_size)

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        logger.debug("Forward pass completed.")
        return out

    def run_training_epoch(self, data_loader, criterion, optimizer):
        """Run a single training epoch using the given data loader."""
        self.train()
        total_loss = 0.0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def run_validation_epoch(self, data_loader, criterion):
        """Run a single validation epoch using the given data loader."""
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in data_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = self(X_val_batch)
                total_loss += criterion(val_outputs, y_val_batch).item()
        return total_loss / len(data_loader)


def load_model(symbol: str, path: str, model_params: dict, input_size: int) -> nn.Module:
    model = PricePredictor(
        input_size=input_size,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_layers"],
        dropout=model_params["dropout"],
        fc_output_size=model_params["fc_output_size"],
    ).to(device)
    model.device = device
    model_path = os.path.join(path, f"{symbol}_best_model.pth")
    logger.info(f"Model: {model}")
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model


def predict(
    model: nn.Module,
    x_data: np.ndarray,
    scaler: StandardScaler,
    future_days: int,
    features: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(model.device)
        predictions = model(x_tensor).cpu().numpy()

        predictions_reshaped = np.zeros((x_data.shape[0], len(features) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions_reshaped = np.pad(
            predictions_reshaped,
            ((0, 0), (0, len(scaler.scale_) - len(predictions_reshaped[0]))),
            "constant",
        )
        predictions = scaler.inverse_transform(predictions_reshaped)[:, 0]

        future_predictions = []
        for _ in range(future_days):
            x_tensor = torch.tensor(x_data[-1:], dtype=torch.float32).to(model.device)
            future_pred = model(x_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, x_data.shape[2]))
            new_row[0, 0] = future_pred
            new_row[0, 1:] = x_data[-1, -1, 1:]

            x_data = np.append(
                x_data, [np.append(x_data[-1][1:], new_row, axis=0)], axis=0
            )

        future_predictions_reshaped = np.zeros((future_days, len(features) + 1))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions_reshaped = np.pad(
            future_predictions_reshaped,
            ((0, 0), (0, len(scaler.scale_) - len(future_predictions_reshaped[0]))),
            "constant",
        )
        future_predictions = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions

