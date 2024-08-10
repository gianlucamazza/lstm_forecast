from typing import List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
from lstm_forecast.logger import setup_logger
from sklearn.preprocessing import StandardScaler

logger = setup_logger("model_logger", "logs/model.log")
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m: nn.Module) -> None:
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
    A neural network model for predicting prices using LSTM.
    
    Attributes:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of recurrent layers in the LSTM.
        lstm (nn.LSTM): The LSTM layer.
        dropout (nn.Dropout): The dropout layer.
        fc (nn.Linear): The fully connected output layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        fc_output_size: int,
    ) -> None:
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
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, fc_output_size)

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, fc_output_size)
        """
        # Add batch dimension if input is unbatched
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.size()
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        # Remove batch dimension if input was unbatched
        if batch_size == 1:
            out = out.squeeze(0)

        logger.debug("Forward pass completed.")
        return out

    def run_training_epoch(self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """
        Run a single training epoch.

        Args:
            data_loader (DataLoader): The data loader for training data.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer.

        Returns:
            float: The average loss for this epoch.
        """
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

    def run_validation_epoch(self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> float:
        """
        Run a single validation epoch.

        Args:
            data_loader (DataLoader): The data loader for validation data.
            criterion (nn.Module): The loss function.

        Returns:
            float: The average loss for this epoch.
        """
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in data_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = self(X_val_batch)
                total_loss += criterion(val_outputs, y_val_batch).item()
        return total_loss / len(data_loader)

def load_model(symbol: str, path: str, model_params: dict, input_size: int) -> nn.Module:
    model_path = os.path.join(path, f"{symbol}_best_model.pth")
    logger.info(f"Loading model from {model_path}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    hidden_size = state_dict['lstm.weight_hh_l0'].size(1)
    saved_input_size = state_dict['lstm.weight_ih_l0'].size(1)
    
    num_lstm_layers = sum(1 for key in state_dict.keys() if key.startswith('lstm.weight_ih_l'))
    
    fc_output_size = state_dict['fc.weight'].size(0)
    
    logger.info(f"Inferred model parameters - Input size: {saved_input_size}, Hidden size: {hidden_size}, "
                f"LSTM layers: {num_lstm_layers}, FC output size: {fc_output_size}")
    
    model = PricePredictor(
        input_size=saved_input_size,
        hidden_size=hidden_size,
        num_layers=num_lstm_layers,
        dropout=model_params["dropout"],
        fc_output_size=fc_output_size,
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    logger.info(f"Model: {model}")
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    
    if saved_input_size != input_size:
        logger.warning(f"Loaded model expects input size of {saved_input_size}, "
                       f"but current data has input size of {input_size}. "
                       f"This may cause issues during prediction.")
    
    return model

def predict(
    model: nn.Module,
    x_data: np.ndarray,
    scaler: StandardScaler,
    future_days: int,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model.

    Args:
        model (nn.Module): The trained model.
        x_data (np.ndarray): Input data.
        scaler (StandardScaler): The scaler used for data normalization.
        future_days (int): Number of days to predict into the future.
        features (List[str]): List of feature names.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predictions for the input data and future predictions.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
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
            x_tensor = torch.tensor(x_data[-1:], dtype=torch.float32).to(device)
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