import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger
from typing import List, Tuple

logger = setup_logger('model_logger', 'logs/model.log')
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PricePredictor(nn.Module):
    """
    A PyTorch neural network module for predicting prices using an LSTM model.

    Attributes:
    ----------
    lstm : nn.LSTM
        The LSTM layer(s) for processing the input sequence.
    dropout : nn.Dropout
        Dropout layer for regularization.
    fc : nn.Linear
        Fully connected layer for the final output.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, fc_output_size: int) -> None:
        """
        Initializes the PricePredictor model.

        Parameters:
        ----------
        input_size : int
            The number of expected features in the input sequence.
        hidden_size : int
            The number of features in the hidden state h.
        num_layers : int
            Number of recurrent layers.
        dropout : float
            Dropout rate for regularization.
        fc_output_size : int
            The size of the output of the fully connected layer.
        """
        super(PricePredictor, self).__init__()
        logger.info(
            f"Initializing PricePredictor with input size: {input_size}, hidden size: {hidden_size}, num layers: {num_layers}, dropout: {dropout}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, fc_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1), representing the predicted price.
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        logger.debug("Forward pass completed.")
        return out


def load_model(symbol: str, path: str, input_shape: int, model_params: dict) -> nn.Module:
    """
    Load the trained model from a given path.

    Args:
        symbol (str): The stock symbol.
        path (str): The path to the trained model.
        input_shape (int): The input shape of the model.
        model_params (dict): The model parameters.

    Returns:
        nn.Module: The trained model.
    """
    model = PricePredictor(
        input_size=input_shape,
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout'],
        fc_output_size=model_params['fc_output_size']
    ).to(device)
    model.device = device
    logger.info(f"Model: {model}")
    logger.info(f"Loading model from {path}/{symbol}_model.pth")
    model.load_state_dict(torch.load(path + f'/{symbol}_model.pth', map_location=device))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model


def predict(model: nn.Module, x_data: np.ndarray, scaler: StandardScaler, future_days: int, features: List) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model.

    Args:
        model (nn.Module): The trained model.
        x_data (np.ndarray): The input data.
        scaler (StandardScaler): The scaler used for normalization.
        future_days (int): Number of days to predict into the future.
        features (List): List of feature names.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predictions and future predictions.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(model.device)
        predictions = model(x_tensor).cpu().numpy()

        predictions_reshaped = np.zeros((x_data.shape[0], len(features) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions_reshaped = np.pad(predictions_reshaped,
                                      ((0, 0), (0, len(scaler.scale_) - len(predictions_reshaped[0]))), 'constant')
        predictions = scaler.inverse_transform(predictions_reshaped)[:, 0]

        future_predictions = []
        for _ in range(future_days):
            x_tensor = torch.tensor(x_data[-1:], dtype=torch.float32).to(model.device)
            future_pred = model(x_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, x_data.shape[2]))
            new_row[0, 0] = future_pred
            new_row[0, 1:] = x_data[-1, -1, 1:]

            x_data = np.append(x_data, [np.append(x_data[-1][1:], new_row, axis=0)], axis=0)

        future_predictions_reshaped = np.zeros((future_days, len(features) + 1))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions_reshaped = np.pad(future_predictions_reshaped,
                                             ((0, 0), (0, len(scaler.scale_) - len(future_predictions_reshaped[0]))),
                                             'constant')
        future_predictions = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions


class EarlyStopping:
    """
    A utility class to perform early stopping during training to prevent overfitting.

    Attributes:
    ----------
    patience : int
        Number of epochs to wait after the last improvement before stopping the training.
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    counter : int
        Counter to keep track of the number of epochs with no improvement.
    best_loss : float or None
        Best recorded loss value for early stopping.
    early_stop : bool
        Boolean flag to indicate whether to stop the training.
    """

    def __init__(self, patience=10, delta=0):
        """
        Initializes the EarlyStopping object.

        Parameters:
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 10).
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement (default is 0).
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        logger.info(f"EarlyStopping initialized with patience={patience}, delta={delta}")

    def __call__(self, val_loss):
        """
        Checks whether the training should be stopped based on the validation loss.

        Parameters:
        ----------
        val_loss : float
            The current epoch's validation loss.

        Returns:
        -------
        None
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            logger.info(f"Initial validation loss set to {val_loss:.6f}")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            logger.info(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            logger.info(f"Improvement in validation loss. Best loss updated to {val_loss:.6f}")
