import torch
import torch.nn as nn
from logger import setup_logger

logger = setup_logger('model_logger', 'logs/model.log')


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
