import torch
import torch.nn as nn


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

    def __init__(self, input_size: int) -> None:
        """
        Initializes the PricePredictor model.

        Parameters:
        ----------
        input_size : int
            The number of expected features in the input sequence.
        """
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

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
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

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
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
