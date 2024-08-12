import logging
import torch


class EarlyStopping:
    """
    Monitors the validation loss during training and triggers early stopping if the validation loss
    does not improve after a specified number of epochs (patience). Also saves the best model state.

    Attributes:
        patience (int): Number of epochs to wait for an improvement in validation loss before stopping.
        delta (float): Minimum change in the monitored validation loss to qualify as an improvement.
        verbose (bool): If True, logs information about validation loss improvements and early stopping.
        path (str): File path to save the model checkpoint with the best validation loss.
        counter (int): Counts the number of consecutive epochs without improvement in validation loss.
        best_loss (float or None): Best validation loss observed during training.
        early_stop (bool): Flag to indicate whether early stopping has been triggered.
        best_model_state (dict or None): State dictionary of the model corresponding to the best validation loss.
        logger (logging.Logger): Logger instance for logging messages related to early stopping.
    """

    def __init__(
        self, patience=10, delta=0.001, verbose=False, path="checkpoint.pt"
    ) -> None:
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): How many epochs to wait after the last improvement in validation loss
                            before stopping the training. Default is 10.
            delta (float): Minimum change in the monitored validation loss to qualify as an improvement.
                           A smaller value makes it harder to qualify as an improvement. Default is 0.001.
            verbose (bool): If True, prints/logs a message each time the validation loss improves or
                            early stopping is triggered. Default is False.
            path (str): Path to the file where the model checkpoint should be saved. Default is 'checkpoint.pt'.

        Returns:
            None
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None  # Store the best model state
        self.logger = logging.getLogger("early_stopping_logger")

    def __call__(self, val_loss: float, model: torch.nn.Module = None) -> bool:
        """
        Checks if the validation loss has improved and updates the early stopping status accordingly.

        Args:
            val_loss (float): The current validation loss to be compared with the best observed loss.
            model (torch.nn.Module, optional): The model to save if the validation loss improves. Default is None.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_best_model_state(model)
            return False
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_best_model_state(model)
            if self.verbose:
                self.logger.info(f"Validation loss improved to {val_loss:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement in validation loss. Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info("Early stopping triggered")
                return True
            return False

    def _save_best_model_state(self, model: torch.nn.Module) -> None:
        """
        Saves the current model's state dictionary if a model is provided.

        Args:
            model (torch.nn.Module): The model whose state is to be saved.

        Returns:
            None
        """
        if model is not None:
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, self.path)

    def reset(self) -> None:
        """
        Resets the early stopping attributes to their initial states. This is useful if
        you want to reuse the EarlyStopping instance for another training session.

        Args:
            None

        Returns:
            None
        """
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        if self.verbose:
            self.logger.info("Early stopping reset")
