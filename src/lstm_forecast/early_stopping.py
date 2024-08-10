import logging
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self, patience=10, delta=0.001, verbose=False, path="checkpoint.pt"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0.001
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
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

    def __call__(self, val_loss, model=None):
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

    def _save_best_model_state(self, model):
        """Saves the best model state if a model is provided."""
        if model is not None:
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, self.path)

    def reset(self):
        """Resets the early stopping parameters."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        if self.verbose:
            self.logger.info("Early stopping reset")
