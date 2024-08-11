import torch
from lstm_forecast.logger import setup_logger

logger = setup_logger("train_logger", "logs/train.log")


def run_training_epoch(
    model, data_loader, loss_fn, optimizer, device, clip_value=None
) -> float:
    """
    Run a single training epoch using the given data loader.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        loss_fn (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the training on.
        clip_value (float, optional): The value to clip gradients with. Defaults to None.

    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        total_loss += loss.item()

    logger.info(
        f"Training epoch completed. Average loss: {total_loss / len(data_loader):.4f}"
    )
    return total_loss / len(data_loader)


def run_validation_epoch(model, data_loader, criterion, device):
    """
    Run a single validation epoch using the given data loader.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (torch.utils.data.DataLoader): The data loader for validation data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run the validation on.

    Returns:
        float: The average loss for this epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    logger.info(
        f"Validation epoch completed. Average loss: {average_loss:.4f}"
    )
    return average_loss


def clip_gradients(model, max_norm) -> None:
    """
    Clip gradients of the model with the given maximum norm.

    Args:
        model (torch.nn.Module): The model to clip gradients for.
        max_norm (float): The maximum norm to clip gradients with.

    Returns:
        None
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    logger.info(f"Gradients clipped with max norm: {max_norm}")
