import torch
from src.logger import setup_logger

logger = setup_logger("model_utils_logger", "logs/model_utils.log")


def run_training_epoch(model, data_loader, criterion, optimizer, device):
    """Run a single training epoch using the given data loader."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    logger.info(f"Training epoch completed. Average loss: {average_loss:.4f}")
    return average_loss


def run_validation_epoch(model, data_loader, criterion, device):
    """Run a single validation epoch using the given data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    logger.info(f"Validation epoch completed. Average loss: {average_loss:.4f}")
    return average_loss
