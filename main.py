import logging
import time
import torch
import numpy as np
from data_loader import get_data, preprocess_data, split_data
from model import PricePredictor
from train import train_model, evaluate_model

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = time.strftime('%Y-%m-%d')

    data = get_data(ticker, start=start_date, end=end_date)
    X, y, scaler = preprocess_data(data)
    train_loader, val_loader = split_data(X, y)

    model = PricePredictor().to(device)
    train_model(model, train_loader, val_loader, num_epochs=100)

    evaluate_model(ticker, model, X, y, scaler)

    logging.info('Model training and evaluation completed')
