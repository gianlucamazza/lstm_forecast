import pytest
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lstm_forecast.model import PricePredictor, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_price_predictor_initialization():
    input_size = 5
    hidden_size = 50
    num_layers = 2
    dropout = 0.5
    fc_output_size = 1

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size
    )

    assert isinstance(model, PricePredictor)
    assert model.hidden_size == hidden_size
    assert model.num_layers == num_layers
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == fc_output_size

def test_forward_pass():
    input_size = 5
    hidden_size = 50
    num_layers = 2
    dropout = 0.5
    fc_output_size = 1
    batch_size = 10
    seq_len = 20

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size
    ).to(device)

    x = torch.randn(batch_size, seq_len, input_size).to(device)
    output = model(x)

    assert output.shape == (batch_size, fc_output_size)

def test_training_epoch():
    input_size = 5
    hidden_size = 50
    num_layers = 2
    dropout = 0.5
    fc_output_size = 1
    batch_size = 10
    seq_len = 20

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size
    ).to(device)

    # Creare un semplice dataset
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size, fc_output_size)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = model.run_training_epoch(data_loader, criterion, optimizer)

    assert isinstance(loss, float)
    assert loss > 0

def test_validation_epoch():
    input_size = 5
    hidden_size = 50
    num_layers = 2
    dropout = 0.5
    fc_output_size = 1
    batch_size = 10
    seq_len = 20

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size
    ).to(device)

    # Creare un semplice dataset
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size, fc_output_size)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()

    loss = model.run_validation_epoch(data_loader, criterion)

    assert isinstance(loss, float)
    assert loss > 0

def test_model_loading():
    # Per questo test, devi avere un modello salvato in un file `.pth`
    symbol = "TEST_SYMBOL"
    path = "models"
    model_params = {"dropout": 0.5}
    input_size = 5

    # Supponiamo di avere un modello già salvato nel percorso specificato
    model_path = f"{path}/{symbol}_best_model.pth"

    # Creare un modello falso per il test
    model = PricePredictor(input_size=input_size, hidden_size=50, num_layers=2, dropout=0.5, fc_output_size=1)
    torch.save(model.state_dict(), model_path)

    loaded_model = load_model(symbol, path, model_params, input_size)

    assert isinstance(loaded_model, PricePredictor)
    assert loaded_model.hidden_size == model.hidden_size
    assert loaded_model.num_layers == model.num_layers

    # Pulizia
    os.remove(model_path)
    