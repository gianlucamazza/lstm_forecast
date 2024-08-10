import pytest
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_forecast.model import PricePredictor, load_model

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def model_params():
    return {
        "input_size": 5,
        "hidden_size": 50,
        "num_layers": 2,
        "dropout": 0.5,
        "fc_output_size": 1
    }

@pytest.fixture(scope="module")
def model(model_params, device):
    return PricePredictor(**model_params).to(device)

@pytest.fixture(scope="module")
def data_loader(model_params, device):
    batch_size = 10
    seq_len = 20
    x = torch.randn(batch_size, seq_len, model_params["input_size"]).to(device)
    y = torch.randn(batch_size, model_params["fc_output_size"]).to(device)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_price_predictor_initialization(model, model_params):
    assert isinstance(model, PricePredictor)
    assert model.hidden_size == model_params["hidden_size"]
    assert model.num_layers == model_params["num_layers"]
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == model_params["fc_output_size"]

def test_forward_pass(model, data_loader, model_params):
    x, _ = next(iter(data_loader))
    output = model(x)
    assert output.shape == (x.shape[0], model_params["fc_output_size"])
    assert not torch.isnan(output).any(), "Output contains NaN values"

def test_training_epoch(model, data_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    initial_state = {name: param.clone() for name, param in model.named_parameters()}
    
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.mean().item():.6f}")
    
    losses = []
    for epoch in range(5):
        loss = model.run_training_epoch(data_loader, criterion, optimizer)
        losses.append(loss)
        print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
    
    print("\nFinal model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.mean().item():.6f}")
    
    print("\nParameter changes:")
    any_change = False
    for name, initial_param in initial_state.items():
        current_param = model.state_dict()[name]
        change = (current_param - initial_param).abs().mean().item()
        print(f"{name} change: {change:.6f}")
        if change > 1e-6:
            any_change = True
    
    assert any_change, "Model weights did not update significantly"
    assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1)), "Loss did not decrease monotonically"


def test_validation_epoch(model, data_loader):
    criterion = nn.MSELoss()
    loss = model.run_validation_epoch(data_loader, criterion)
    assert isinstance(loss, float)
    assert loss > 0

@pytest.fixture(scope="module")
def temp_model_path(tmp_path_factory, model, model_params):
    temp_dir = tmp_path_factory.mktemp("models")
    model_path = temp_dir / "TEST_SYMBOL_best_model.pth"
    torch.save(model.state_dict(), model_path)
    yield str(temp_dir), "TEST_SYMBOL", model_params
    model_path.unlink()

def test_model_loading(temp_model_path, model_params):
    path, symbol, _ = temp_model_path
    loaded_model = load_model(symbol, path, model_params, model_params["input_size"])

    assert isinstance(loaded_model, PricePredictor)
    assert loaded_model.hidden_size == model_params["hidden_size"]
    assert loaded_model.num_layers == model_params["num_layers"]
    
    # Compare state dictionaries
    original_state_dict = torch.load(os.path.join(path, f"{symbol}_best_model.pth"), weights_only=True)
    for key in original_state_dict:
        assert torch.equal(loaded_model.state_dict()[key], original_state_dict[key])

if __name__ == "__main__":
    pytest.main()