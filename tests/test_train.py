import pytest
import torch
import os
import shutil
from unittest.mock import MagicMock, patch
from lstm_forecast.config import Config
from lstm_forecast.model import PricePredictor
from lstm_forecast.train import (
    initialize_model,
    train_model,
    main as train_main,
)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.model_settings = {
        "hidden_size": 50,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "clip_value": 1.0,
        "sequence_length": 60,
    }
    config.data_settings = {
        "targets": ["Close", "Volume"],
        "symbol": "TEST",
        "selected_features": ["Open", "High", "Low", "Close", "Volume"],
    }
    config.training_settings = {"epochs": 5, "model_dir": "test_models"}
    return config


@pytest.fixture
def mock_data_loader():
    loader = MagicMock(spec=torch.utils.data.DataLoader)
    loader.__iter__.return_value = [
        (torch.randn(10, 60, 5), torch.randn(10, 2)) for _ in range(5)
    ]
    loader.__len__.return_value = 5
    return loader


@pytest.fixture
def mock_model():
    model = MagicMock(spec=PricePredictor)
    # Add real parameters to the model
    model.parameters.return_value = [
        torch.nn.Parameter(torch.randn(1)) for _ in range(5)
    ]
    return model


def test_initialize_model(mock_config):
    num_features = 5
    model = initialize_model(mock_config, num_features)

    assert isinstance(model, PricePredictor)
    assert model.lstm.input_size == num_features
    assert model.lstm.hidden_size == mock_config.model_settings["hidden_size"]
    assert model.lstm.num_layers == mock_config.model_settings["num_layers"]
    assert model.fc.out_features == len(mock_config.data_settings["targets"])


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    # Setup: create the directory
    os.makedirs("test_models", exist_ok=True)

    yield  # This is where the testing happens

    # Teardown: remove the directory and all its contents
    shutil.rmtree("test_models", ignore_errors=True)


@patch("lstm_forecast.train.EarlyStopping")
@patch("lstm_forecast.train.run_training_epoch")
@patch("lstm_forecast.train.run_validation_epoch")
@patch("lstm_forecast.train.save_model_checkpoint")
def test_train_model(
    mock_save_checkpoint,
    mock_run_val,
    mock_run_train,
    mock_early_stopping,
    mock_config,
    mock_model,
    mock_data_loader,
):
    mock_run_train.return_value = 0.5
    mock_run_val.return_value = 0.4

    # Configure EarlyStopping mock to not stop training
    mock_early_stopping_instance = mock_early_stopping.return_value
    mock_early_stopping_instance.early_stop = False

    train_model(
        mock_config,
        mock_model,
        mock_data_loader,
        mock_data_loader,
        num_epochs=5,
        learning_rate=0.001,
        model_dir="test_models",
        weight_decay=0.0001,
        _device=torch.device("cpu"),
        fold_idx=1,
    )
    assert mock_run_train.call_count == 5
    assert mock_run_val.call_count == 5
    assert mock_save_checkpoint.call_count == 5


@patch("lstm_forecast.train.prepare_data")
@patch("lstm_forecast.train.initialize_model")
@patch("lstm_forecast.train.train_model")
@patch("lstm_forecast.train.evaluate_model")
@patch("lstm_forecast.train.export_to_onnx")
@patch("torch.save")
def test_main(
    mock_torch_save,
    mock_export_onnx,
    mock_evaluate,
    mock_train,
    mock_initialize,
    mock_prepare_data,
    mock_config,
):
    mock_prepare_data.return_value = (
        [(MagicMock(), MagicMock())],
        None,
        None,
        None,
        None,
        None,
        5,
    )

    def initialize_side_effect(*args, **kwargs):
        print(f"initialize_model called with args: {args}, kwargs: {kwargs}")
        return MagicMock(spec=PricePredictor)

    mock_initialize.side_effect = initialize_side_effect

    mock_evaluate.return_value = 0.3
    train_main(mock_config)
    print("\nDebug Information:")
    print(f"prepare_data call count: {mock_prepare_data.call_count}")
    print(f"initialize_model call count: {mock_initialize.call_count}")
    print(f"train_model call count: {mock_train.call_count}")
    print(f"evaluate_model call count: {mock_evaluate.call_count}")
    print(f"export_to_onnx call count: {mock_export_onnx.call_count}")
    print(f"torch.save call count: {mock_torch_save.call_count}")
    print("\ninitialize_model call args:")
    for i, call in enumerate(mock_initialize.call_args_list):
        print(f"Call {i + 1}: {call}")
    assert mock_prepare_data.call_count == 1
    assert (
        mock_initialize.call_count == 1
    ), f"initialize_model was called {mock_initialize.call_count} times"
    assert mock_train.call_count == 1
    assert mock_evaluate.call_count == 1
    assert mock_export_onnx.call_count == 1
    assert mock_torch_save.call_count == 1


if __name__ == "__main__":
    pytest.main()
