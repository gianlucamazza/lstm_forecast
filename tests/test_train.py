import pytest
import torch
from unittest.mock import MagicMock, patch
from lstm_forecast.config import Config
from lstm_forecast.model import PricePredictor
from lstm_forecast.train import main as train_main


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
    config.training_settings = {
        "epochs": 5,
        "model_dir": "test_models",
        "plot_dir": "test_plots",
    }
    return config


@patch("lstm_forecast.train.prepare_data")
@patch("lstm_forecast.train.initialize_model")
@patch("lstm_forecast.train.train_model")
@patch("lstm_forecast.train.evaluate_model")
@patch("lstm_forecast.train.export_to_onnx")
@patch("torch.save")
@patch("torch.optim.Adam")
def test_main(
    mock_adam,
    mock_torch_save,
    mock_export_onnx,
    mock_evaluate,
    mock_train,
    mock_initialize,
    mock_prepare_data,
    mock_config,
):
    mock_prepare_data.return_value = (
        [(MagicMock(), MagicMock()) for _ in range(5)],  # 5 folds
        ["feature1", "feature2", "feature3", "feature4", "feature5"],
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        5,
    )

    def initialize_side_effect(*args, **kwargs):
        print(f"initialize_model called with args: {args}, kwargs: {kwargs}")
        model = MagicMock(spec=PricePredictor)
        model.parameters.return_value = [
            torch.nn.Parameter(torch.randn(1)) for _ in range(5)
        ]
        model.to.return_value = model
        return model

    mock_initialize.side_effect = initialize_side_effect

    def mock_train_side_effect(config, model, *args, **kwargs):
        print(f"train_model called with model: {model}")
        assert hasattr(model, "parameters"), "Model should have parameters"
        return model

    mock_train.side_effect = mock_train_side_effect
    mock_evaluate.return_value = 0.3

    result = train_main(mock_config)
    print(f"\ntrain_main returned: {result}")

    print("\nDetailed Debug Information:")
    print(f"prepare_data call count: {mock_prepare_data.call_count}")
    print(f"initialize_model call count: {mock_initialize.call_count}")
    print(f"train_model call count: {mock_train.call_count}")
    print(f"evaluate_model call count: {mock_evaluate.call_count}")
    print(f"export_to_onnx call count: {mock_export_onnx.call_count}")
    print(f"torch.save call count: {mock_torch_save.call_count}")

    for i, call in enumerate(mock_initialize.call_args_list):
        print(f"initialize_model Call {i + 1}: {call}")
    for i, call in enumerate(mock_train.call_args_list):
        print(f"train_model Call {i + 1}: {call}")

    assert mock_prepare_data.call_count == 1
    assert (
        mock_initialize.call_count == 1
    ), f"initialize_model was called {mock_initialize.call_count} times, expected 1"
    assert (
        mock_train.call_count == 5
    ), f"train_model was called {mock_train.call_count} times, expected 5"
    assert mock_evaluate.call_count == 5
    assert mock_export_onnx.call_count == 1
    assert mock_torch_save.call_count == 1

    # Verify that the optimizer was created correctly
    assert mock_adam.call_count > 0
    adam_args, adam_kwargs = mock_adam.call_args
    assert len(adam_args) > 0, "Adam optimizer should receive parameters"
