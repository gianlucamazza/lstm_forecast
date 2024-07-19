import os
import sys
import argparse
import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import TrialState

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_preprocess_data
from src.feature_engineering import calculate_technical_indicators
from src.model import PricePredictor
from src.early_stopping import EarlyStopping
from src.config import load_config, update_config
from src.logger import setup_logger
from src.model_utils import run_training_epoch, run_validation_epoch
from src.train import initialize_model, train_model, evaluate_model

train_logger = setup_logger("train_logger", "logs/train.log")
optuna_logger = setup_logger("optuna_logger", "logs/optuna.log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(optuna_trial, config):
    hidden_size = optuna_trial.suggest_int("hidden_size", 32, 256)
    num_layers = optuna_trial.suggest_int("num_layers", 1, 5)
    dropout = optuna_trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = optuna_trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = optuna_trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    config.model_settings.update({
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    })

    optuna_logger.info(f"Starting Trial {optuna_trial.number}")
    optuna_logger.info(f"Trial {optuna_trial.number}: hidden_size={hidden_size}, num_layers={num_layers}, "
                       f"dropout={dropout}, learning_rate={learning_rate}, weight_decay={weight_decay}")

    try:
        train_val_loaders, selected_features, _, _, _, _ = load_and_preprocess_data(config)

        fold_val_losses = []
        for fold_idx, (train_loader, val_loader) in enumerate(train_val_loaders):
            model = PricePredictor(
                input_size=len(selected_features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                fc_output_size=len(config.targets)
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True,
                                           path=f"models/optuna/model_{optuna_trial.number}_fold_{fold_idx}.pt")

            model.train()
            val_loss = float('inf')
            for epoch in range(config.epochs):
                train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.epochs}, "
                                   f"Train Loss: {train_loss:.4f}")
                val_loss = run_validation_epoch(model, val_loader, criterion, device)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.epochs}, "
                                   f"Validation Loss: {val_loss:.4f}")

                if early_stopping(val_loss, model):
                    optuna_logger.info(f"Early stopping triggered for trial {optuna_trial.number} at epoch {epoch + 1}")
                    break

            fold_val_losses.append(val_loss)

        avg_val_loss = np.mean(fold_val_losses)
        optuna_logger.info(f"Trial {optuna_trial.number} completed with Average Validation Loss: {avg_val_loss:.4f}")

    except Exception as e:
        optuna_logger.error(f"Error during trial {optuna_trial.number}: {e}")
        raise e

    return avg_val_loss


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    arg_parser.add_argument("--rebuild-features", action="store_true", help="Rebuild features")
    return arg_parser.parse_args()


def rebuild_features(config, n_trials=50):
    """Rebuild features using Optuna for feature selection."""
    train_logger.info("Starting feature selection using Optuna")

    def feature_selection_objective(optuna_trial):
        optuna_logger.info(f"Starting Trial {optuna_trial.number}")
        features = []
        all_features = config.all_features

        # Create a boolean parameter for each feature to decide if it's included or not
        for feature in all_features:
            if optuna_trial.suggest_categorical(f"use_{feature}", [False, True]):
                features.append(feature)

        optuna_logger.info(f"Trial {optuna_trial.number}: Selected features: {features}")

        if len(features) == 0:
            optuna_logger.warning(f"Trial {optuna_trial.number}: No features selected, returning infinity loss")
            return float('inf')  # Penalize trials with no features selected

        # Load and preprocess data with the selected features
        train_val_loaders, features, _, _, _, _ = load_and_preprocess_data(config)
        optuna_logger.info(f"Trial {optuna_trial.number}: Data loaded and preprocessed")

        fold_val_losses = []
        for fold_idx, (train_loader, val_loader) in enumerate(train_val_loaders):
            model = PricePredictor(
                input_size=len(features),  # Update input_size based on selected features
                hidden_size=config.model_settings['hidden_size'],
                num_layers=config.model_settings['num_layers'],
                dropout=config.model_settings['dropout'],
                fc_output_size=len(config.targets)
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config.model_settings['learning_rate'],
                                   weight_decay=config.model_settings['weight_decay'])
            early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True,
                                           path=f"models/optuna/"
                                                f"feature_selection_{optuna_trial.number}_fold_{fold_idx}.pt")

            model.train()
            val_loss = float('inf')
            for epoch in range(config.epochs):
                train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = run_validation_epoch(model, val_loader, criterion, device)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.epochs}, "
                                   f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                if early_stopping(val_loss, model):
                    optuna_logger.info(
                        f"Trial {optuna_trial.number}, Fold {fold_idx}: Early stopping at epoch {epoch + 1}")
                    break

            fold_val_losses.append(val_loss)

        avg_val_loss = np.mean(fold_val_losses)
        optuna_logger.info(f"Trial {optuna_trial.number} completed with Average Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    # Run the Optuna optimization for feature selection
    study = optuna.create_study(direction="minimize")
    optuna_logger.info(f"Starting Optuna study with {n_trials} trials")
    study.optimize(feature_selection_objective, n_trials=n_trials)  # Adjust n_trials as needed

    # Get the best set of features
    best_trial = study.best_trial
    selected_features = [feature for feature in best_trial.params.keys() if best_trial.params[feature]]

    # Update the configuration with the best features
    config.selected_features = selected_features
    config.save()
    optuna_logger.info(f"Feature selection completed. Best features: {selected_features}")


def main():
    args = parse_arguments()

    config = load_config(args.config)
    optuna_logger.info(f"Loaded configuration from {args.config}")

    if args.rebuild_features:
        optuna_logger.info("Rebuilding features")
        historical_data = pd.read_csv(config.historical_data_path, index_col="Date", parse_dates=True)
        _, _ = calculate_technical_indicators(
            historical_data,
            windows=config.indicator_windows,
            asset_type=config.asset_type,
            frequency=config.data_resampling_frequency,
        )
        rebuild_features(config)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, config), n_trials=100)

    best_params = study.best_trial.params
    optuna_logger.info(f"Best hyperparameters: {best_params}")

    config.model_settings.update(best_params)
    update_config(config, "model_settings", config.model_settings)

    train_val_loaders, selected_features, scaler_prices, _, _, historical_data = load_and_preprocess_data(config)

    model = initialize_model(config)

    train_loader, val_loader = train_val_loaders[0]
    train_model(
        config.symbol,
        model,
        train_loader,
        val_loader,
        num_epochs=config.epochs,
        learning_rate=config.model_settings.get("learning_rate", 0.001),
        model_dir=config.model_dir,
        weight_decay=config.model_settings.get("weight_decay", 0.0),
        _device=device
    )

    evaluate_model(
        model,
        data_loader=train_loader,
        loss_fn=nn.MSELoss(),
        _device=device,
    )

    trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    trials_df.to_csv("reports/optuna_trials.csv", index=False)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    optuna_logger.info("Study statistics: ")
    optuna_logger.info(f"  Number of finished trials: {len(study.trials)}")
    optuna_logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    optuna_logger.info(f"  Number of complete trials: {len(complete_trials)}")

    optuna_logger.info("Best trial:")
    trial = study.best_trial

    optuna_logger.info(f"  Value: {trial.value}")
    optuna_logger.info(f"  Params: ")
    for key, value in trial.params.items():
        optuna_logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()
