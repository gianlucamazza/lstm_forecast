import os
import sys
import argparse
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.trial import TrialState

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_preprocess_data
from src.model import PricePredictor
from src.train import train_model, evaluate_model
from src.early_stopping import EarlyStopping
from src.config import load_config, update_config
from src.logger import setup_logger
from src.model_utils import run_training_epoch, run_validation_epoch
from src.feature_selection import time_series_feature_selection, correlation_analysis

# Setup loggers
train_logger = setup_logger("train_logger", "logs/train.log")
optuna_logger = setup_logger("optuna_logger", "logs/optuna.log")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(optuna_trial, config, selected_features):
    # Suggest hyperparameters
    hidden_size = optuna_trial.suggest_int("hidden_size", 32, 512)
    num_layers = optuna_trial.suggest_int("num_layers", 1, 5)
    dropout = optuna_trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = optuna_trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = optuna_trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    batch_size = optuna_trial.suggest_int("batch_size", 16, 256)
    sequence_length = optuna_trial.suggest_int("sequence_length", 10, 100)
    clip_value = optuna_trial.suggest_float("clip_value", 0.1, 5.0)

    # Update config with suggested hyperparameters
    config.model_settings.update({
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "clip_value": clip_value
    })

    optuna_logger.info(f"Starting Trial {optuna_trial.number} with params: {config.model_settings}")

    try:
        # Load and preprocess data
        train_val_loaders, _, _, _, _, _ = load_and_preprocess_data(config, selected_features)

        if not train_val_loaders:
            optuna_logger.warning(f"No data loaded for trial {optuna_trial.number}")
            return float('inf')
        
        feature_penalty = 0.01 * len(selected_features)

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
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            early_stopping = EarlyStopping(
                patience=10, delta=0.001, verbose=True,
                path=f"models/optuna/model_{optuna_trial.number}_fold_{fold_idx}.pt"
            )

            model.train()
            for epoch in range(config.epochs):
                train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device, clip_value=clip_value)
                val_loss = run_validation_epoch(model, val_loader, criterion, device)
                scheduler.step(val_loss)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.epochs}, "
                                   f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                if early_stopping(val_loss, model):
                    optuna_logger.info(f"Early stopping triggered for trial {optuna_trial.number} at epoch {epoch + 1}")
                    break

            fold_val_losses.append(val_loss)

        avg_val_loss = np.mean(fold_val_losses)
        optuna_logger.info(f"Trial {optuna_trial.number} completed with Average Validation Loss: {avg_val_loss:.4f}")

    except Exception as e:
        optuna_logger.error(f"Error during trial {optuna_trial.number}: {e}")
        return float('inf')

    return avg_val_loss + feature_penalty

def filter_available_features(config, data, selected_features):
    available_features = config.data_settings["all_features"]
    filtered_features = [feature for feature in selected_features if feature in available_features]
    missing_features = [feature for feature in selected_features if feature not in available_features]
    if missing_features:
        optuna_logger.warning(f"Missing features: {missing_features}")
    return filtered_features

def feature_selection_objective(optuna_trial, config, data: pd.DataFrame, min_features=5):
    all_features = config.data_settings["all_features"]
    available_features = data.columns
    selected_features = []

    for feature in all_features:
        if feature in available_features:
            use_feature = optuna_trial.suggest_categorical(f"use_{feature}", [False, True])
            if use_feature:
                selected_features.append(feature)

    num_selected_features = len(selected_features)
    optuna_logger.info(f"Trial {optuna_trial.number}: Selected features: {selected_features} (Total: {num_selected_features})")
    
    if num_selected_features < min_features:
        optuna_logger.warning(f"Trial {optuna_trial.number}: Selected features are less than the minimum required ({min_features}), returning infinity loss")
        return float('inf')

    filtered_data = data[selected_features + config.data_settings["targets"]]
    train_val_loaders, _, _, _, _, _ = load_and_preprocess_data(config, filtered_data)
    optuna_logger.info(f"Trial {optuna_trial.number}: Data loaded and preprocessed")

    fold_val_losses = []
    for fold_idx, (train_loader, val_loader) in enumerate(train_val_loaders):
        model = PricePredictor(
            input_size=num_selected_features,
            hidden_size=config.model_settings['hidden_size'],
            num_layers=config.model_settings['num_layers'],
            dropout=config.model_settings['dropout'],
            fc_output_size=len(config.data_settings["targets"])
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.model_settings['learning_rate'],
                               weight_decay=config.model_settings['weight_decay'])
        early_stopping = EarlyStopping(
            patience=10, delta=0.001, verbose=True,
            path=f"models/optuna/feature_selection_{optuna_trial.number}_fold_{fold_idx}.pt"
        )

        model.train()
        for epoch in range(config.training_settings["epochs"]):
            train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device, clip_value=config.model_settings['clip_value'])
            val_loss = run_validation_epoch(model, val_loader, criterion, device)
            optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.training_settings['epochs']}, "
                               f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if early_stopping(val_loss, model):
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}: Early stopping at epoch {epoch + 1}")
                break

        fold_val_losses.append(val_loss)

    avg_val_loss = np.mean(fold_val_losses)
    optuna_logger.info(f"Trial {optuna_trial.number} completed with Average Validation Loss: {avg_val_loss:.4f}")

    feature_penalty = 0.01 * num_selected_features
    return avg_val_loss + feature_penalty

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    arg_parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter tuning")
    arg_parser.add_argument("--n_feature_trials", type=int, default=50, help="Number of trials for feature selection")
    arg_parser.add_argument("--min_features", type=int, default=5, help="Minimum number of features to select")
    arg_parser.add_argument("--force", action="store_true", help="Force re-run of Optuna study, both feature selection and hyperparameter tuning")
    return arg_parser.parse_args()

def main():
    try:
        args = parse_arguments()
        config = load_config(args.config)
        optuna_logger.info(f"Loaded configuration from {args.config}")

        train_val_loaders, _, _, _, train_data, _ = load_and_preprocess_data(config)
        data = train_data
        
        if args.force:
            optuna_logger.info("Forcing re-run of Optuna study")
            if os.path.exists("data/optuna_feature_selection.db"):
                os.remove("data/optuna_feature_selection.db")
            if os.path.exists("data/optuna_hyperparameter_tuning.db"):
                os.remove("data/optuna_hyperparameter_tuning.db")
            
        optuna_logger.info("Starting feature selection")
        feature_study = optuna.create_study(
            direction="minimize",
            study_name="feature_selection_study",
            storage="sqlite:///data/optuna_feature_selection.db",
            load_if_exists=not args.force
        )

        feature_study.optimize(lambda t: feature_selection_objective(t, config, data, min_features=args.min_features), n_trials=args.n_feature_trials)

        best_feature_trial = feature_study.best_trial
        selected_features = [feature for feature in config.data_settings["all_features"] if best_feature_trial.params.get(f"use_{feature}", False)]
        
        selected_features = filter_available_features(data, selected_features)
        
        if not selected_features:
            optuna_logger.error("No valid features selected. Aborting.")
            return
        
        selected_features = correlation_analysis(data[selected_features])
        selected_features = time_series_feature_selection(data[selected_features], data[config.data_settings["targets"]], num_features=len(selected_features))

        config.selected_features = selected_features
        update_config(config, "selected_features", selected_features)
        optuna_logger.info(f"Selected features: {selected_features}")

        optuna_logger.info("Starting hyperparameter tuning")
        study = optuna.create_study(
            direction="minimize",
            study_name="hyperparameter_tuning_study",
            storage="sqlite:///data/optuna_hyperparameter_tuning.db",
            load_if_exists=not args.force
        )
        study.optimize(lambda t: objective(t, config, selected_features), n_trials=args.n_trials)

        best_params = study.best_trial.params
        optuna_logger.info(f"Best hyperparameters: {best_params}")

        config.model_settings.update(best_params)
        update_config(config, "model_settings", config.model_settings)

        model = PricePredictor(
            input_size=len(selected_features),
            hidden_size=config.model_settings['hidden_size'],
            num_layers=config.model_settings['num_layers'],
            dropout=config.model_settings['dropout'],
            fc_output_size=len(config.targets)
        ).to(device)

        train_loader, val_loader = train_val_loaders[0]

        train_model(
            config,
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
            _device=device
        )

        trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        trials_df.to_csv("reports/optuna_trials.csv", index=False)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        optuna_logger.info("Study statistics:")
        optuna_logger.info(f"  Number of finished trials: {len(study.trials)}")
        optuna_logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
        optuna_logger.info(f"  Number of complete trials: {len(complete_trials)}")

        optuna_logger.info("Best trial:")
        trial = study.best_trial

        optuna_logger.info(f"  Value: {trial.value}")
        optuna_logger.info(f"  Params:")
        for key, value in trial.params.items():
            optuna_logger.info(f"    {key}: {value}")
    except Exception as e:
        optuna_logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
