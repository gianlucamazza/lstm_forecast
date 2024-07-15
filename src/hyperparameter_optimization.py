import argparse
import optuna
from optuna.trial import TrialState
from src.data_loader import load_and_preprocess_data
from src.model import PricePredictor
from src.early_stopping import EarlyStopping
from src.config import load_config, update_config
from src.logger import setup_logger
from src.model_utils import run_training_epoch, run_validation_epoch
import torch
import torch.nn as nn
import torch.optim as optim
from src.train import initialize_model, train_model, evaluate_model

train_logger = setup_logger("train_logger", "logs/train.log")
optuna_logger = setup_logger("optuna_logger", "logs/optuna.log")
early_stopping_logger = setup_logger("early_stopping_logger", "logs/early_stopping.log")

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
        train_loader, val_loader, selected_features, _, _, _ = load_and_preprocess_data(config)

        model = PricePredictor(
            input_size=len(selected_features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            fc_output_size=len(config.targets)
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True)

        model.train()
        val_loss = float('inf')
        for epoch in range(config.epochs):
            train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
            train_logger.info(f"Trial {optuna_trial.number}, Epoch {epoch + 1}/{config.epochs}, "
                               f"Train Loss: {train_loss:.4f}")
            val_loss = run_validation_epoch(model, val_loader, criterion, device)
            train_logger.info(f"Trial {optuna_trial.number}, Epoch {epoch + 1}/{config.epochs}, "
                               f"Validation Loss: {val_loss:.4f}")

            if early_stopping(val_loss, model):
                early_stopping_logger.info(f"Early stopping triggered for trial {optuna_trial.number} at epoch {epoch + 1}")
                break

        optuna_logger.info(f"Trial {optuna_trial.number} completed with Validation Loss: {val_loss:.4f}")

    except Exception as e:
        optuna_logger.error(f"Error during trial {optuna_trial.number}: {e}")
        raise e

    return val_loss


def main():
    args = parse_arguments()
    config = load_config(args.config)
    optuna_logger.info(f"Loaded configuration from {args.config}")
    if args.rebuild_features:
        rebuild_features(config)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, config), n_trials=100)

    best_params = study.best_trial.params
    optuna_logger.info(f"Best hyperparameters: {best_params}")

    config.model_settings.update(best_params)
    update_config(config, "model_settings", config.model_settings)

    train_loader, val_loader, selected_features, scaler_prices, scaler_volume, historical_data = (
        load_and_preprocess_data(config))

    update_config_with_best_features(config, selected_features)

    model = initialize_model(config)

    train_model(
        config.symbol,
        model,
        train_loader,
        val_loader,
        num_epochs=config.epochs,
        learning_rate=config.model_settings.get("learning_rate", 0.001),
        model_dir=config.model_dir,
        weight_decay=config.model_settings.get("weight_decay", 0.0),
    )

    x, y = [], []
    for data, target in train_loader:
        x.append(data)
        y.append(target)
    x = torch.cat(x)
    y = torch.cat(y)

    evaluate_model(
        config.symbol,
        model,
        x,
        y,
        scaler_prices,
        scaler_volume,
        historical_data.index
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


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file")
    arg_parser.add_argument(
        "--rebuild-features", action="store_true", help="Rebuild features"
    )
    return arg_parser.parse_args()


def rebuild_features(config):
    update_config(config, "best_features", [])
    config.save()
    optuna_logger.info("Rebuilding features")


def update_config_with_best_features(config, selected_features):
    optuna_logger.info(f"Selected features: {selected_features}")
    update_config(config, "best_features", selected_features)
    config.save()


if __name__ == "__main__":
    main()
