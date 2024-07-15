import argparse
import optuna
from optuna.trial import TrialState
from src.data_loader import preprocess_data, get_data, split_data
from src.model import PricePredictor, EarlyStopping
from src.config import load_config, update_config
from src.logger import setup_logger
from src.model_utils import run_training_epoch, run_validation_epoch
import torch
import torch.nn as nn
import torch.optim as optim
from src.train import initialize_model, train_model, evaluate_model

logger = setup_logger("train_logger", "logs/train.log")
optuna_logger = setup_logger("optuna_logger", "logs/optuna.log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(config):
    historical_data, features = get_data(
        config.ticker,
        config.symbol,
        config.asset_type,
        config.start_date,
        config.end_date,
        config.indicator_windows,
        config.data_sampling_interval,
        config.data_resampling_frequency,
    )

    x, y, scaler_features, scaler_prices, scaler_volume, selected_features = preprocess_data(
        config.symbol,
        config.data_sampling_interval,
        historical_data,
        config.targets,
        config.look_back,
        config.look_forward,
        features,
        config.disabled_features,
        config.best_features,
    )

    train_loader, val_loader = split_data(x, y, batch_size=config.batch_size)
    return train_loader, val_loader, selected_features, scaler_prices, scaler_volume, historical_data


def objective(trial, config):
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    config.model_settings.update({
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    })

    optuna_logger.info(f"Trial {trial.number}: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, "
                       f"learning_rate={learning_rate}, weight_decay={weight_decay}")

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
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        model.train()
        val_loss = float('inf')
        for epoch in range(config.epochs):
            train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
            optuna_logger.info(f"Trial {trial.number}, Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}")
            val_loss = run_validation_epoch(model, val_loader, criterion, device)
            optuna_logger.info(f"Trial {trial.number}, Epoch {epoch + 1}/{config.epochs}, Validation Loss: {val_loss:.4f}")

            if early_stopping(val_loss):
                optuna_logger.info(f"Early stopping triggered for trial {trial.number}")
                break
    except Exception as e:
        optuna_logger.error(f"Error during trial {trial.number}: {e}")
        raise e

    return val_loss


def main():
    args = parse_arguments()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    if args.rebuild_features:
        rebuild_features(config)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, config), n_trials=100)

    best_params = study.best_trial.params
    logger.info(f"Best hyperparameters: {best_params}")

    config.model_settings.update(best_params)
    update_config(config, "model_settings", config.model_settings)

    train_loader, val_loader, selected_features, scaler_prices, scaler_volume, historical_data = load_and_preprocess_data(config)

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

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


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
    logger.info("Rebuilding features")


def get_historical_data(config):
    logger.info(
        f"Getting historical data for {config.ticker} from {config.start_date} to {config.end_date}"
    )
    return get_data(
        config.ticker,
        config.symbol,
        config.asset_type,
        config.start_date,
        config.end_date,
        config.indicator_windows,
        config.data_sampling_interval,
        config.data_resampling_frequency,
    )


def update_config_with_best_features(config, selected_features):
    logger.info(f"Selected features: {selected_features}")
    update_config(config, "best_features", selected_features)
    config.save()


if __name__ == "__main__":
    main()
