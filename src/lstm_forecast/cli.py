# src/lstm_forecast/cli.py
import argparse
from lstm_forecast.data_loader import main as prepare_data
from lstm_forecast.hyperparameter_optimization import (
    main as hyperparameter_optimization,
)
from lstm_forecast.train import main as train_main
from lstm_forecast.predict import main as predict_main
from lstm_forecast.api.app import create_app
from lstm_forecast.config import Config


def prepare(args):
    config = Config(args.config)
    prepare_data(config)


def optimize(args):
    config = Config(args.config)
    hyperparameter_optimization(
        config,
        args.n_trials,
        args.n_feature_trials,
        args.min_features,
        args.force,
    )


def train(args):
    config = Config(args.config)
    train_main(config)


def predict(args):
    config = Config(args.config)
    predict_main(config)


def server(args):
    config = Config(args.config)
    app = create_app(config)
    app.run(debug=True)


def main():
    parser = argparse.ArgumentParser(prog="lstm_forecast")
    subparsers = parser.add_subparsers(dest="command")

    # parent parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )

    # prepare command
    parser_prepare = subparsers.add_parser(
        "prepare", parents=[parent_parser], help="Prepare data"
    )
    parser_prepare.set_defaults(func=prepare)

    # optimize command
    parser_optimize = subparsers.add_parser(
        "optimize",
        parents=[parent_parser],
        help="Optimize feature selection and hyperparameters",
    )
    parser_optimize.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning",
    )
    parser_optimize.add_argument(
        "--n_feature_trials",
        type=int,
        default=15,
        help="Number of trials for feature selection",
    )
    parser_optimize.add_argument(
        "--min_features",
        type=int,
        default=5,
        help="Minimum number of features to select",
    )
    parser_optimize.add_argument(
        "--force", action="store_true", help="Force re-run of Optuna study"
    )
    parser_optimize.set_defaults(func=optimize)

    # train command
    parser_train = subparsers.add_parser(
        "train", parents=[parent_parser], help="Train the model"
    )
    parser_train.set_defaults(func=train)

    # predict command
    parser_predict = subparsers.add_parser(
        "predict", parents=[parent_parser], help="Make predictions"
    )
    parser_predict.set_defaults(func=predict)

    # server command
    parser_server = subparsers.add_parser(
        "server", parents=[parent_parser], help="Start the API server"
    )
    parser_server.set_defaults(func=server)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
