import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.metrics import root_mean_squared_error

from .utils import feature_encoding, main_args_parser, plot_pred_distribution


def hyperparameter_tuning(
    df_train,
    df_val,
    categorical_features,
    numerical_features,
    target_feature,
    number_of_boosting_rounds=10,
    early_stopping_rounds=5,
):
    """
    Perform hyperparameter tuning for an XGBoost model using the Hyperopt library.

    Parameters:
    - df_train: DataFrame for training data.
    - df_val: DataFrame for validation data.
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - target_feature: Name of the target feature.
    - number_of_boosting_rounds: Number of boosting rounds (default is 10).
    - early_stopping_rounds: Number of rounds for early stopping (default is 5).
    """
    print("tuning hyperparameters ....")

    print("preprocessing data...")
    x_train, x_val = feature_encoding(
        df_train=df_train[categorical_features + numerical_features],
        df_val=df_val[categorical_features + numerical_features],
        save_vectorizer=False,
    )

    y_train = df_train[target_feature]
    y_val = df_val[target_feature]

    train_matrix = xgb.DMatrix(x_train, label=y_train)
    val_matrix = xgb.DMatrix(x_val, label=y_val)

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("process", "hyperparameter_tuning")

            xgb_model = xgb.train(
                params,
                train_matrix,
                num_boost_round=number_of_boosting_rounds,
                evals=[(val_matrix, "validation")],
                early_stopping_rounds=early_stopping_rounds,
            )
            y_pred = xgb_model.predict(val_matrix)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", float(rmse))
            return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "max_depth": scope.int(hp.quniform("max_depth", 1, 10, 1)),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    print("optimizing hyperparameters...")
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )

    best_params = space_eval(search_space, best_result)
    print(f"best hyperparameters: {best_params}")
    mlflow.log_param("best_hyperparameters", best_params)
    return best_params


def train_model(
    df_train,
    df_val,
    categorical_features,
    numerical_features,
    target_feature,
    best_parameters,
):
    print(f"training model with optimized hyperparameters {best_parameters}")

    x_train, x_val = feature_encoding(
        df_train=df_train[categorical_features + numerical_features],
        df_val=df_val[categorical_features + numerical_features],
        save_vectorizer=True,
    )

    y_train = df_train[target_feature]
    y_val = df_val[target_feature]

    train_matrix = xgb.DMatrix(x_train, label=y_train)
    val_matrix = xgb.DMatrix(x_val, label=y_val)

    xgb_model = xgb.train(
        best_parameters,
        train_matrix,
        num_boost_round=10,
        evals=[(val_matrix, "validation")],
        early_stopping_rounds=5,
    )

    y_pred_train = xgb_model.predict(train_matrix)
    y_pred_val = xgb_model.predict(val_matrix)

    rmse_train = root_mean_squared_error(y_val, y_pred_val)
    rmse_val = root_mean_squared_error(y_val, y_pred_val)

    print(f"RMSE on training data: {rmse_train:.2f}")
    print(f"RMSE on validation data: {rmse_val:.2f}")

    return rmse_train, rmse_val, y_pred_train, y_pred_val


def parse_args():
    default_categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]
    default_numerical_features = [
        "trip_distance",
        "passenger_count",
        "hour_of_day",
        "day_of_week",
        "duration",
    ]

    parser = main_args_parser(default_categorical_features, default_numerical_features)
    parser.add_argument(
        "--target_feature", type=str, help="The feature to be used as target"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(" ".join(f"{k}={v}\n" for k, v in vars(args).items()))

    mlflow.xgboost.autolog()
    mlflow.log_params(vars(args))

    df_train = pd.read_csv(args.train_data_path)
    df_val = pd.read_csv(args.validation_data_path)

    mlflow.log_param("train_samples", df_train.shape[0])
    mlflow.log_param("validation_samples", df_val.shape[0])

    categorical_features = args.categorical_features
    numerical_features = args.numerical_features
    target = args.target

    mlflow.log_param("features", categorical_features + numerical_features)
    mlflow.log_param("target", target)

    mlflow.log_param("estimator", "XGBRegressor")

    best_parameters = hyperparameter_tuning(
        df_train=df_train,
        df_val=df_val,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_feature=target,
    )

    mlflow.log_param("best_parameters", best_parameters)

    _, rmse_val, y_pred_train, y_pred_val = train_model(
        df_train=df_train,
        df_val=df_val,
        categorical_features=args.categorical_features,
        numerical_features=args.numeric_features,
        target_feature=args.target_feature,
        best_parameters=best_parameters,
    )

    mlflow.log_metric("rmse", float(rmse_val))

    fig_train = plot_pred_distribution(
        y=df_train[target],
        y_pred=y_pred_train,
        y_label="Actual (Train dataset)",
        y_pred_label="Predicted (Train dataset)",
        x_label="Duration (min)",
        title="Actual vs Predicted Duration Distribution",
    )
    mlflow.log_figure(fig_train, "train_dist.png")

    fig_val = plot_pred_distribution(
        y=df_val[target],
        y_pred=y_pred_val,
        y_label="Actual (Validation dataset)",
        y_pred_label="Predicted (Validation dataset)",
        x_label="Duration (min)",
        title="Actual vs Predicted Duration Distribution",
    )

    mlflow.log_figure(fig_val, "val_dist.png")


if __name__ == "__main__":

    main()
