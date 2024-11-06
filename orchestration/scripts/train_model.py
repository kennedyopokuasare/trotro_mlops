import os

import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from orchestration.scripts.prepare_data import compute_features
from orchestration.scripts.utils import main_args_parser, plot_pred_distribution


def hyperparameter_tuning(
    train_data,
    validate_data,
    categorical_features,
    numerical_features,
    target_feature,
    number_of_boosting_rounds=10,
    early_stopping_rounds=5,
):
    """
    Perform hyperparameter tuning for an XGBoost model using the Hyperopt library.

    Parameters:
    - train_data: DataFrame for training data.
    - validate_data: DataFrame for validation data.
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - target_feature: Name of the target feature.
    - number_of_boosting_rounds: Number of boosting rounds (default is 10).
    - early_stopping_rounds: Number of rounds for early stopping (default is 5).
    """
    print("tuning hyperparameters ....")

    print("preprocessing data...")

    df_train = train_data[categorical_features + numerical_features]
    df_val = validate_data[categorical_features + numerical_features]

    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(df_train.to_dict(orient="records"))
    x_val = dv.transform(df_val.to_dict(orient="records"))  # type: ignore

    y_train = train_data[target_feature]
    y_val = validate_data[target_feature]

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
    df_train_full,
    categorical_features,
    numerical_features,
    target_feature,
    best_parameters,
):
    """
    Train an XGBoost model using the provided training data and optimized hyperparameters.

    Parameters:
    - df_train_full: DataFrame containing the full training data.
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - target_feature: Name of the target feature.
    - best_parameters: Dictionary of best hyperparameters for the model.
    """
    print(f"training model with optimized hyperparameters {best_parameters}")

    x_train = df_train_full[categorical_features + numerical_features].to_dict(
        orient="records"
    )
    y_train = df_train_full[target_feature].values

    pipeline = make_pipeline(
        DictVectorizer(sparse=False), XGBRegressor(**best_parameters)
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_train)
    rmse = root_mean_squared_error(y_train, y_pred)
    print(f"RMSE on full training data: {rmse:.2f}")

    return rmse, y_pred


def score_model(run_id, data, features, target):
    """
    Score the model using the provided data and features.

    Parameters:
    - run_id: The ID of the MLflow run.
    - data: DataFrame containing the data for scoring.
    - features: List of feature names to be used for predictions.
    - target: Name of the target feature for evaluation.
    """

    print(f"Scoring model saved for run Id: {run_id}")
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)

    prepared_data = compute_features(data=data)

    prediction = model.predict(prepared_data[features])
    rmse = root_mean_squared_error(y_true=prepared_data[target], y_pred=prediction)
    results = prepared_data[f"predicted_{target}"] = prediction
    return rmse, pd.DataFrame(results)


def parse_args():
    default_categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]
    default_numerical_features = [
        "trip_distance",
        "passenger_count",
        "hour_of_day",
        "day_of_week",
    ]

    parser = main_args_parser(default_categorical_features, default_numerical_features)
    parser.add_argument(
        "--target_feature",
        type=str,
        help="The feature to be used as target",
        default="duration",
    )

    parser.add_argument(
        "--scoring_data", type=str, help="The data for scoring the model"
    )
    return parser.parse_args()


def main():
    with mlflow.start_run() as run:
        args = parse_args()
        print(" ".join(f"{k}={v}\n" for k, v in vars(args).items()))

        mlflow.xgboost.autolog()
        mlflow.log_params(vars(args))

        train_data_file = os.path.join(args.train_data_path, "train_data.csv")
        validate_data_file = os.path.join(args.train_data_path, "validate_data.csv")
        df_train = pd.read_csv(train_data_file)
        df_val = pd.read_csv(validate_data_file)

        mlflow.log_param("train_samples", df_train.shape[0])
        mlflow.log_param("validation_samples", df_val.shape[0])

        categorical_features = args.categorical_features
        numerical_features = args.numerical_features
        target = args.target_feature

        mlflow.log_param("features", categorical_features + numerical_features)
        mlflow.log_param("target", target)

        mlflow.log_param("estimator", "XGBRegressor")

        best_parameters = hyperparameter_tuning(
            train_data=df_train,
            validate_data=df_val,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target_feature=target,
        )

        mlflow.log_param("best_parameters", best_parameters)

        ## After hyperparameter tuning, train the full dataset
        df_train_full = pd.concat([df_train, df_val])

        _, y_pred = train_model(
            df_train_full=df_train_full,
            categorical_features=args.categorical_features,
            numerical_features=args.numerical_features,
            target_feature=args.target_feature,
            best_parameters=best_parameters,
        )

        fig_train = plot_pred_distribution(
            y=df_train[target],
            y_pred=y_pred,
            y_label="Actual (Full train dataset)",
            y_pred_label="Predicted (Full train dataset)",
            x_label="Duration (min)",
            title="Actual vs Predicted Duration Distribution",
        )
        mlflow.log_figure(fig_train, "train_dist.png")

        print(f"Completed train pipeline with mlflow run id {run.info.run_id}")

        df_scoring = pd.read_parquet(args.scoring_data)

        print(f"Scoring the model with {df_scoring.shape[0]} data points")

        rmse, predictions = score_model(
            run_id=run.info.run_id,
            data=df_scoring,
            features=categorical_features + numerical_features,
            target=target,
        )

        mlflow.log_metric("rmse", float(rmse))
        print("Prediction on unseen data", predictions.sample(50))
        return run.info.run_id


if __name__ == "__main__":

    runId = main()

    ## RunIf to be retrieved in GitHub Actions
    print(runId)
