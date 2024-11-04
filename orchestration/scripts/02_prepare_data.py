from typing import List, Tuple

import mlflow
import pandas as pd

from .utils import main_args_parser


def read_and_clean_data(file_path: str) -> pd.DataFrame:
    """Reads the data from the given file path and cleans it by filtering out invalid entries."""
    df = pd.read_parquet(file_path)
    df = df[df.passenger_count > 0]
    return pd.DataFrame(df)


def clean_data(
    train_data_path, validation_data_path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans the data by filtering out invalid entries and saving the cleaned data."""
    print("cleaning data...")

    df_train = read_and_clean_data(train_data_path)
    df_val = read_and_clean_data(validation_data_path)

    print("num_train_samples:", df_train.shape[0])
    print("num_val_samples:", df_val.shape[0])

    return df_train, df_val


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the features for the given data."""

    data["duration"] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data["hour_of_day"] = data.tpep_pickup_datetime.dt.hour
    data["day_of_week"] = data.tpep_pickup_datetime.dt.dayofweek

    data["duration"] = data.duration.dt.total_seconds() / 60
    return pd.DataFrame(data[(data.duration >= 1) & (data.duration <= 60)])


def select_features(
    data: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]
):
    """Selects the features from the given data."""
    data[categorical_features] = data[categorical_features].astype(str)
    data[numerical_features] = data[numerical_features].astype(float)
    data = pd.DataFrame(data[categorical_features + numerical_features])
    return data


def feature_engineering(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs feature engineering on the cleaned training and validation data."""
    print("feature engineering...")

    df_train = compute_features(df_train)
    df_train = select_features(df_train, categorical_features, numerical_features)
    df_val = compute_features(df_val)
    df_val = select_features(df_val, categorical_features, numerical_features)

    return df_train, df_val


def parse_args():
    """Parses the command-line arguments for the feature_engineering script."""

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
        "--features_train_path", type=str, help="Path to the features training data"
    )
    parser.add_argument(
        "--features_validation_path",
        type=str,
        help="Path to the features validation data",
    )
    return parser.parse_args()


def main():

    with mlflow.start_run():
        args = parse_args()
        print("prepare data...")
        print(" ".join(f"{k}={v}\n" for k, v in vars(args).items()))

        mlflow.log_params(vars(args))

        df_train, df_val = clean_data(args.train_data_path, args.validation_data_path)
        mlflow.log_param("num_train_samples", df_train.shape[0])
        mlflow.log_param("num_val_samples", df_val.shape[0])

        df_train, df_val = feature_engineering(
            df_train, df_val, args.categorical_features, args.numerical_features
        )

        df_train.to_csv(args.features_train_path, index=False)
        df_val.to_csv(args.features_validation_path, index=False)


if __name__ == "__main__":
    main()
