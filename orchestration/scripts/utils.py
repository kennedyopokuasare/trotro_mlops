import argparse
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data


def plot_pred_distribution(y, y_pred, y_label, y_pred_label, x_label, title):
    fig, _ = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y, label=y_label, fill=True)
    sns.kdeplot(y_pred, label=y_pred_label, fill=True)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    #plt.show()
    return fig


def main_args_parser(default_categorical_features, default_numerical_features):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, help="Path to the training data")
    parser.add_argument(
        "--validation_data_path",
        type=str,
        help="Path to the validation data",
    )

    parser.add_argument(
        "--categorical_features",
        type=List[str],
        default=default_categorical_features,
        help="List of categorical features",
    )
    parser.add_argument(
        "--numerical_features",
        type=List[str],
        default=default_numerical_features,
        help="List of numerical features",
    )

    return parser


def register_data(ml_client):
    """Registers training, validation, and test data with the given ML client."""

    train_data = Data(
        name="trotro_train_data",
        path="../data/yellow_tripdata_2024-01.parquet",
        type=AssetTypes.URI_FILE,
        description="Training data for the Trotro duration prediction model",
        tags={"source_type": "file", "source": "Local file"},
        version="1.0.0",
    )

    validation_data = Data(
        name="trotro_validation_data",
        path="../data/yellow_tripdata_2024-02.parquet",
        type=AssetTypes.URI_FILE,
        description="Validation data for the Trotro duration prediction model",
        tags={"source_type": "file", "source": "Local file"},
        version="1.0.0",
    )

    test_data = Data(
        name="trotro_test_data",
        path="../data/yellow_tripdata_2024-03.parquet",
        type=AssetTypes.URI_FILE,
        description="Test data for the Trotro duration prediction model",
        tags={"source_type": "file", "source": "Local file"},
        version="1.0.0",
    )

    try:
        ml_client.data.create_or_update(train_data)
        ml_client.data.create_or_update(validation_data)
        ml_client.data.create_or_update(test_data)

    except OSError as e:
        print(f"Error registering data: {e}")

    return train_data, validation_data, test_data
