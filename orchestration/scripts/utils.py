import argparse
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer


def feature_encoding(
    df_train: pd.DataFrame, df_val: pd.DataFrame, save_vectorizer: bool
):
    """Encodes the features in the given files."""
    print("Feature encoding...")

    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(df_train.to_dict(orient="records"))

    x_val = dv.transform(df_val.to_dict(orient="records"))

    if save_vectorizer:
        pass

    return x_train, x_val


def plot_pred_distribution(y, y_pred, y_label, y_pred_label, x_label, title):
    fig, _ = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y, label=y_label, fill=True)
    sns.kdeplot(y_pred, label=y_pred_label, fill=True)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def main_args_parser(categorical_features, numerical_features):
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
        default=categorical_features,
        help="List of categorical features",
    )
    parser.add_argument(
        "--numerical_features",
        type=List[str],
        default=numerical_features,
        help="List of numerical features",
    )

    return parser
