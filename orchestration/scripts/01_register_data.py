from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data


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
