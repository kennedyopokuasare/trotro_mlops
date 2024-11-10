import pandas as pd

from ..scripts.prepare_data import compute_training_features_with_target


def test_compute_training_features_with_target():
    # Sample data for testing
    data = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "tpep_dropoff_datetime": ["2023-01-01 10:30:00", "2023-01-01 11:15:00"],
            "passenger_count": [1, 2],
        }
    )

    # Convert to datetime
    data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
    data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])

    # Call the function
    result = compute_training_features_with_target(data)

    # Assertions to check the output
    assert result.shape[0] == 2  # Check if two rows are returned
    assert "hour_of_day" in result.columns  # Check if 'hour_of_day' is a column
    assert "day_of_week" in result.columns  # Check if 'day_of_week' is a column
    assert "duration" in result.columns  # Check if 'duration' is a column
    assert all(result["duration"] >= 1)  # Check if all durations are >= 1 minute
    assert all(result["duration"] <= 60)  # Check if all durations are <= 60 minutes


def test_empty_dataframe():
    # Test with an empty DataFrame
    data = pd.DataFrame(
        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count"]
    )
    result = compute_training_features_with_target(data)
    assert result.empty  # Check if the result is empty
