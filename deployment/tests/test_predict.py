from unittest.mock import MagicMock

import pandas as pd
import pytest

from ..prediction_service import TrotroDurationPredictionService


# pylint: disable=redefined-outer-name
@pytest.fixture
def mock_model_fixture():
    """Fixture to create a mock model."""
    model = MagicMock()
    model.predict.return_value = [5.0]  # Mock prediction value
    return model


@pytest.fixture
def prediction_service(mock_model_fixture):
    """Fixture to create an instance of TrotroDurationPredictionService with a mock model."""
    return TrotroDurationPredictionService(
        model_uri="mock_model_uri", model=mock_model_fixture
    )


def test_constructor_with_mock_model(prediction_service):
    """Test the constructor with a mocked model."""
    assert prediction_service.model is not None
    assert prediction_service.model.predict.return_value == [5.0]


def test_compute_features(prediction_service):
    """Test the compute_features method."""
    input_data = pd.DataFrame(
        {"tpep_pickup_datetime": ["2023-01-01 10:00:00", "2023-01-02 15:30:00"]}
    )

    expected_output = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-02 15:30:00"]
            ),
            "hour_of_day": [10.0, 15.0],
            "day_of_week": [6.0, 0.0],  # Sunday and Monday
        }
    )

    result = prediction_service.compute_features(input_data)

    pd.testing.assert_frame_equal(
        result[["hour_of_day", "day_of_week"]],
        expected_output[["hour_of_day", "day_of_week"]],
    )


def test_predict(prediction_service):
    """Test the predict method."""
    trotro_data = {"tpep_pickup_datetime": "2023-01-01 10:00:00"}
    result = prediction_service.predict(trotro_data)

    assert result == 5.0  # Check if the prediction is as expected
    prediction_service.model.predict.assert_called_once()  # Ensure the model's predict method was called
