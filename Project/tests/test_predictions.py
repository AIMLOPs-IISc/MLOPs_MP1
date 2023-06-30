import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from sklearn.metrics import r2_score

from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = len(sample_input_data)

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    print(predictions)
    assert isinstance(predictions, np.ndarray ) #
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["cnt"]
    accuracy = r2_score(_predictions, y_true)
    assert accuracy > 0.9