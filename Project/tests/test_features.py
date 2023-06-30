
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import random
import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.data_manager import get_year_and_month

def test_weathersit_impuation(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config.weathersit_var,
    )
    #assert sample_input_data.weathersit.isna().sum() !=0

    # Fit
    transformer.fit(sample_input_data)
    assert transformer.fill_value == "Clear"
    # Transform
    subject = transformer.transform(sample_input_data)
    assert subject.weathersit.isna().sum() ==0
    
def test_weekday_impuation(sample_input_data):
    # Given
    transformer = WeekdayImputer(
        variables=config.model_config.weekday_var,
        ref_variables=config.model_config.dteday_var,
        len_day_name=config.model_config.len_var
    )
    #assert sample_input_data.weekday.isna().sum() !=0

    # When
    df = get_year_and_month(sample_input_data)
    subject = transformer.fit(df).transform(df)

    # Then
    assert subject.weekday.isna().sum() == 0

def test_mapper(sample_input_data):
    # Given
    transformer = Mapper(
        variables=config.model_config.season_var,
        mappings = config.model_config.dict_season
    )
    df = sample_input_data.dropna(subset=["season"])
    assert set(df.season.unique()) == set(config.model_config.dict_season.keys())

    # fit
    transformer.fit(df)
    assert transformer.mappings == config.model_config.dict_season

    # transform
    subject = transformer.transform(df)
    assert set(subject.season.unique()) == set(config.model_config.dict_season.values())

def test_hum_outlier(sample_input_data):
    # Given
    transformer = OutlierHandler(
        variables = config.model_config.hum_var,
        lower_bound_val = config.model_config.lower_bound_var,
        upper_bound_val = config.model_config.upper_bound_var
    )
    assert (sample_input_data.loc[8898,'hum'] < 2.625) | (sample_input_data.loc[8898,'hum'] > 123.625)

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[8898,'hum'] == 2.625

def test_onehot(sample_input_data):
    # Given
    transformer = WeekdayOneHotEncoder(
        variables = config.model_config.weekday_var
    )
    df = sample_input_data.dropna(subset=["weekday"]).reset_index(drop=True)
    initial_col_len = len(df.columns)
    initial_len = len(df)

    # When
    subject = transformer.fit(df).transform(df)

    # Then
    assert len(subject.columns) == initial_col_len+7
    assert len(subject) == initial_len
    for _ in range(10): # Positive test
        idx = random.randint(0, initial_len-1)
        day = df.loc[idx, "weekday"]
        assert subject.loc[idx, f"weekday_{day}"] == 1
    for _ in range(10): # Negative test
        idx = random.randint(0, initial_len-1)
        day = df.loc[idx, "weekday"]
        days = list(df.weekday.unique())
        days.remove(day)
        not_today = random.choices(days)[0]
        assert subject.loc[idx, f"weekday_{not_today}"] == 0
    