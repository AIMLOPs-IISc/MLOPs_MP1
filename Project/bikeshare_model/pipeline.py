import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from feature_engine.selection import DropFeatures

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder

bikeshare_pipe=Pipeline([
    ('weekday_imputer', WeekdayImputer(config.model_config.weekday_var,config.model_config.dteday_var,config.model_config.len_var)),
    ('weathersit_imputer', WeathersitImputer(config.model_config.weathersit_var)),
    ##Mapper##
    ('map_yr',Mapper(config.model_config.yr_var,config.model_config.dict_yr)),
    ('map_mnth',Mapper(config.model_config.mnth_var,config.model_config.dict_mnth)),
    ('map_season',Mapper(config.model_config.season_var,config.model_config.dict_season)),
    ('map_weathersit',Mapper(config.model_config.weathersit_var,config.model_config.dict_weathersit)),
    ('map_holiday',Mapper(config.model_config.holiday_var,config.model_config.dict_holiday)),
    ('map_workingday',Mapper(config.model_config.workingday_var,config.model_config.dict_workingday)),
    ('map_hr',Mapper(config.model_config.hr_var,config.model_config.dict_hr)),
    ##Outliers##
    ('temp_outlier',OutlierHandler(config.model_config.temp_var,config.model_config.lower_bound_var,config.model_config.upper_bound_var)),
    ('atemp_outlier',OutlierHandler(config.model_config.atemp_var,config.model_config.lower_bound_var,config.model_config.upper_bound_var)),
    ('hum_outlier',OutlierHandler(config.model_config.hum_var,config.model_config.lower_bound_var,config.model_config.upper_bound_var)),
    ('windspeed_outlier',OutlierHandler(config.model_config.windspeed_var,config.model_config.lower_bound_var,config.model_config.upper_bound_var)),
    ##Onehot encoding##
    ('weekday_onehotencoding',WeekdayOneHotEncoder(config.model_config.weekday_var)),
    ##Drop columns##
    ('drop_features_dteday', DropFeatures(features_to_drop = ['dteday'])),
    ('drop_features_casual', DropFeatures(features_to_drop = ['casual'])),
    ('drop_features_registered', DropFeatures(features_to_drop = ['registered'])),
    ('drop_features_weekday', DropFeatures(features_to_drop = [config.model_config.weekday_var])),
    ##scale##
    ('scaler', StandardScaler()),
    ('model_random_regressor'   , RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
])