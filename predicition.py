import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
#feature engineering
def set_missing_security_deposit(df):
    process = df.drop(["price", "cleaning_fee"], axis = 1)
    temp = process.pop("security_deposit")
    process.fillna(0,inplace = True)
    process.insert(0,"security_deposit", temp)
    know = process[process.security_deposit.notnull()].as_matrix()
    unknow = process[process.security_deposit.isnull()].as_matrix()
    x = know[:,1:]
    y = know[:,0]
    rfr = RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = 1)
    rfr.fit(x,y)
    predicted = rfr.predict(unknow[:,1:])
    df.loc[(df.security_deposit.isnull()), "security_deposit"] = predicted
    return df
def set_missing_cleaning_fee(df):
    process = df.drop(["price", "security_deposit"], axis = 1)
    temp = process.pop("cleaning_fee")
    process.fillna(0,inplace = True)
    process.insert(0,"cleaning_fee", temp)
    know = process[process.cleaning_fee.notnull()].as_matrix()
    unknow = process[process.cleaning_fee.isnull()].as_matrix()
    x = know[:,1:]
    y = know[:,0]
    rfr = RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = 1)
    rfr.fit(x,y)
    predicted = rfr.predict(unknow[:,1:])
    df.loc[(df.cleaning_fee.isnull()), "cleaning_fee"] = predicted
    return df
data = set_missing_security_deposit(data)
data = set_missing_cleaning_fee(data)
data.fillna(0, inplace = True)
data["if_has_review"] = np.where(data["number_of_reviews"] == 0,0,1)
data["bedroom per people"] = data["bedrooms"]/data["accommodates"]
data["bathroom per people"] = data["bathrooms"]/data["accommodates"]
data["beds per people"] = data["beds"] / data["accommodates"]
def filtering_region(df):
    lat_min = df["latitude"].min()
    lon_min = df["longitude"].min()
    lat_dis = (df["latitude"].max() - df["latitude"].min()) / 15
    lon_dis = (df["longitude"].max() - df["longitude"].min()) / 20
    region = {}
    i = 1
    for k in range(1, 11):
        for j in range(1, 21):
            key = "region {}".format(i)
            value = [lat_min + lat_dis * (k - 1), lat_min + lat_dis * k, lon_min + lon_dis * (j - 1),
                     lon_min + lon_dis * j]
            region[key] = value
            i += 1
    s = df.iloc[:, 0].size
    df["region"] = 0
    for i in range(s):
        for key, value in region.items():
            if value[2] <= df.loc[i, "longitude"] <= value[3] and value[0] <= df.loc[i, "latitude"] <= value[1]:
                df.loc[i, "region"] = key
    times = {}
    for key in region.keys():
        count = 0
        for l in range(s):
            if df.loc[l, "region"] == key:
                count += 1
        times[key] = count

    df['count'] = 0
    for l in range(s):
        for key, value in times.items():
            if df.loc[l, "region"] == key:
                df.loc[l, "count"] = value

    df.drop("region", axis=1, inplace=True)
    return df
data = filtering_region(data)
from sklearn.preprocessing import PolynomialFeatures
poly_data = data[["accommodates", "bathrooms", "bedrooms", "beds", "security_deposit","cancellation_policy_strict",
                 "cleaning_fee", "if_has_review","bedroom per people", "bathroom per people", "beds per people",
                 "distance", "host_is_superhost_t", "room_type_Entire home/apt", "extra_people",
                 "property_type_Bed & Breakfast","minimum_nights","review_scores_location",
                 "room_type_Private room", "room_type_Shared room","cancellation_policy_flexible"]]
poly = PolynomialFeatures(interaction_only=True)
df_poly = pd.DataFrame(poly.fit_transform(poly_data), columns=poly.get_feature_names(poly_data.columns))
df_poly = df_poly.iloc[:,22:]
data = data.join(df_poly)

