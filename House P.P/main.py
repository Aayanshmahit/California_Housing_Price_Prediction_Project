import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = model.pkl
PIPELINE_FILE = pipepline.pkl

def build_pipeline(num_atributes , cat_atributes):
    num_atributes = Pipeline([
        ("Simple_Imputer", SimpleImputer(strategy = "median")),
        ("standard_scaler", StandardScaler())
        ])
    
    cat_atributes = Pipeline([
        "onehot" , OneHotEncoder()
        ])
    
    full_pipeline = ColumnTransformer([
        ("num" , num_pipeline , num_atributes),
        ("cat" , cat_pipeline , cat_atributes)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing  = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut["housing_income"] , bins = [0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf()] , labels = [1 ,2 , 3, 4, 5]

    split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)
    
    for test_index , train_index in split.split(housing , housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv" , index = False)
        housing = housing[train_index].drop("income_cat" , axis = 1)

        housing_label = housing["median_house_value"].copy()
        housing_features = housing.drop["median_house_value"]

        housing_cat = housing_features["ocean_proximity"].to_array
        housing_num = housing_features.drop("ocean_proximity")

        