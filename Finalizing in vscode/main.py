#IMPORTING ALL THE LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#READING THE DATA
housing = pd.read_csv("housing.csv")

#SELCTING THE ROW BY WHICH I WILL APLLY StratifiedShuffleSplit
housing["income_cat"] = pd.cut["median_income"] , bins = [0.0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf] , labels = [1 ,2, 3, 4 ,5]

#SPLITTING THE TEST AND TRAIN SET BY StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(random_state = 42 , n_split = 1 , test_size = 0.2  )
for train_index , test_index in split.split(housing , housing["income_cat"]):
    train_data = housing.loc[:train_index].drop("income_cat" , index = 1)
    test_data = housing.loc[test_index:].drop("income_cat" , index =1)

housing = train_data.copy()

housing_labels = housing["median_income"].copy()
housing  = housing.drop("median_income" , axis = 1)

housing_num = housing.drop("ocean_proximity" , axis = 1)
housing_cat = housing["ocean_proximity"]

num_pipeline = Pipeline([
    ("Simple_Imputer" , SimpleImputer(strategy = "mean"))
    ("standard_scaler" , StandardScaler())
])

cat_pipeline =Pipeline([
    ("OneHot_Encoder" , OneHotEncoder())
])

full_pipeline = ColumnTransformer(
    "num" , num_pipeline, housing_num ,
    "cat" , cat_pipeline, housing_cat
)

housing_prepared = full_pipeline.fit_transform(housing)
print(housing)