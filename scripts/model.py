import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime
import os
import joblib
#fxn to load data set
def load_Data(path=None):
    basedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(basedir,"..","data","transformed")
    datapath = os.path.join(datadir,"transformed_clean_car_dataset.csv")
    dataset = pd.read_csv(datapath)
    return dataset
#fxn to define the modelpaths
def load_modelData(path):
    basedirectory = os.path.dirname(os.path.abspath(__file__))
    modeldir = os.path.join(basedirectory,"..","models")
    return os.path.join(modeldir,path)
#define the x and y variables and
#split adata for modelling
def splitdata(df,target='Present Price',test_size=0.2,random_state=1):
    x = df.drop(columns=[target],axis=1)
    y = df[target]
    price_band = pd.qcut(y, q=3, labels=["Low", "Mid", "High"])
    y_log = np.log1p(y)
    return train_test_split(x,y_log,stratify=price_band,test_size=test_size,random_state=random_state)
#fxn to perform prediction modelling
def modelling(df,target='Present_Price'):
    #split data
    x_train,x_test,y_train,y_test = splitdata(df,target = target)
    print("Training feature order:")
    print(x_train.columns.tolist())
    #initializemodels
    lineargr = LinearRegression()
    decisionrgr = DecisionTreeRegressor(max_depth=5,min_samples_split=10,random_state=1)
    randforestrgr = RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_split=10,random_state=1)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, reg_alpha=0.1, reg_lambda=1.0, random_state=1)
    #fitting
    lineargr.fit(x_train,y_train)
    decisionrgr.fit(x_train,y_train)
    randforestrgr.fit(x_train,y_train)
    xgb.fit(x_train,y_train)
    #save the models
    joblib.dump(lineargr,load_modelData("linear_app.pkl"))
    joblib.dump(decisionrgr,load_modelData("decisiontree_app.pkl"))
    joblib.dump(randforestrgr,load_modelData("randomforest_app.pkl"))
    joblib.dump(xgb,load_modelData("xgb_app.pkl"))
    return {
        'status': 'success',
        'models_saved': [
            'linear_app.pkl',
            'decisiontree_app.pkl',
            'randomforest_app.pkl',
            'xgb_app.pkl'
        ],
        'timestamp': datetime.now().isoformat()
    }
if __name__ == "__main__":
    df = load_Data()
    result = modelling(df)
    print(result)







