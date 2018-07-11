import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew
 
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)

def data_preprocess(train,test):
        
    outlier_idx = []

    for i in train[train['GrLivArea'] > 4000]['GrLivArea'].index:
        outlier_idx.append(i)
    
    for i in train[train['GarageArea'] > 1200]['GarageArea'].index:
        outlier_idx.append(i)
    
    
    for i in train[train['TotalBsmtSF'] > 3000]['TotalBsmtSF'].index:
        outlier_idx.append(i)
    
    for i in train[train['1stFlrSF'] > 3000 ]['1stFlrSF'].index:
        outlier_idx.append(i)

    for i in train[train['MasVnrArea'] > 800]['MasVnrArea'].index:
        outlier_idx.append(i)
    
    for i in train[train['BsmtFinSF1'] > 2000]['BsmtFinSF1'].index:
        outlier_idx.append(i)
    
    for i in train[train['LotFrontage'] > 150]['LotFrontage'].index:
        outlier_idx.append(i)
    
    for i in train[train['WoodDeckSF'] > 600]['WoodDeckSF'].index:
        outlier_idx.append(i)
    
    for i in train[train['2ndFlrSF'] > 1650]['2ndFlrSF'].index:
        outlier_idx.append(i)
    
    for i in train[train['OpenPorchSF'] > 400]['OpenPorchSF'].index:
        outlier_idx.append(i)
    
    for i in train[train['LotArea'] > 50000]['LotArea'].index:
        outlier_idx.append(i)
    
    for i in train[train['BsmtUnfSF'] > 2000]['BsmtUnfSF'].index:
        outlier_idx.append(i)
        
    train.drop(train.index[outlier_idx],inplace=True)
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    
    to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    all_data = all_data.drop(to_delete,axis=1)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    #log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train,X_test,y
    
def model_random_forecast(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {}
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_xgb_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid = {}
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('eXtreme Gradient Boosting regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


# read data, build model and do prediction
train = pd.read_csv("train.csv") # read train data
test = pd.read_csv("test.csv") # read test data
Xtrain,Xtest,ytrain = data_preprocess(train,test)


test_predict,score = model_random_forecast(Xtrain,Xtest,ytrain)
test_predict,score = model_xgb_regression(Xtrain,Xtest,ytrain)
test_predict,score = model_extra_trees_regression(Xtrain,Xtest,ytrain)
test_predict,score = model_gradient_boosting_tree(Xtrain,Xtest,ytrain)

create_submission(np.exp(test_predict),score)