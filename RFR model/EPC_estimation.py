from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
import os
from sklearn.model_selection import RandomizedSearchCV
import sklearn.model_selection as ms
import pickle
import sklearn.metrics as sm
from sklearn.metrics import mean_squared_error
import sklearn.ensemble as se
import scipy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
from rfpimp import permutation_importances
from math import sqrt

# Training the RFR model
def train_model():
    # features datasets path
    frames =[]  #I:/TFlearning_VGGF/Features/Jiedao_Features_5/
    for dirpath, filename, filenames in os.walk('I:/data/results_622/Jiedao_features/'):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.csv':
                path = dirpath + '\\' + filename
                da = pd.read_csv(path,encoding='gbk')
                frames.append(da)
    Data = pd.concat(frames)
    Data = Data.drop(columns=['NAME','COUNTYNAME','COUNTYCODE','XZDM'])
    colList = ['FC0', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FC7', 'FC8', 'FC9', 'FC10', 'FC11', 'FC12',
             'FC13', 'FC14', 'FC15', 'FC16', 'FC17', 'FC18', 'FC19', 'FC20', 'FC21', 'FC22', 'FC23', 'FC24',
             'NTL_MEAN', 'Population_statistic','Area_km','EPC_kwh_statistic', 'Population density', 'price_MIN','price_MAX','price_MEAN']
    # rename col names
    Data.columns = colList
    y = Data['EPC_kwh_statistic']
    X = Data[['NTL_MEAN','Population density']]
    train_x, X_test, train_y, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    scores = []
    
    # The model uses trained hyperparameters
    rf = RandomForestRegressor(oob_score=True,n_estimators=134,max_depth=12, max_features='auto',
                               min_samples_split=2,min_samples_leaf=2,criterion='mse',random_state=90)
    rf.fit(X, y)
    pred_train_y = rf.predict(X_test)
    mse_train = mean_squared_error(y_test, pred_train_y)
    print(mse_train)
    print('oob_score:', rf.oob_score_)
    print('r2:', rf.score(X_test, y_test))
    
    # save the model to disk
    filename = 'I:/data/RFmodel.sav'
    pickle.dump(rf, open(filename, 'wb'))
    '''
    Adjust parameter: Scikit-learn RandomizedSearchCV class 
    optimal hyperparameters of RFR model using GridSearchCV
    '''
    rfr = RandomForestRegressor(criterion='mse',oob_score=True, # min_samples_leaf=2, min_samples_split=6,
                                         random_state=90)

    n_estimators =[np.arange(10, 300, 10)] # np.arange(200, 260, step=1)
    max_features = ['sqrt']
    max_depth = [15] #np.arange(1, 30, step=1)  # list(np.arange(10, 100, step=10)) + [None]
    min_samples_split = [3] #np.arange(2, 8+1, step=1)
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True,False]
    oob_score = [True]
    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        # "max_depth": max_depth,
        # "min_samples_split": min_samples_split,
        # "min_samples_leaf": min_samples_leaf,
        # "bootstrap": bootstrap,
        #'oob_score':oob_score
    }
    # GridSearchCV 
    #model_random = GridSearchCV(estimator=model_random, param_grid=param_grid, n_jobs=-1, cv=5) #scoring='neg_mean_absolute_error',

    #rfr.fit(X, y)
    #y_pre = rfr.predict(X_test)
    #score_ = cross_val_score(rfr, X_test, y_test, cv=10, scoring="neg_mean_squared_error").mean()
    #print("mse==",score_ * -1)
    #rf_val_mae = mean_absolute_error(y_pre, y_test)
    #print("Validation MAE for Random Forest Model: {:,.2f}".format(rf_val_mae))


def EPC_estimation():
    filename = './RFmodel.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    for dirpath, filename, filenames in os.walk('./patch_Features/'): 
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.csv':
                path = dirpath + '\\' + filename
                Data = pd.read_csv(path,encoding='gbk')

                Data_cal = Data.drop(columns=['x','y'])
                order_2 = Data_cal.columns[2:].tolist()
                order_1 = Data_cal.columns[:2].tolist()
                order = order_2+order_1

                x_data = Data_cal[order]

                y_pre = loaded_model.predict(x_data)
                data = Data[['x', 'y']]
                data['epc'] = y_pre
                print(y_pre)
                data.to_csv('./data/' + 'EPC_' + filename, index=False)


"""
    Calculate the correction factor for each sub-district in each year: 
    calculate the ratio of statistical data to predicted data
    
"""
def normalization(year):

    # Read the EPC statistics at the sub-district level
    sensus_EPC = pd.read_csv('I:/data/Jiedao_data_613/Jiedao_features/'+'Jiedao_'+year+'.csv',encoding='gbk')
    
    # Read the EPC prediction results at the sub-district level
    pre_EPC_jiedao = pd.read_csv('./EPC_predic_results/EPC_Shenzhen_' + year + '.csv')
    
    sensus_EPC['NAME'] = sensus_EPC['NAME'].apply(lambda x: x[:3]).tolist()
    pre_EPC_jiedao = pre_EPC_jiedao[['NAME', 'SUM']]
    results_jiedao = pd.merge(sensus_EPC,pre_EPC_jiedao,on='NAME')

    # Calculate the regularization factor for each sub-district
    results_jiedao['rate'] = results_jiedao['EPC_statistic'] / results_jiedao['SUM']
    '''
      Correct the predicted value of each pixel for each sub-district according to the regularization factor
    '''
    # Get the set of pixel coordinates contained in each street
    results = results_jiedao[['NAME','rate']]

    xy_pd = pd.read_csv('./EPC_pre_name/EPC_pre_name_'+year+'.csv',encoding='gbk')

    xy_pd['NAME'] = xy_pd['NAME'].apply(lambda x: x[:3]).tolist()
    xy_pd = xy_pd[['x','y','epc','NAME']] # Here epc refers to the predicted value of each sub-district

    datamerge = pd.merge(xy_pd,results,on='NAME') # Information contained in each street

    datamerge['epc_normal'] = datamerge['epc']*datamerge['rate']

    # save datasets
    datamerge.to_csv('./EPC_results_normal/normal_EPC_'+year+'.csv',encoding='gbk',index=False)


if __name__=="__main__":
    os.chdir('I:/')
    #EPC_estimation()

    #train_model()
    
    for i in range(2013,2020): # normalized the EPC estimation results from 2013 to 2019
        normalization(str(i))


