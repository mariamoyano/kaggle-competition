from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
def HistGradientBoostingR(X_train,y_train):
        params = { 'loss': ["least_squares", "least_absolute_deviation"],
                   'learning_rate': [0.1, 0.5, 1]
            }
        Hist = GridSearchCV(estimator=HistGradientBoostingRegressor(), 
            param_grid=params, scoring='neg_mean_squared_error',n_jobs=3,iid=False, cv=5,verbose=5)
        Hist.fit(X_train, y_train)
        print("Train complete")
        return Hist

def DecisionTreeR(X_train,y_train):
        params = { 'criterion': ["mse", "friedman_mse", "mae"],
                   'splitter': ["best", "random"],
                   "max_features":["auto", "sqrt", "log2"],
                   "random_state": [0, "RandomState"]

            }
        dtree = GridSearchCV(estimator=DecisionTreeRegressor(), 
            param_grid=params, scoring='neg_mean_squared_error',n_jobs=3,iid=False, cv=5,verbose=5)
        dtree.fit(X_train, y_train)
        print("Train complete")
        return dtree

def GradientBoostingR(X_train,y_train):
   
    params = {"learning_rate":[0.15,0.1,0.05],
            "n_estimators":[1500,1750],
            "min_samples_split":[50, 100],
            "min_samples_leaf": [50, 100]}

    gradient = GridSearchCV(estimator=GradientBoostingRegressor(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
            param_grid=params, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5,verbose=5)
    gradient.fit(X_train, y_train)
    print("Train complete")
    return gradient

def RamdomForestR(X_train,y_train):
    params = { 'n_estimators': [1500]
               
            }

    randomForest = GridSearchCV(estimator=RandomForestRegressor( oob_score = True), 
            param_grid=params, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5,verbose=5)
    randomForest.fit(X_train, y_train)
    print("Train complete")
    return randomForest

def KNeighborsR(X_train,y_train):
    params =   {"n_neighbors":[5,10,15],
                "algorithm":['kd_tree',"ball_tree","brute"],
                "leaf_size":[50], 
                "p":[1,2]}

    KNeighbors = GridSearchCV(estimator=KNeighborsRegressor(), 
            param_grid=params, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5,verbose=5)
    KNeighbors.fit(X_train, y_train)
    print("Train complete")
    return KNeighbors

def printResult(estimator,test_dataset,X_test,y_test):
    printMetric = lambda label,value:print(f"\t {label}: {round(value,3)}")
    y_pred = estimator.predict(X_test)
    y_predict = estimator.predict(test_dataset)
    printMetric("rsme", mean_squared_error(y_test, y_pred))
    print("      r2score:",r2_score(y_test, y_pred, multioutput='raw_values')[0])
    return y_predict

