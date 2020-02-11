import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#--------------------------------------------
data = pd.read_csv('50_Startups.csv')
x=data.iloc[:, :-1].values
y= data.iloc[:,4].values
#---------------------------------------------
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
encoderX=LabelEncoder()
x[:,3]=encoderX.fit_transform(x[:,3])
hotEncoder=OneHotEncoder(categorical_features=[3])
x=hotEncoder.fit_transform(x).toarray()
#---------------------------------------------------
x = x[: , 1:]
#-----------------------------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#-----------------------------------------------------------
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
#---------------------------------------------------------------------
y_pred = reg.predict(X_test)
#----------------------------------------------------------------------
import statsmodels.api as sm
x= np.append(arr = np.ones((50,1)).astype(int),values=x,axis =1)
x_opt = x[:,[0,1,2,3,4,5]]
regression_OLS = sm.OLS(endog =y ,exog = x_opt).fit()
regression_OLS.summary()
#-------------------------------------------------------
x_opt = x[:,[0,1,3,4,5]]
regression_OLS = sm.OLS(endog =y ,exog = x_opt).fit()
regression_OLS.summary()
x_opt = x[:,[0,3,5]]
regression_OLS = sm.OLS(endog =y ,exog = x_opt).fit()
regression_OLS.summary()
x_opt = x[:,[0,3]]
regression_OLS = sm.OLS(endog =y ,exog = x_opt).fit()
regression_OLS.summary()
