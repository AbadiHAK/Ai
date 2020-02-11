import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
x= data.iloc[ : , 1:2 ].values
y= data.iloc[: ,2 ].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train ,y_test = train_test_split(x , y ,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x,y)

def SimpleLinear():
    plt.scatter(x,y,color='red')
    plt.plot(x,regression.predict(x),color='blue')
    plt.title('simple Linear Regerssion'),
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.show()
    return
SimpleLinear()

from sklearn.preprocessing import PolynomialFeatures  
regressionPloy= PolynomialFeatures(degree=4)
Xpoly=regressionPloy.fit_transform(x)
PolyRegression=LinearRegression()
PolyRegression.fit(Xpoly,y)

def PolyLinearRegression():
    plt.scatter(x,y,color='red')
    plt.plot(x,PolyRegression.predict(regressionPloy.fit_transform(x)),color='blue')
    plt.title('Polynomial')
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.show()
    return
PolyLinearRegression()
    
    
    
    
    
    
    
    
    
    
    