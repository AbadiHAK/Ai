import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('Ages_salary.csv')
Ages = data.iloc[ : , : -1]
Salary = data.iloc[ : , 1 ]
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train,y_test =train_test_split(Ages,Salary,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
y_pred = regression.predict(X_test)
y_pred_train = regression.predict(X_train)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,y_pred_train,color='blue')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title('Ages and Salary')
plt.xlabel('Ages')
plt.ylabel('salary')
plt.show()
score=regression.score(X_train,y_train)
print('Score:',score*100)
score1=regression.score(X_test,y_test)
print('Score',score*100)