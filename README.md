# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ALDRIN LIJO J E 
RegisterNumber: 212222240007
*/
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```



## Output:
df.haed()

![image](https://user-images.githubusercontent.com/118544279/229984558-cad15d1c-967c-473a-a0c6-becd7a8ede7e.png)



df.tail()

![image](https://user-images.githubusercontent.com/118544279/229984585-cef73ba1-578d-48e6-9f61-5b01ee778949.png)


Array value of X

![image](https://user-images.githubusercontent.com/118544279/229984622-f6717a0d-01a9-4e13-8be6-1e00242b52b7.png)



Array value of Y

![image](https://user-images.githubusercontent.com/118544279/229984672-1fe56468-70fd-4bd3-8a70-80d36d4e1897.png)


Values of Y prediction

![image](https://user-images.githubusercontent.com/118544279/229984708-dd6a29d5-01b4-42d2-a7fc-55cbb540cdce.png)


Array values of Y test

![image](https://user-images.githubusercontent.com/118544279/229984760-ab7c73ff-db72-496b-aeb9-f29d1c92a2a6.png)


Training Set Graph

![image](https://user-images.githubusercontent.com/118544279/229984808-1eb5649c-b437-4411-96c8-c4898ecff3a2.png)


Test Set Graph

![image](https://user-images.githubusercontent.com/118544279/229984843-5660d6ff-a035-4833-be6a-9f24bd3f8c0f.png)


Values of MSE, MAE and RMSE

![image](https://user-images.githubusercontent.com/118544279/229984866-6423b4f3-4796-4d88-9a4e-cb2f6ad4d907.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
