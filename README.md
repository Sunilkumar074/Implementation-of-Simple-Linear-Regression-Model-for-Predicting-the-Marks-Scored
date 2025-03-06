# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sunil Kumar P.B.
RegisterNumber:  212223040213
*/
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

df.head()

![image](https://github.com/user-attachments/assets/c5514c3e-402c-4478-8abe-31277fdd3c0f)

df.tail()

![image](https://github.com/user-attachments/assets/e58b025c-3b44-42a6-a3b0-66a8bea4e365)

Array value of X


![image](https://github.com/user-attachments/assets/21a72b48-a269-4c4a-b93b-26c850813659)


Array value of Y

![image](https://github.com/user-attachments/assets/b4e9ee3f-8bae-4ded-ae36-87061cc63857)

Values of Y prediction

![image](https://github.com/user-attachments/assets/db31094b-a84e-4f3f-ae51-17f553c600d4)

Array values of Y test

![image](https://github.com/user-attachments/assets/d89fc587-a973-4af1-9ff4-588ab108be0b)

Training Set Graph

![image](https://github.com/user-attachments/assets/322fe0c8-aeb1-4494-ba4f-a6080d868524)

Test Set Graph

![image](https://github.com/user-attachments/assets/47c0b12c-5d79-497d-88fb-acd819dd7235)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/d0ba63a8-43f4-4b97-8aa1-34f2bc6fa4cf)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
