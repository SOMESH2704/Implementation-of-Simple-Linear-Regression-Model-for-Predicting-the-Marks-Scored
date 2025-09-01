# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Someshwar Kumar
RegisterNumber: 212224240157
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
print("Name:Someshwar Kumar")
print("Reg.No:212224240157")
plt.scatter(X_train,Y_train,color="brown")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name:Someshwar Kumar")
print("Reg.No:212224240157")
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name:Someshwar Kumar")
print("Reg.No:212224240157")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae )
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:

Head:

<img width="168" height="207" alt="image" src="https://github.com/user-attachments/assets/a099b56b-b4b6-4715-afe4-1c7664274e81" />

Tail:

<img width="178" height="201" alt="image" src="https://github.com/user-attachments/assets/ad4c9393-cfe3-4915-ae38-65ad2db9921f" />

X:

<img width="147" height="541" alt="image" src="https://github.com/user-attachments/assets/4cd084fa-1102-4a77-a5e3-4401756c149a" />

Y:

<img width="716" height="47" alt="image" src="https://github.com/user-attachments/assets/b2adbf0b-6923-4a2f-b1b9-cdaf794677b0" />

Y_pred:

<img width="708" height="53" alt="image" src="https://github.com/user-attachments/assets/fcd25593-0fd5-4412-ae40-79c1b0274270" />

Y_test:

<img width="562" height="27" alt="image" src="https://github.com/user-attachments/assets/fbb7b432-dde5-406d-95cb-bb0157f1de27" />

Training Set:

<img width="646" height="502" alt="Screenshot 2025-08-18 115145" src="https://github.com/user-attachments/assets/1c105f0f-b7ac-46b7-9032-16504cc33bf0" />

Test Set:

<img width="698" height="501" alt="Screenshot 2025-08-18 115154" src="https://github.com/user-attachments/assets/227f7f7c-d3b2-4cd8-b81a-2d9d402ffbb2" />

Values:

<img width="252" height="99" alt="Screenshot 2025-08-18 115201" src="https://github.com/user-attachments/assets/00a61e80-e5e5-4bfd-80c7-7a397066f1c7" />








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
