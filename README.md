# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries like 'pandas' for data manipulation and 'numpy' for numerical operations.
2. Load the CSV file ('Placement_Data.csv') into a pandas DataFrame and preview the first 5 rows using 'df.head()'.
3. Make a copy of the original DataFrame ('df') to preserve the original data.
4. Drop the columns 'sl_no' (serial number) and 'salary' because they are not required for modeling.
5. '.isnull().sum()' returns the count of missing values for each column.
6. '.duplicated().sum()' returns the count of duplicate rows in the dataset.
7. The 'LabelEncoder' from 'sklearn' is used to transform string labels into numeric labels.
8. This is done for columns 'gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', and 'status' where each unique category is assigned a numeric value.
9. 'x' is created by selecting all columns except for the last column ('status'), which is the target variable.
10. 'y' is the 'status' column, which represents whether the student got placed or not (binary classification).
11. The data is split using 'train_test_split' from 'sklearn'.
12. 'test_size=0.2' means that 20% of the data will be used for testing, and 80% will be used for training.
13. 'random_state=0' ensures that the split is reproducible.
14. 'LogisticRegression(solver="liblinear")' creates a logistic regression classifier using the 'liblinear' solver (good for smaller datasets).
15. 'lr.fit(x_train, y_train') trains the model on the training data.
16.  Use the trained logistic regression model ('lr') to predict the target values ('status') for the test set ('x_test').
17.  'accuracy_score' compares the predicted values ('y_pred') with the true values ('y_test') and calculates the accuracy of the model.
18.  'confusion_matrix(y_test, y_pred)' outputs the confusion matrix based on the true values and predicted values.
19.  'classification_report(y_test, y_pred)' provides metrics such as precision, recall, F1-score, and support for each class (in this case, whether a student got placed or not).
20.  The input should match the features used in the model (numeric values representing different attributes of the student).

 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sukirthana . M 
RegisterNumber: 212224220112

import pandas as pd
import numpy as np
df=pd.read_csv('Placement_Data.csv')
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis = 1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1

x=df1.iloc[:,:-1]
x

y=df1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![image](https://github.com/user-attachments/assets/241892bf-1f05-4eaf-bff2-c27d18444ac1)
![image](https://github.com/user-attachments/assets/0384b301-5196-4b05-b644-ef33c5f9fad5)
![image](https://github.com/user-attachments/assets/6c6df437-ed23-4814-8790-9f420fd4a302)
![image](https://github.com/user-attachments/assets/a94841d5-5a68-463b-a4aa-44dca3b32e5a)
![image](https://github.com/user-attachments/assets/a064ea05-6432-4c09-91ea-a1bd04f6c9cf)
![image](https://github.com/user-attachments/assets/bc2da33c-421b-45e4-a5ed-6f8688e4dae9)
![image](https://github.com/user-attachments/assets/088d3ebf-cf47-4079-a81e-07335f3c229d)
![image](https://github.com/user-attachments/assets/b8ee1f56-a9bf-4e8c-acd7-e062f3536e1e)
![image](https://github.com/user-attachments/assets/2d8915a3-e268-4bf3-9b89-3f74ae08c155)
![image](https://github.com/user-attachments/assets/2eeae7e6-91c3-454c-8c33-ab9f7b627889)
![image](https://github.com/user-attachments/assets/8c31fcdb-2c0e-4040-8a5b-ff5d1fafefc7)
![image](https://github.com/user-attachments/assets/528e8f28-a13f-4161-a161-724f40ac8951)









## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
