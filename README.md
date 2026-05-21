# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program & Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRIGANESH S
RegisterNumber:  25017178
*/
```
```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("drive/MyDrive/ML/Employee.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/bec2d980-826a-44c4-8a3e-d96629c63928)
```
data.info()
```
![image](https://github.com/user-attachments/assets/2d015ae9-8a78-4fc8-a21e-9af8b503b80e)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/89fe5c0f-433f-4f19-b9f4-998413ffcbb6)
```
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/acdec0bd-0bb2-4808-89b8-bc9c0b354bb3)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/c0fe2818-69da-45ab-a105-fafb8702cf37)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/f42ce1a0-334a-4840-a56c-59c5a790304f)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/01d6ac58-0762-4f69-bc98-a02788edd157)

```
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![image](https://github.com/user-attachments/assets/0ea08056-38b9-451c-8db4-45225d9f3a77)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
