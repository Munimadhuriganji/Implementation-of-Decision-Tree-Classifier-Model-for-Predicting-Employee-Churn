# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GANJI MUNI MADHURI
RegisterNumber:  212223230060
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
x = data[["satisfaction_level", "last_evaluation", "number_project", 
          "average_montly_hours", "time_spend_company", "Work_accident", 
          "promotion_last_5years", "salary"]]
y = data["left"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Decision Tree Model:", accuracy)
input_data = pd.DataFrame([[0.5, 0.8, 9, 260, 6, 0, 1, 2]], columns=x.columns)
predicted_class = dt.predict(input_data)
print("Prediction for input data:", predicted_class)
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=["Stayed", "Left"], filled=True)
plt.title("Decision Tree - Employee Churn Prediction")
```

## Output:

![image](https://github.com/user-attachments/assets/2a928ded-7c28-4bed-b3fe-87c791aa8164)

![image](https://github.com/user-attachments/assets/c92c7ee4-7dc3-49d0-a61c-32d994472ecb)

![image](https://github.com/user-attachments/assets/602d4741-9e24-4278-a36e-a18c8e340ba8)
![Uploading image.png…]()




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
