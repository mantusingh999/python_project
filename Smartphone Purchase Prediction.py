# #You have been tasked with creating a decision tree model to predict whether  a person is likelty to purchase  a new smartphone based on their age,income,and education level.You are provided with a dataset containg these attributes and  the target variable indicating whether the person made a purchase or not
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array([
    [25, 50000, 3],
    [35, 90000, 2],
    [40, 60000, 5],
    [45, 80000, 3],
    [20, 30000, 2],
    [55, 120000, 4],
    [28, 40000, 1],
    [32, 100000, 3],
    [38, 75000, 2]
])

y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

age = float(input("Enter Age: "))
income = float(input("Enter Annual Income: "))
education = float(input("Enter Education Level: "))

user_input = np.array([[age, income, education]])

prediction = model.predict(user_input)

if prediction[0] == 1:
    print("Prediction: The person is likely to purchase a new smartphone.")
else:
    print("Prediction: The person is unlikely to purchase a new smartphone.")