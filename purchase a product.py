# We want to predict whether a customer is likely to purchase a product based on their age, time spent on the website, and whether they added a product to the cart. This is a binary classification problem where the output is either 1 (Customer will Purchase) or 0 (Customer will Not Purchase). To solve this problem, we use the Logistic Regression algorithm from Scikit-learn. The model is trained using sample customer data, and after training, it predicts whether a new customer is likely to purchase the product based on the details entered by the user.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = np.array([
    [25, 30, 0],
    [30, 40, 1],
    [20, 35, 0],
    [35, 45, 1]
])

y = np.array([0, 1, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

user_age = float(input("Enter Customer's Age: "))
user_time_spend = float(input("Enter Time Spent on Website: "))
user_added_to_cart = int(input("Enter 1 if Added to Cart, else 0: "))

user_data = np.array([[user_age, user_time_spend, user_added_to_cart]])

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("Prediction: Customer is likely to Purchase the Product.")
else:
    print("Prediction: Customer is NOT likely to Purchase the Product.")