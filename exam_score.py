import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

mantudata = {
    'Hours_study': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_score': [50, 60, 65, 70, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(mantudata)

x = df[['Hours_study']]
y = df[['Exam_score']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

user_input = float(input("How many hours did you study? : "))
predicted_score = model.predict([[user_input]])

print(f"Estimated Exam Score: {predicted_score[0][0]:.2f}")