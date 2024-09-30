import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


X = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)
Y = np.array([60, 75, 85, 90, 95])
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)
slope = model.coef_[0]
intercept = model.intercept_
plt.scatter(X, Y, label='Data')
plt.plot(X, Y_pred, color='red', label=f'Regression Line (y = {slope:.2f}x + {intercept:.2f})')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.title('Linear Regression: Study Hours vs. Exam Score')
plt.show()
new_SH = int(input("Enter the number of hours: "))
pred_Sc = model.predict(np.array([[new_SH]]))
print(f'Predicted Exam Score for {new_SH} study hours: {pred_Sc[0]:.2f}')