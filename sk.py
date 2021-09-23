import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x = diabetes.data[:]
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_predict = model.predict(diabetes_x_test)

print("The Mean Squared Error Value is: ", mean_squared_error(diabetes_y_test, diabetes_y_predict))
print("Weights: ", model.coef_)
print("Intercepts: ", model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predict)
plt.show()

# The Mean Squared Error Value is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercepts:  153.39713623331698

# The Mean Squared Error Value is:  1826.5364191345425
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercepts:  153.05827988224112
