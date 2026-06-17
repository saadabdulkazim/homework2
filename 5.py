import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==================================================
# توليد البيانات
# ==================================================

x = np.arange(0, 20.1, 0.1)

np.random.seed(0)

y = (
      1 * x**5
    + 3 * x**4
    - 100 * x**3
    + 8 * x**2
    - 300 * x
    - 1e5
    + np.random.randn(len(x)) * 1e5
)

# ==================================================
# دالة إنشاء مصفوفة الخصائص متعددة الحدود
# ==================================================

def polynomial_features(x, p):

    X = np.ones((len(x), p + 1))

    for i in range(1, p + 1):
        X[:, i] = x ** i

    return X


# ==================================================
# Polynomial Regression with p = 5
# ==================================================

p = 5

X5 = polynomial_features(x, p)

model5 = LinearRegression(fit_intercept=False)

model5.fit(X5, y)

theta5 = model5.coef_

print("Coefficients for p = 5")
for i in range(len(theta5)):
    print(f"theta{i} = {theta5[i]}")


# ==================================================
# Polynomial Regression with p = 4
# ==================================================

p = 4

X4 = polynomial_features(x, p)

model4 = LinearRegression(fit_intercept=False)

model4.fit(X4, y)

theta4 = model4.coef_

print("\nCoefficients for p = 4")
for i in range(len(theta4)):
    print(f"theta{i} = {theta4[i]}")


# ==================================================
# رسم النتائج
# ==================================================

y_pred5 = model5.predict(X5)

y_pred4 = model4.predict(X4)

plt.figure(figsize=(10,6))

plt.scatter(x, y, s=15, label="Data")

plt.plot(x, y_pred5, linewidth=3, label="Degree 5")

plt.plot(x, y_pred4, linewidth=3, label="Degree 4")

plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.show()