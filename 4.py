import numpy as np
import pandas as pd
import time
import torch
import autograd.numpy as anp
from autograd import grad

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# قراءة البيانات
# ==========================================================

file_path = r"C:\Users\haider\Downloads\haider\Real estate valuation data set.xlsx"

df = pd.read_excel(file_path)

# حذف الصفوف التي تحتوي على قيم مفقودة
df = df.dropna()

# حذف العمود No
df = df.drop(columns=['No'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ==========================================================
# تطبيع البيانات بين 0 و 1
# ==========================================================

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# RMS
# ==========================================================

def RMS(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ==========================================================
# (a) Normal Equation
# ==========================================================

def normalEquationRegression(X, y):

    X_bias = np.c_[np.ones(X.shape[0]), X]

    # استعمال pseudo inverse
    theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    return theta


# ==========================================================
# (b) Manual Gradient Descent
# ==========================================================

def gradientDescentRegression(X,
                              y,
                              alpha=0.01,
                              iterations=10000):

    m, d = X.shape

    X_bias = np.c_[np.ones(m), X]

    theta = np.zeros(d + 1)

    for i in range(iterations):

        y_pred = X_bias @ theta

        error = y_pred - y

        gradient = (1/m) * (X_bias.T @ error)

        theta -= alpha * gradient

    return theta


# ==========================================================
# (c) Autograd
# ==========================================================

def gradientDescentAutogradRegression(X,
                                      y,
                                      alpha=0.01,
                                      iterations=10000):

    m, d = X.shape

    X_bias = anp.c_[anp.ones(m), X]

    theta = anp.zeros(d + 1)

    def cost(theta):

        y_pred = anp.dot(X_bias, theta)

        return anp.mean((y_pred - y) ** 2)

    gradient_function = grad(cost)

    for i in range(iterations):

        theta -= alpha * gradient_function(theta)

    return np.array(theta)


# ==========================================================
# (d) PyTorch
# ==========================================================

def gradientDescentPyTorchRegression(X,
                                     y,
                                     alpha=0.01,
                                     iterations=10000):

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    m, d = X.shape

    ones = torch.ones((m, 1))

    X_bias = torch.cat((ones, X), dim=1)

    theta = torch.zeros(d + 1,
                        dtype=torch.float32,
                        requires_grad=True)

    for i in range(iterations):

        y_pred = X_bias @ theta

        loss = torch.mean((y_pred - y) ** 2)

        loss.backward()

        with torch.no_grad():
            theta -= alpha * theta.grad

        theta.grad.zero_()

    return theta.detach().numpy()


# ==========================================================
# المقارنة
# ==========================================================

results = []

X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]


# Normal Equation
start = time.time()

theta_NE = normalEquationRegression(X_train, y_train)

y_pred_NE = X_test_bias @ theta_NE

end = time.time()

results.append([
    "Normal Equation",
    RMS(y_test, y_pred_NE),
    end - start
])


# Manual GD
start = time.time()

theta_GD = gradientDescentRegression(X_train, y_train)

y_pred_GD = X_test_bias @ theta_GD

end = time.time()

results.append([
    "Gradient Descent",
    RMS(y_test, y_pred_GD),
    end - start
])


# Autograd
start = time.time()

theta_AG = gradientDescentAutogradRegression(X_train, y_train)

y_pred_AG = X_test_bias @ theta_AG

end = time.time()

results.append([
    "Autograd GD",
    RMS(y_test, y_pred_AG),
    end - start
])


# PyTorch
start = time.time()

theta_PT = gradientDescentPyTorchRegression(X_train, y_train)

y_pred_PT = X_test_bias @ theta_PT

end = time.time()

results.append([
    "PyTorch GD",
    RMS(y_test, y_pred_PT),
    end - start
])


# Scikit-Learn
start = time.time()

model = LinearRegression()

model.fit(X_train, y_train)

y_pred_SK = model.predict(X_test)

end = time.time()

results.append([
    "Scikit-Learn",
    RMS(y_test, y_pred_SK),
    end - start
])


# ==========================================================
# النتائج
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Method",
        "RMS Error",
        "Execution Time (sec)"
    ]
)

print("\nComparison of Methods:\n")
print(results_df)