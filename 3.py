# ===============================
# Real Estate Valuation Dataset
# Linear Regression using Scikit-Learn
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------
# قراءة البيانات
# ---------------------------------------------------
file_path = r"C:\Users\haider\Downloads\haider\Real estate valuation data set.xlsx"

df = pd.read_excel(file_path)

print("First 5 rows:")
print(df.head())

# حذف العمود الأول (No) لأنه مجرد رقم تسلسلي
df = df.drop(columns=['No'])

# ---------------------------------------------------
# تحديد المتغيرات
# ---------------------------------------------------
X = df.iloc[:, :-1]     # Features
y = df.iloc[:, -1]      # Target (House price)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===================================================
# (a) تدريب نموذج الانحدار وحساب RMS
# ===================================================
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# حساب RMS
rms = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n(a) RMS Error on Test Data =")
print(rms)

# ===================================================
# (b) معاملات الانحدار قبل التطبيع
# ===================================================
print("\n(b) Regression Coefficients (Before Normalization):")

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coef_df)

# رسم المعاملات
plt.figure(figsize=(8,5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title('Coefficients Before Normalization')
plt.show()

# ===================================================
# (c) التطبيع إلى المجال [0,1]
# ===================================================
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# تقسيم البيانات بعد التطبيع
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# تدريب النموذج مرة أخرى
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train_scaled)

print("\n(c) Regression Coefficients (After Normalization):")

coef_scaled_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_scaled.coef_
})

print(coef_scaled_df)

# رسم المعاملات بعد التطبيع
plt.figure(figsize=(8,5))
sns.barplot(data=coef_scaled_df, x='Coefficient', y='Feature')
plt.title('Coefficients After Normalization')
plt.show()

# ===================================================
# (d) تحليل البواقي Residuals
# ===================================================

# التنبؤ باستخدام النموذج المطبّع
y_pred_scaled = model_scaled.predict(X_test_scaled)

# حساب البواقي
residuals = y_test_scaled - y_pred_scaled

print("\n(d) Residual Statistics:")
print(residuals.describe())

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=20, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.show()

# Q-Q Plot
import scipy.stats as stats

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()