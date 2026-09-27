import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
data = pd.read_csv("data/advertising.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())


# 2. Data Preprocessing

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(data.shape)


# 3. Feature Selection

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]


# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Model Building

model = LinearRegression()

model.fit(X_train, y_train)


# 6. Prediction

y_pred = model.predict(X_test)


# 7. Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n----- Model Evaluation -----")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)


# 8. Save Model

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")