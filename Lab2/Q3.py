import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.DataFrame({
    'Age': [20, 25, 30, 35, 40, 45],
    'Weight': [50, 55, 60, 65, 70, 75],
    'Sugar': [90, 100, 115, 130, 145, 160]
})
X=df[['Age','Weight']]
y=df['Sugar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
print(model.coef_)
print(model.intercept_)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
