from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

data=pd.read_csv('Boston.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=42)
model=XGBRegressor(learning_rate=0.01,n_estimators=100,max_depth=3)
model.fit(train_x,train_y)
y_pred=model.predict(test_x)
r2=r2_score(test_y,y_pred)
print("R2:",r2)