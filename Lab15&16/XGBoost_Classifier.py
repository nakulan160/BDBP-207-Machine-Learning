from xgboost import XGBClassifier, plot_importance
from ISLP import load_data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=load_data("Default")
X=data.iloc[:,1:]
X=X.drop(columns=['income'])
y=data.iloc[:,0]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1607)
cat_cols=X.select_dtypes(include=['object','category']).columns
num_cols=X.select_dtypes(exclude=['object','category']).columns
pp=ColumnTransformer(
    transformers=[
        ("nums",StandardScaler(),num_cols),
        ("cat",OneHotEncoder(),cat_cols)
    ]
)
X_train_scaled=pp.fit_transform(X_train)
X_test_scaled=pp.transform(X_test)
label=LabelEncoder()
y_train_scaled=label.fit_transform(y_train)
y_test_scaled=label.transform(y_test)
model=XGBClassifier()
model.fit(X_train_scaled,y_train_scaled)
y_pred=model.predict(X_test_scaled)
acc=accuracy_score(y_test_scaled,y_pred)
print("Accuracy:",acc)
plt.figure()
feature_names = pp.get_feature_names_out().tolist()
model.get_booster().feature_names = feature_names
plot_importance(model, importance_type='gain')
plt.show()


