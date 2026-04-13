import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score,mean_squared_error

def main():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X=data.iloc[:,:-2]
    y=data["disease_score"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    model = DecisionTreeRegressor(random_state=1607, max_depth=15)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2= r2_score(y_test,y_pred)
    print(r2)
    mse = mean_squared_error(y_test,y_pred)
    print(mse)
    plot_tree(model, feature_names=X.columns, filled=True, rounded=True,fontsize=18)
    plt.show()
    



if __name__ == '__main__':
    main()
