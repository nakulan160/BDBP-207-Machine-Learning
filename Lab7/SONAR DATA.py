import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def main():
    data = pd.read_csv('sonar data.csv')
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1].map({"R":1,"M":0})
    kf = KFold(n_splits=10, shuffle=True)
    acc = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression()
        # model.fit(X_train_scaled, y_train)
        # y_pred = model.predict(X_test_scaled)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc.append(accuracy)
        print(f'Fold {fold+1}: {accuracy}')
    avg = sum(acc) / len(acc)
    print(f'Average accuracy score: {avg}')
if __name__ == "__main__":
    main()
