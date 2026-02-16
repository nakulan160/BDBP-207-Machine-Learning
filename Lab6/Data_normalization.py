import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1607)
    return X_train, X_test, y_train, y_test

def normalize_data(X_train):
    for i in range(X_train.shape[1]):
        column = X_train.iloc[:, i]
        min_val=column.min()
        max_val=column.max()
        X_train.iloc[:,i]=(column-min_val)/(max_val-min_val)
    return X_train

def standardize_train(X_train):
    mean_list=[]
    std_list=[]
    for i in range(X_train.shape[1]):
        column =X_train.iloc[:, i]
        mea=column.mean()
        mean_list.append(mea)
        st=X_train.iloc[:,i].std()
        std_list.append(st)
        X_train.iloc[:,i]=(column-mea)/st
    return X_train,mean_list,std_list

def standardize_test(X_test,mean_list,std_list):
    for i in range(X_test.shape[1]):
        X_test.iloc[:,i]=(X_test.iloc[:,i]-mean_list[i])/std_list[i]
    return X_test


def main():
    data=pd.read_csv("Fishers maket.csv")
    X=data.iloc[:,2:]
    y=data.iloc[:,1]
    X_train, X_test, y_train, y_test = split_data(X,y)
    normalized_X_train=normalize_data(X_train)
    normalized_X_test=normalize_data(X_test)
    final_X_train,mean,std=standardize_train(normalized_X_train)
    final_X_test=standardize_test(normalized_X_test,mean,std)
    model=LinearRegression()
    model.fit(final_X_train,y_train)
    y_pred=model.predict(final_X_test)
    r2=r2_score(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    print("Linear Regression R^2:",r2)
    print("Linear Regression MSE:",mse)


if __name__=="__main__":
    main()