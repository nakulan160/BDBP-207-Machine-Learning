import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class BaggingRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.models = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=5,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = BaggingRegressorScratch(n_estimators=20, max_depth=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))