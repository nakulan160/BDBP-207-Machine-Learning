import numpy as np
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.models = []
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")
            for feature in range(n_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1
                        misclassified = w[y != predictions]
                        error = sum(misclassified)
                        if error > 0.5:
                            error = 1 - error
                            polarity *= -1
                        if error < min_error:
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_index = feature
                            min_error = error
            EPS = 1e-10
            stump.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))
            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)
            self.models.append(stump)
    def predict(self, X):
        model_preds = np.array([
            model.alpha * model.predict(X) for model in self.models
        ])
        y_pred = np.sum(model_preds, axis=0)
        return np.sign(y_pred)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = AdaBoost(n_estimators=20)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))