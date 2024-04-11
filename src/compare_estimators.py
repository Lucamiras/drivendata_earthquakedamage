from sklearn.model_selection import cross_val_score

def compare_estimators(estimators:tuple, X, y):
    # Iterate over estimators and perform cross-validation
    for name, estimator in estimators:
        scores = cross_val_score(estimator, X, y, cv=5)
        print(f"{name}: Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
