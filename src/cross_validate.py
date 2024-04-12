from sklearn.model_selection import cross_val_score, StratifiedKFold

def compare_estimators(estimators:tuple, X, y):
    
    #stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, estimator in estimators:
        scores = cross_val_score(estimator, X, y, cv=5, n_jobs=10, scoring='f1_micro')
        print(f"{name}: Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
