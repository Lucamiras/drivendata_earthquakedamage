from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def grid_search_best_estimator(model, param_grid:dict, X, y, cv:int, scoring:str, verbose:int):

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose)
    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Mean Cross-validated Score:", grid_search.best_score_)

    return grid_search.best_estimator_