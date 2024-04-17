import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder

def transform_pipeline(numerical_columns_to_encode:list, categorical_columns_to_encode:list):
    """
        Takes in a list of tuples containing names of estimators and the actual estimators, as well as columns to transform, and trains the different estimators.'
    """
    
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', OneHotEncoder(), numerical_columns_to_encode),
            ('cat', OneHotEncoder(), categorical_columns_to_encode)
        ], remainder='passthrough'
    )

    return preprocessor

def training_pipeline(estimators:list, preprocessor, X:pd.DataFrame, y:pd.DataFrame, cv:int=5) -> dict:
    
    pipelines = {}

    for name, classifier in estimators:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        pipe.fit(X,y)
        
        scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_micro')
        
        pipelines[name] = pipe
        
        print(f"{name}: Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    return pipelines
