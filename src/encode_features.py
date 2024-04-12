import pandas as pd

def encode_categorical_features(df, non_numerical_columns):
    encoded_features = pd.get_dummies(df[non_numerical_columns])
    numerical_features = df.drop(non_numerical_columns, axis=1)
    new_df = pd.concat([encoded_features,numerical_features],axis=1)
    return new_df