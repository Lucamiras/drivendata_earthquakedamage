import pandas as pd

def encode_features(df):
    non_numerical_columns = ["land_surface_condition",
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "legal_ownership_status"]
    encoded_features = pd.get_dummies(df[non_numerical_columns])
    numerical_features = df.drop(non_numerical_columns, axis=1)
    new_df = pd.concat([encoded_features,numerical_features],axis=1)
    return new_df