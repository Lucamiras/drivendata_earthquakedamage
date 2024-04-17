import pandas as pd

def generate_predictions(model, X_test, file_path_and_name:str):
    predictions = model.predict(X_test.drop('building_id',axis=1))
    
    submission = pd.DataFrame()
    submission['building_id'] = X_test.building_id
    submission['damage_grade'] = predictions

    submission.to_csv(f'{file_path_and_name}.csv',index=False)