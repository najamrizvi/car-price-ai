import pandas as pd

def preprocess_data(df):
    df = df.drop(['car_ID', 'CarName'], axis=1)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded