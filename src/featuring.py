import pandas as pd
import joblib

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(df, target_column):
    
    df = shuffle(df, random_state=42).reset_index(drop=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]


    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numeric_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int64']]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    joblib.dump(preprocessor, "models/preprocessing_pipeline.pkl")

    return preprocessor