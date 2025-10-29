import pytest
import pandas as pd
from src.model_train import train_model

def test_train_model_output():
    """
    Test that train_model returns a fitted model.
    """
    df = pd.read_csv("data/processed/cleaned_credit_rick.csv")
    sample_df = df.sample(n=20, random_state=42)  

    model = train_model(sample_df, target_column='loan_status', use_smote=False)
    assert model is not None, "Model training failed, got None."
    assert hasattr(model, "predict"), "Trained model has no predict method."