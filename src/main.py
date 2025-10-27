from preprocess import load_data, clean_and_save_date
from featuring import create_preprocessing_pipeline
from model_train import train_model
import os

def main():
    df = load_data("data/raw/credit_risk_dataset.csv")
    df = clean_and_save_date(df)
    create_preprocessing_pipeline(df, target_column='loan_status')
    best_model = train_model(df, target_column='loan_status')
    print("Model training complete and best model saved.")
    return best_model

if __name__ == "__main__":
    main()