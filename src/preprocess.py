import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_and_save_date(df):
    df = df.dropna()
    df.drop_duplicates()

    drop_cols = ['cb_person_default_on_file','loan_percent_income','loan_grade']
    df_cleaned = df.drop(columns = drop_cols, errors='ignore')

    df_cleaned.to_csv("data/processed/cleaned_data.csv", index=False)

    return df_cleaned


if __name__ == "__main__":
    df = load_data("data/raw/credit_risk_dataset.csv")
    df_cleaned = clean_and_save_date(df)
    df_cleaned.head()