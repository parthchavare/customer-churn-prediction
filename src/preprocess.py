import pandas as pd


def load_and_clean_data(file_path):
    """
    Load churn dataset and clean basic issues.
    """
    df = pd.read_csv(file_path)

    # Remove useless ID column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove rows with missing values
    df.dropna(inplace=True)

    return df