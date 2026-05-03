import pandas as pd


def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def get_basic_info(df):
    """
    Extract basic dataset information.
    """
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return info


def get_sample_rows(df, n=5):
    """
    Get sample rows from dataset.
    """
    return df.head(n).to_dict(orient="records")


def summarize_dataset(df):
    """
    Combine all dataset information into a summary.
    """
    summary = {
        "basic_info": get_basic_info(df),
        "sample_data": get_sample_rows(df)
    }
    return summary