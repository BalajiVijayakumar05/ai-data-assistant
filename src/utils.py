import pandas as pd

def load_dataframe(path="../data/sample_data.csv"):
    return pd.read_csv(path)

def row_to_text(row):
    """Convert each row to natural language text for embeddings."""
    return (
        f"Customer {row['customer_id']} bought {row['product']} "
        f"for {row['amount']} in {row['city']}"
    )