import csv
import io
import pandas as pd

def read_pumping_data(file_obj) -> pd.DataFrame:
    """
    Reads CSV or Excel files, automatically detecting Brazilian CSV formats
    (';' delimiter and ',' decimal separator).
    """
    filename = getattr(file_obj, "name", "").lower()

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)

    # Read a sample to sniff delimiter
    content = file_obj.read()
    if isinstance(content, bytes):
        text_sample = content[:4096].decode("utf-8", errors="ignore")
        file_obj.seek(0)
    else:
        text_sample = content[:4096]

    # Delimiter detection
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=";,|\t")
        sep = dialect.delimiter
    except Exception:
        sep = ";" if ";" in text_sample else ","

    # Parse with standard decimal detection
    try:
        df = pd.read_csv(file_obj, sep=sep, decimal=",")
        # Validate if numeric columns loaded as objects (string decimal conversion issue)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, sep=sep, decimal=".")
        return df
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=",", decimal=".")
