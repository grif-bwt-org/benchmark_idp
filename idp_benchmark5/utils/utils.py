import pandas as pd
from datetime import datetime

def json_converter(o):
    if pd.isna(o):
        return None
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")