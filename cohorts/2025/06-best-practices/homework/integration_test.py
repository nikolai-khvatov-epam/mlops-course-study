# integration_test.py

import pandas as pd
from datetime import datetime
import os
import subprocess

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

year = 2023
month = 1

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
input_pattern = os.getenv("INPUT_FILE_PATTERN")
output_pattern = os.getenv("OUTPUT_FILE_PATTERN")

input_file = input_pattern.format(year=year, month=month)
output_file = output_pattern.format(year=year, month=month)

options = {
    "client_kwargs": {
        "endpoint_url": S3_ENDPOINT_URL
    }
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)
print(f"Uploaded input file to {input_file}")

exit_code = os.system(f"python my_batch.py {year} {month}")
assert exit_code == 0

df_output = pd.read_parquet(output_file, storage_options=options)

total_duration = round(df_output['predicted_duration'].sum(), 2)
print(f" Sum of predictions(durations): {total_duration}")
