#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)

    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_input_pattern = 's3://nyc-duration/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_input_pattern)

    return input_pattern.format(year=year, month=month)

def read_data(path) -> pd.DataFrame:

    options = (
        {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
    )
    df = pd.read_parquet(path, storage_options=options)

    return df

def save_data(df: pd.DataFrame, output_file: str) -> None:

    options = {
        "client_kwargs": {
            "endpoint_url": s3_endpoint_url
        }
    }
    df.to_parquet(
        output_file,
        engine="pyarrow",
        index=False,
        storage_options=options
    )

    return df

def prepare_data(df, categorical):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df_raw = read_data(input_file)
    df = prepare_data(df_raw, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f'Predicted mean duration for {year}-{month:02d}: {y_pred.mean():.2f} minutes')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)

    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)