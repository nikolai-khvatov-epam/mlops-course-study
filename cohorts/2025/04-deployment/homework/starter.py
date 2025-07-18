#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
import os
import click

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

@click.command()
@click.option('--year', default=2023, type=int, help='Year for data')
@click.option('--month', default=3, type=int, help='Month for data')
def main(year, month):
    input_file = f'./data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)

    dicts = df[categorical].to_dict(orient='records')
    numerical = ['trip_distance']
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['prediction'] = y_pred

    std_dev = df['prediction'].std()

    print(std_dev)

    mean_pred = df['prediction'].mean()
    print(f"Mean predicted duration: {mean_pred}")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = df[['ride_id', 'prediction']]
    output_file = f'predictions_yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(size_mb)

if __name__ == '__main__':
    main()