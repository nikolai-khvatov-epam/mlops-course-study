import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from prefect import flow, task, get_run_logger
import datetime as dt
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnQuantileMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric
)

@task
def read_dataframe(filename: str):
    logger = get_run_logger()

    df = pd.read_parquet(filename)
    print(f"✅ Data details : {df.describe()}")
    print(f"✅ Data shape: {df.shape}")

    df["duration_min"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration_min = df.duration_min.apply(lambda td: float(td.total_seconds()) / 60)

    df = df[(df.duration_min >= 0) & (df.duration_min <= 60)]
    df = df[(df.passenger_count > 0) & (df.passenger_count <= 8)]

    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    return df

@task
def train_model(df: pd.DataFrame):
    logger = get_run_logger()

    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    features = num_features + cat_features
    print(f"✅ Filtered shape: {df.shape}")

    train_data = df[:30000]
    val_data = df[30000:]

    model = LinearRegression()
    model.fit(train_data[features], train_data[target])

    train_preds = model.predict(train_data[features])
    val_preds = model.predict(val_data[features])

    train_data = train_data.copy()
    val_data = val_data.copy()
    train_data['prediction'] = train_preds
    val_data['prediction'] = val_preds

    rmse_train = root_mean_squared_error(train_data.duration_min, train_data.prediction)
    rmse_val = root_mean_squared_error(val_data.duration_min, val_data.prediction)

    logger.info(f"Model intercept: {model.intercept_}")
    logger.info(f"Train RMSE: {rmse_train:.2f}")
    logger.info(f"Validation RMSE: {rmse_val:.2f}")

    return model, train_data, val_data, num_features, cat_features

@task
def calculate_daily_fare_amount_median(df: pd.DataFrame):
    logger = get_run_logger()

    column_mapping = ColumnMapping(
        numerical_features=["fare_amount"],
        categorical_features=[],
        target=None,
        prediction=None,
    )

    start = dt.date(2024, 3, 1)
    end = dt.date(2024, 3, 31)

    daily_results = []

    for n in range((end - start).days + 1):
        day = start + dt.timedelta(days=n)
        nxt = day + dt.timedelta(days=1)
        day_ts = pd.Timestamp(day)
        nxt_ts = pd.Timestamp(nxt)
        day_data = df.loc[
            df["lpep_pickup_datetime"].between(day_ts, nxt_ts, inclusive="left")
        ]
        if day_data.empty or day_data["fare_amount"].dropna().empty:
            continue
        report = Report(metrics=[
            ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)
        ])
        report.run(reference_data=None, current_data=day_data, column_mapping=column_mapping)
        metric_result = report.as_dict()["metrics"][0]["result"]

        median_val    = metric_result.get("current", {}).get("value", None)

        if median_val is not None:
            daily_results.append((day, float(median_val)))

    daily_df = pd.DataFrame(daily_results, columns=["date", "median_fare"])

    max_row = daily_df.loc[daily_df["median_fare"].idxmax()]
    print(f"Maximum daily median fare_amount from parquet source: "
        f"{max_row.median_fare:.2f} on {max_row.date}")

    logger.info(f"📈 Max median fare_amount: {max_row.median_fare:.2f} on {max_row.date}")

    return max_row.median_fare, max_row.date

@task
def evidently_report(train_data, val_data, num_features, cat_features):
    logger = get_run_logger()

    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        DataQualityPreset()
    ])

    report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)

    report.save_html("reports/evidently_report.html")

    result = report.as_dict()

    logger.info(f"Prediction drift score: {result['metrics'][0]['result']['drift_score']}")
    logger.info(f"Number of drifted columns: {result['metrics'][1]['result']['number_of_drifted_columns']}")
    logger.info(f"Share of missing values: {result['metrics'][2]['result']['current']['share_of_missing_values']}")

    return result

@task
def prep_db():
    dbname = "test"
    user = "postgres"
    password = "example"
    host = "localhost"
    port = 5432

    conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE {dbname}')
        print(f"Database '{dbname}' created")
    else:
        print(f"Database '{dbname}' already exists")
    cur.close()
    conn.close()

    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS dummy_metrics (
        timestamp TIMESTAMP,
        prediction_drift FLOAT,
        num_drifted_columns INT,
        share_missing_values FLOAT,
        median_fare FLOAT
    )
    """
    cur.execute(create_table_sql)
    conn.commit()
    cur.close()
    conn.close()

@task
def save_metrics_to_postgres(val_data: pd.DataFrame, reference_data: pd.DataFrame):
    val_data['lpep_pickup_datetime'] = pd.to_datetime(val_data['lpep_pickup_datetime'])
    val_data['lpep_pickup_datetime'] = val_data['lpep_pickup_datetime'].dt.tz_localize(None)

    current_ts = pd.Timestamp.now().replace(microsecond=0)

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)
    ])

    report.run(
        reference_data=reference_data,
        current_data=val_data,
        column_mapping=ColumnMapping(
            prediction='prediction',
            numerical_features=['fare_amount', 'prediction']
        )
    )

    result = report.as_dict()

    pred_drift = float(result['metrics'][0]['result']['drift_score'])
    num_drifted = int(result['metrics'][1]['result']['number_of_drifted_columns'])
    share_missing = float(result['metrics'][2]['result']['current']['share_of_missing_values'])
    median_fare = float(result['metrics'][3]['result']['current']['value'])

    conn_params = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "example",
        "dbname": "test"
    }

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dummy_metrics (timestamp, prediction_drift, num_drifted_columns, share_missing_values, median_fare)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (current_ts, pred_drift, num_drifted, share_missing, median_fare)
            )
        conn.commit()
    print("Metrics inserted.")
    print(f"Metrics to insert: pred_drift={pred_drift}, num_drifted={num_drifted}, share_missing={share_missing}, median_fare={median_fare}")


@task
def save_artifacts(model, val_data):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with open("models/model.pkl", "wb") as f_out:
        pickle.dump(model, f_out)

    val_data.to_parquet("data/reference.parquet", index=False)

@flow(name="Green Taxi Monitoring with Evidently + PostgreSQL")
def main_flow(data_file: str = "./data/green_tripdata_2024-03.parquet"):
    df = read_dataframe(data_file)
    model, train_data, val_data, num_feat, cat_feat = train_model(df)
    save_artifacts(model, val_data)
    max_median, max_day = calculate_daily_fare_amount_median(df)
    metrics_report = evidently_report(train_data, val_data, num_feat, cat_feat)

    prep_db()
    save_metrics_to_postgres(val_data, reference_data=train_data)

    logger = get_run_logger()
    if max_day and max_median:
        logger.info(f"Max median fare_amount: {max_median:.2f} on {max_day}")
    logger.info(f"Evidently report snippet: {metrics_report}")

if __name__ == '__main__':
    main_flow()
