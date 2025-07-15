import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
from prefect import flow, task, get_run_logger
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("nyc-taxi-prediction")

@task
def read_dataframe(filename: str):
    """Load and prepare data (Q3 and Q4)."""
    logger = get_run_logger()
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    logger.info(f"{len(df)} filtered records loaded.")
    return df

@task
def train_model(df: pd.DataFrame):
    logger = get_run_logger()

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info(f"Model intercept: {model.intercept_}")
    return dv, model, X_train

@task
def evaluate_and_log(dv, model, X_train):
    """Log to MLflow and get model size (Q6)."""
    logger = get_run_logger()

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, "model")
        model_size = len(pickle.dumps(model))
        logger.info(f"Model size: {model_size} bytes")
        mlflow.log_metric("model_size_bytes", model_size)
        run_id = mlflow.active_run().info.run_id
        return run_id

@flow(name="NYC Taxi Training Pipeline")
def main_flow(data_file: str = "./data/yellow_tripdata_2023-03.parquet"):
    logger = get_run_logger()

    df_raw = pd.read_parquet(data_file)
    logger.info(f"{len(df_raw)} initial records loaded")

    df = read_dataframe(data_file)
    dv, model, X_train = train_model(df)
    run_id = evaluate_and_log(dv, model, X_train)
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "nyc-taxi-regressor")
    logger.info("Model registered.")

if __name__ == '__main__':
    main_flow()