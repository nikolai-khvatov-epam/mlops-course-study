import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
import numpy as np  # Added for rstate if needed


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_optimization(num_trials=15):
    # Set the tracking URI to connect to your local server (this ensures writing to the server)
    mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Adjust port if you used --port 5001, etc.

    # Set the experiment (runs will appear here in the UI)
    mlflow.set_experiment("random-forest-hyperopt-new")

    X_train, y_train = load_pickle("./output/train.pkl")
    X_val, y_val = load_pickle("./output/val.pkl")

    def objective(params):
        # Start a Nested run for each trial (logs to the server)
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Log validation RMSE
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    # Run optimization
    rstate = np.random.default_rng(42)  # For reproducibility
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()