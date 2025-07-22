import pandas as pd
from datetime import datetime
from my_batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']

    df_output = prepare_data(df_input, categorical)

    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0)
    ]
    expected_columns = columns + ['duration']
    df_expected = pd.DataFrame(expected_data, columns=expected_columns)

    output_dicts = df_output.to_dict('records')
    expected_dicts = df_expected.to_dict('records')

    assert len(df_output) == 2
    assert output_dicts == expected_dicts
    print(f"List of rows after prepare_data: {(df_output)}")
    print(f"Number of rows after prepare_data: {len(df_output)}")