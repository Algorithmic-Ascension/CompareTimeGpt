from TimeGptClient import TimeGptClient
import numpy as np
import pandas as pd

# python .\main.py -m .\TimeGptClient.py

model = "timegpt-1"
# model = 'timegpt-1-long-horizon'
# model = 'long-horizon'
# model = 'short-horizon'
# model = 'azureai'
clean_ex_first = True
# finetune_loss = 'default'
# finetune_loss = 'mae'
# finetune_loss = "mse"
finetune_loss = "rmse"
# finetune_loss = 'mape'
# finetune_loss = 'smape'

selected_countries = [
    "Australia",
    "Austria",
    "Belgium",
    "Canada",
    "Denmark",
    "Finland",
    "France",
    "G7",
    "Italy",
    "Japan",
    "Korea",
    "Luxembourg",
    "Mexico",
    "Netherlands",
    "Norway",
    "Portugal",
    "Spain",
    "Sweden",
    "United Kingdom",
    "United States",
]

filter_missing_data = lambda df: df[
    (df["Period"] > "2000-Q0") & (df["Period"] < "2024-Q0")
]


def split_data(df, forecast_horizon):
    training_list = []
    test_list = []
    grouped = df.sort_values(by=["Country", "Period"]).groupby("Country")
    for _, group in grouped:
        training_list.append(group.iloc[:-forecast_horizon])
        test_list.append(group.iloc[-forecast_horizon:])
    df_training = pd.concat(training_list)
    df_test = pd.concat(test_list)
    return (
        df_training.pivot_table(
            index=["Period", "Country"],
            values=["Consumption", "Unemployment", "GDP"],
        ),
        df_test.pivot_table(
            index=["Period", "Country"],
            values=["Consumption", "Unemployment", "GDP"],
        ),
    )


def load_data(selected_countries):
    consumption = pd.read_csv("./consumption.csv")[
        ["Period"] + selected_countries
    ].melt(id_vars=["Period"], var_name="Country", value_name="Consumption")
    unemployment = pd.read_csv("./unemployment.csv")[
        ["Period"] + selected_countries
    ].melt(id_vars=["Period"], var_name="Country", value_name="Unemployment")
    gdp = pd.read_csv("./gdp.csv")[["Period"] + selected_countries].melt(
        id_vars=["Period"], var_name="Country", value_name="GDP"
    )
    df = pd.merge(
        pd.merge(consumption, unemployment, on=["Period", "Country"], how="inner"),
        gdp,
        on=["Period", "Country"],
        how="inner",
    )
    return filter_missing_data(df)


data_full = load_data(selected_countries)

# finetune_loss mse
# RMSE for forecast_horizon 1 finetune mse 43: 0.030962689630510458
# RMSE for forecast_horizon 2 finetune mse 34: 0.03646148353703208
# RMSE for forecast_horizon 3 finetune mse 31: 0.07123146607394658
# RMSE for forecast_horizon 4 finetune mse 00: 0.05753704201756719
# RMSE for forecast_horizon 5 finetune mse 08: 0.04281400348816353
# RMSE for forecast_horizon 6 finetune mse 02: 0.047832236540700415

# finetune_loss rmse
# RMSE for forecast_horizon 1 finetune rmse 39: 0.03094369411200133
# RMSE for forecast_horizon 2 finetune rmse 34: 0.03647808932147662
# RMSE for forecast_horizon 3 finetune rmse 25: 0.0707391428871976
# RMSE for forecast_horizon 4 finetune rmse 00: 0.05753704391472815
# RMSE for forecast_horizon 5 finetune rmse 10: 0.04284289681635131
# RMSE for forecast_horizon 6 finetune rmse 01: 0.04787052631138558

for forecast_horizon in [1, 2, 3, 4, 5, 6]:
    print()
    data_training, data_test = split_data(data_full, forecast_horizon)

    finetune_cache = {}
    for finetune_steps in range(0, 51, 5):
        df_predicted = TimeGptClient.forecast_multi_series(
            data_training,
            model,
            clean_ex_first,
            finetune_steps,
            finetune_loss,
            forecast_horizon,
        )
        rmse = np.sqrt(((df_predicted - data_test) ** 2).mean())
        finetune_cache[finetune_steps] = rmse.GDP
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {rmse.GDP}"
        )
    approximate_steps = min(finetune_cache, key=finetune_cache.get)
    for finetune_steps in range(max(0, approximate_steps - 5), approximate_steps + 5):
        df_predicted = TimeGptClient.forecast_multi_series(
            data_training,
            model,
            clean_ex_first,
            finetune_steps,
            finetune_loss,
            forecast_horizon,
        )
        rmse = np.sqrt(((df_predicted - data_test) ** 2).mean())
        finetune_cache[finetune_steps] = rmse.GDP
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {rmse.GDP}"
        )

    finetune_for_smallest_rmse = min(finetune_cache, key=finetune_cache.get)
    smallest_rmse = finetune_cache[finetune_for_smallest_rmse]
    print(
        f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_for_smallest_rmse}: {smallest_rmse}"
    )
