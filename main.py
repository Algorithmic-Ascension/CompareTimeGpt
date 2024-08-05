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


def split_data(df, forecast_horizon, data_shift):
    training_list = []
    test_list = []
    grouped = df.sort_values(by=["Country", "Period"]).groupby("Country")
    for _, group in grouped:
        training_list.append(group.iloc[: -(forecast_horizon + data_shift)])
        test_list.append(
            group.iloc[len(group) - data_shift - 1 : len(group) - data_shift]
        )
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


def calculate_RMSE(data_shifts, finetune_steps):
    total_squared_error = 0
    total_count = 0
    for data_shift in range(data_shifts):
        data_training, data_test = split_data(data_full, forecast_horizon, data_shift)
        df_predicted = TimeGptClient.forecast_multi_series(
            data_training,
            model,
            clean_ex_first,
            finetune_steps,
            finetune_loss,
            forecast_horizon,
        )
        last_quarter = df_predicted.index.get_level_values("Period").max()
        errors_GDP = (df_predicted.loc[last_quarter] - data_test).GDP
        total_squared_error += (errors_GDP**2).sum()
        total_count += len(errors_GDP)
    return np.sqrt(total_squared_error / total_count)


data_full = load_data(selected_countries)

# finetune_loss mse

# finetune_loss rmse
# RMSE for forecast_horizon 1 finetune rmse 40: 0.03139025746377753


for forecast_horizon in [1, 2, 3, 4, 5, 6]:
    print()

    # argmin of average_rmse
    finetune_cache = {}
    for finetune_steps in range(0, 51, 5):
        average_rmse = calculate_RMSE(2, finetune_steps)
        finetune_cache[finetune_steps] = average_rmse
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {average_rmse}"
        )
    approximate_steps = min(finetune_cache, key=finetune_cache.get)
    for finetune_steps in range(max(0, approximate_steps - 5), approximate_steps + 5):
        average_rmse = calculate_RMSE(2, finetune_steps)
        finetune_cache[finetune_steps] = average_rmse
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {average_rmse}"
        )

    finetune_for_smallest_rmse = min(finetune_cache, key=finetune_cache.get)
    smallest_rmse = finetune_cache[finetune_for_smallest_rmse]
    print(
        f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_for_smallest_rmse}: {smallest_rmse}"
    )
