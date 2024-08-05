from TimeGptClient import TimeGptClient
import numpy as np
import pandas as pd
import time

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

# Cameron selected countries
selected_countries = [
    "Australia",
    "Austria",
    "Belgium",
    "Canada",
    "Chile",
    "Czechia",
    "Denmark",
    "Estonia",
    "Euro area (20 countries)",
    "European Union (27 countries from 01/02/2020)",
    "Finland",
    "France",
    "G7",
    "Greece",
    "Hungary",
    "Israel",
    "Italy",
    "Japan",
    "Korea",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Mexico",
    "Netherlands",
    "Norway",
    "Poland",
    "Portugal",
    "Slovak Republic",
    "Slovenia",
    "Spain",
    "Sweden",
    "United Kingdom",
    "United States",
]


def split_data(df, forecast_horizon, data_shift):
    training_list = []
    test_list = []
    for _, group in df.groupby("Country"):
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
    def filter_missing_data_from_countries(df):
        print("Determining countries...")
        missing_data_countries = (
            df[
                (df["Consumption"] == -1)
                | (df["Consumption"] == -100)
                | (df["Unemployment"] == -1)
                | (df["Unemployment"] == -100)
                | (df["GDP"] == -1)
                | (df["GDP"] == -100)
            ]
            .index.get_level_values("Country")
            .unique()
        )
        df = df[~df.index.get_level_values("Country").isin(missing_data_countries)]

        print("\nMissing countries:")
        for country in missing_data_countries:
            print(country + ",")

        print("\nIncluded countries:")
        for country in df.index.get_level_values("Country").unique():
            print(country + ",")
        return df

    consumption = pd.read_csv("./consumption.csv")
    unemployment = pd.read_csv("./unemployment.csv")
    gdp = pd.read_csv("./gdp.csv")
    if selected_countries is not None:
        subselection = ["Period"] + selected_countries
        consumption = consumption[subselection]
        unemployment = unemployment[subselection]
        gdp = gdp[subselection]
    consumption = consumption.melt(
        id_vars=["Period"], var_name="Country", value_name="Consumption"
    )
    unemployment = unemployment.melt(
        id_vars=["Period"], var_name="Country", value_name="Unemployment"
    )
    gdp = gdp.melt(id_vars=["Period"], var_name="Country", value_name="GDP")
    df = pd.merge(
        pd.merge(consumption, unemployment, on=["Period", "Country"], how="inner"),
        gdp,
        on=["Period", "Country"],
        how="inner",
    )
    df.set_index(["Period", "Country"], inplace=True)
    df = df[["Consumption", "GDP", "Unemployment"]]
    df = df[(df.index.get_level_values("Period") > "2000-Q0")]  # missing data rows
    df = filter_missing_data_from_countries(df)  # missing data columns
    return df.sort_values(by=["Country", "Period"])


def calculate_RMSE(data_full, data_shifts, finetune_steps):
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


data_full = load_data(None)

# finetune_loss mse

# finetune_loss rmse
# RMSE for forecast_horizon 1 finetune rmse 35: 0.05957993940219445

# there is an optimization to be made to use all the data points from an API
# but that would slightly lower performance
for forecast_horizon in [
    1,
    2,
    3,
    4,
    5,
    6,
]:
    print()
    # TODO rate limiting needs work
    time.sleep(1)

    # argmin of average_rmse
    # grid search
    finetune_cache = {}
    for finetune_steps in range(0, 51, 5):
        average_rmse = calculate_RMSE(data_full, 2, finetune_steps)
        finetune_cache[finetune_steps] = average_rmse
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {average_rmse}"
        )
    approximate_steps = min(finetune_cache, key=finetune_cache.get)
    for finetune_steps in range(max(0, approximate_steps - 4), approximate_steps + 4):
        average_rmse = calculate_RMSE(data_full, 2, finetune_steps)
        finetune_cache[finetune_steps] = average_rmse
        print(
            f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_steps}: {average_rmse}"
        )

    finetune_for_smallest_rmse = min(finetune_cache, key=finetune_cache.get)
    smallest_rmse = finetune_cache[finetune_for_smallest_rmse]
    print(
        f"RMSE for forecast_horizon {forecast_horizon} finetune {finetune_loss} {finetune_for_smallest_rmse}: {smallest_rmse}"
    )
