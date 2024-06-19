import json
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime
from pathlib import Path

# TODO move paramters into Powershell script
model = "timegpt-1"
# model = 'timegpt-1-long-horizon'
# model = 'long-horizon'
# model = 'short-horizon'
# model = 'azureai'
clean_ex_first = True
finetune_steps = 0
# finetune_loss = 'default'
# finetune_loss = 'mae'
finetune_loss = "mse"
# finetune_loss = 'rmse'
# finetune_loss = 'mape'
# finetune_loss = 'smape'

forecast_horizon = 2
bearer_token_path = "APIKey_TimeGPT.txt"

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
    # 'United States', 'Japan'
    # , 'Canada', 'Australia', 'United Kingdom', 'Portugal'
    # 'Netherlands', 'Sweden', 'Belgium', 'France', 'Italy', 'Denmark', 'Luxembourg',
    # 'Spain', 'Mexico', 'Finland', 'Norway', 'Korea', 'Germany', 'Austria', 'Greece',
    # 'Iceland', 'Switzerland', 'New Zealand', 'Turkey', 'Chile', 'Czechia',
    # 'Slovak Republic', 'Israel', 'Hungary', 'Slovenia', 'Poland', 'Estonia',
    # 'Latvia', 'Lithuania', 'Costa Rica', 'Colombia'
]

filter_missing_data = lambda df: df[
    (df["Period"] > "1993-Q0") & (df["Period"] < "2024-Q0")
]


def split_data(df):
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
            index=["Period", "Country"], values=["Consumption", "Unemployment", "GDP"]
        ),
        df_test.pivot_table(
            index=["Period", "Country"], values=["Consumption", "Unemployment", "GDP"]
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


def extract_json_data(response_data):
    convert_period = lambda s: (
        str(int(s[:4]) + 1) + "-Q1"
        if s[5:] == "12-31"
        else (
            s[:5] + "Q2"
            if s[5:] == "03-31"
            else (
                s[:5] + "Q3"
                if s[5:] == "06-30"
                else s[:5] + "Q4" if s[5:] == "09-30" else s[5:]
            )
        )
    )
    df = pd.DataFrame(response_data, columns=["Metric_Country", "Period", "Value"])
    df["Period"] = df["Period"].apply(convert_period)
    df[["Metric", "Country"]] = df["Metric_Country"].str.split("_", expand=True)
    df_pivot = (
        df.drop(columns=["Metric_Country"])
        .pivot_table(index=["Period", "Country"], columns="Metric", values="Value")
        .reset_index()
    )
    df_pivot.columns.name = None
    return df_pivot.set_index(["Period", "Country"])


# GDP only
def generate_body_table(data):
    melted_df = data.reset_index().melt(
        id_vars=["Period", "Country"], var_name="Metric", value_name="Value"
    )
    melted_df["Metric_Country"] = melted_df["Metric"] + "_" + melted_df["Country"]
    return melted_df[["Metric_Country", "Period", "Value"]].values.tolist()


data_full = load_data(selected_countries)
# TODO repeat for experiments with different time series
data_training, data_test = split_data(data_full)
response = requests.post(
    "https://dashboard.nixtla.io/api/forecast_multi_series",
    json={
        "model": model,
        "freq": "Q",  # quarterly
        "fh": forecast_horizon,
        "clean_ex_first": clean_ex_first,
        "finetune_steps": finetune_steps,
        "finetune_loss": finetune_loss,
        "y": {
            "columns": ["unique_id", "ds", "y"],
            "data": generate_body_table(data_training),
        },
    },
    headers={
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {Path(bearer_token_path).read_text()}",
    },
)
jsonresp = response.json()["data"]["forecast"]["data"]

# TODO save requests as well
filename = f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{model}_{finetune_loss}_{finetune_steps}_{clean_ex_first}.txt"
if not os.path.exists("Responses"):
    os.makedirs("Responses")

with open(f"Responses/json_{filename}.json", "w+") as output_file_handler:
    output_file_handler.write(json.dumps(jsonresp, indent=4))

# TODO move calculate into separate python file, calculating test data from data files and json responses
df_predicted = extract_json_data(jsonresp)
# Calculate the MSE
rmse = np.sqrt(((df_predicted - data_test) ** 2).mean())

print("RMSE for each column:")
print(rmse)
