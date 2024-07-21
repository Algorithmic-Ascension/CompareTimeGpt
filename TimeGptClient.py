import pandas as pd
import requests
from pathlib import Path

bearer_token_path = "APIKey_TimeGPT.txt"


# GDP only
def generate_body_table(data):
    melted_df = data.reset_index().melt(
        id_vars=["Period", "Country"], var_name="Metric", value_name="Value"
    )
    melted_df["Metric_Country"] = melted_df["Metric"] + "_" + melted_df["Country"]
    return melted_df[["Metric_Country", "Period", "Value"]].values.tolist()


# TODO move calculate into separate python file, calculating test data from data files and json responses
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


class TimeGptClient:
    def forecast_multi_series(
        data_training,
        model,
        clean_ex_first,
        finetune_steps,
        finetune_loss,
        forecast_horizon,
    ):
        jsonbody_request_data = generate_body_table(data_training)
        try:
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
                        "data": jsonbody_request_data,
                    },
                },
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "authorization": f"Bearer {Path(bearer_token_path).read_text()}",
                },
            )
            if not response.ok:
                response.raise_for_status()
            jsonresp = response.json()["data"]["forecast"]["data"]

            return extract_json_data(jsonresp)
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            print(f"Response content: {err.response.content}")
            raise err
        except Exception as err:
            print(f"Other error occurred: {err}")
            raise err
