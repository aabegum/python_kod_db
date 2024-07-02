# Standard library imports
from pathlib import Path
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import Cm, Inches, Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE

# Configure seaborn
sns.set_style(style="whitegrid")
sns.set_palette("muted")

# Ignore warnings
warnings.filterwarnings("ignore")

HOME_DIRECTORY = Path("/onur9")  # Path.home()
BENCHMARK_DIRECTORY = HOME_DIRECTORY / "MRC" / "MRC - MI9050_Various_Benchmark"
MASTER_FILE = BENCHMARK_DIRECTORY / "New folder/Alınan Veriler/2023/2.Dönem/2023_YILLIK_MASTER_DOSYA.xlsx"

# Değişebilir verilerin alınması(Şirket adı, yıl, dosya adı)
YEAR = "2023_v1"    # "2018","2019","2020-v1","2020-v2","2021-v1","2021-v2","2022-v1","2022-v2"
COMPANY_NAME = "ENERJİSA"  # MRC,'ENERJİSA','AYDEM','SEDAŞ','YEDAŞ','TREDAŞ','UEDAŞ','ARAS EDAŞ','VEDAŞ'

def mean_within_sigma(df: pd.DataFrame, column_name: str, sigma: int = 2) -> float:
    """
    Calculate the mean of a DataFrame column within a specified number of standard deviations.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to compute the mean for.
    sigma (int, optional): The number of standard deviations to include. Default is 2.

    Returns:
    float: The mean of the values within the specified number of standard deviations.
    """
    # Compute the mean and standard deviation
    mean = df[column_name].mean()
    std = df[column_name].std()

    # Define the range within the specified number of standard deviations
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std

    # Filter the DataFrame to include only values within the range
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Compute the mean of the filtered values
    filtered_mean = filtered_df[column_name].mean()

    return filtered_mean

def format_percentage(value: float, decimal_digits: int = 2) -> str:
    """
    Format a float value as a percentage with 2 decimal points and the percent sign in front.

    Parameters:
    value (float): The float value to be formatted as a percentage.

    Returns:
    str: The formatted percentage string.
    """
    return "{:.{}f}%".format(value * 100, decimal_digits).replace('.', ',')

if __name__ == "__main__":
    # Run the code
    # if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:  # 17
    dataframe_dict = pd.read_excel(MASTER_FILE, sheet_name=["2023_Total_Veriler", "pptx_layout"])
    df = dataframe_dict["2023_Total_Veriler"]
    pptx_layout = dataframe_dict["pptx_layout"]
    df['Category No'] = df['APG No'].str.split('.').str[0]

