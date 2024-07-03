# Standard library imports
import argparse
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

# Define the directory paths
CODING_DIRECTORY = Path(__file__).parent
HOME_DIRECTORY = Path("/onur9")  # Path.home()  # BU EN SONDA DEĞİŞECEK
BENCHMARK_DIRECTORY = HOME_DIRECTORY / "MRC" / "MRC - MI9050_Various_Benchmark"
MAIN_DIRECTORY = BENCHMARK_DIRECTORY / "Main_Directory"
WORKING_DIRECTORY = MAIN_DIRECTORY / "Alınan Veriler/2023/2.Dönem"
TEMPLATE_DIRECTORY = MAIN_DIRECTORY / "Sunum_Şablonları"
GRAPHICS_DIRECTORY = MAIN_DIRECTORY / "Grafikler"

# Define the file paths
MASTER_FILE = WORKING_DIRECTORY / "2023_YILLIK_MASTER_DOSYA.xlsx"
YARIYIL_TEMPLATE_FILE = TEMPLATE_DIRECTORY / "Kıyaslama_Çalışması_Yarıyıl_Raporu_Template.pptx"
YILSONU_TEMPLATE_FILE = TEMPLATE_DIRECTORY / "Kıyaslama_Çalışması_Yıl_Sonu_Raporu_Template.pptx"

COMPANY_GROUPS = {
    "AYDEM": ["ADM EDAŞ", "GDZ EDAŞ"],
    "ENERJİSA": ["BAŞKENT EDAŞ", "AYEDAŞ", "TOROSLAR EDAŞ"],
    "AKSA": ["FIRAT EDAŞ", "ÇORUH EDAŞ"],
    "SEDAŞ": ["SEDAŞ"],
    "YEDAŞ": ["YEDAŞ"],
    "TREDAŞ": ["TREDAŞ"],
    "VEDAŞ": ["VEDAŞ"],
    "UEDAŞ": ["UEDAŞ"],
    "ARAS EDAŞ": ["ARAS EDAŞ"],
    "MRC": [],
}
NUM_OF_COMPANIES = sum(len(companies) for companies in COMPANY_GROUPS.values())

# Değişebilir verilerin alınması(Şirket adı, yıl, dosya adı)
YEAR = "2023_v1"    # "2018","2019","2020-v1","2020-v2","2021-v1","2021-v2","2022-v1","2022-v2"

SIGMA: int = 2
START_COL = 3  # 4th column (index starts from 0)
END_COL = START_COL + NUM_OF_COMPANIES

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create powerpoint files based on benchmark data')
    parser.add_argument('year', type=int, help='Year of the data (e.g. 2023)')
    parser.add_argument(
        'type', type=str, choices=['yariyillik', 'yillik'],
        help='Type of the data (must be "yariyillik" or "yillik")'
    )
    return parser.parse_args()


def filtered_mean(row) -> int:
    """
    Calculate the mean of each row within a specified number of standard deviations
    and add it to a new column named 'filtered_mean'.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    start_col (int): The starting column index for the range of columns to consider.
    end_col (int): The ending column index (exclusive) for the range of columns to consider.
    sigma (int, optional): The number of standard deviations to include. Default is 2.

    Returns:
    pandas.DataFrame: The DataFrame with the new column 'filtered_mean'.
    """
    # Extract the relevant columns
    data = row[START_COL:END_COL]

    # Compute the mean and standard deviation
    mean = data.mean()
    std = data.std()

    # Define the range within the specified number of standard deviations
    lower_bound = mean - SIGMA * std
    upper_bound = mean + SIGMA * std

    # Filter the data to include only values within the range
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # Compute the mean of the filtered values
    return filtered_data.mean()



def format_percentage(value: float, decimal_digits: int = 2) -> str:
    """
    Format a float value as a percentage with 2 decimal points and the percent sign in front.

    Parameters:
    value (float): The float value to be formatted as a percentage.

    Returns:
    str: The formatted percentage string.
    """
    return "{:.{}f}%".format(value * 100, decimal_digits).replace('.', ',')


def shuffle_columns(df, company_list):
    """
    Shuffle the columns of a DataFrame, excluding the first three and the last one.
    Place the company_list columns at the beginning of the shuffle range.
    """
    # Fixed columns are the first three and the last one
    fixed_columns = df.columns[:START_COL].tolist() + df.columns[END_COL:].tolist()

    # Columns to be shuffled, excluding the company_list columns
    columns_to_shuffle = [col for col in df.columns[START_COL:END_COL] if col not in company_list]

    # Shuffle the columns
    shuffled_columns = np.random.permutation(columns_to_shuffle)

    # New column order: first the fixed columns, then the company_list, then the shuffled columns
    new_column_order = fixed_columns[:START_COL] + company_list + shuffled_columns.tolist() + fixed_columns[START_COL:]

    return df[new_column_order]


# This line ensures that the code is only run when the script is executed directly, not when it is imported as a module.
if __name__ == "__main__":
    args = parse_arguments()
    report_year = args.year
    report_type = args.type

    # Read the data from the Excel file and divide it into two DataFrames for the data and the layout
    dataframe_dict = pd.read_excel(MASTER_FILE, sheet_name=["2023_Total_Veriler", "pptx_layout"])
    df = dataframe_dict["2023_Total_Veriler"]
    pptx_layout = dataframe_dict["pptx_layout"]

    # Add Category from APG No
    df['Category No'] = df['APG No'].str.split('.').str[0]

    for company_group, company_list in COMPANY_GROUPS.items():
        # Shuffle columns for each group and store in a list
        shuffled_groups = []
        for _, group in df.groupby('Category No'):
            shuffled_group = shuffle_columns(group, company_list)
            shuffled_group .columns = range(shuffled_group.shape[1])
            shuffled_groups.append(shuffled_group)

        # Merge the shuffled groups back together and reset the column names
        merged_df = pd.concat(shuffled_groups).reset_index(drop=True)
        merged_df.columns = range(merged_df.shape[1])

        # merged_df.to_excel(f"/desktop/{YEAR}_{company_group}_Shuffled.xlsx", index=False)

        # Apply the filtered_mean function to each row and create a new column
        merged_df['filtered_mean'] = merged_df.apply(filtered_mean, axis=1)
