#! /usr/bin/env python

# Standard library imports
import argparse
import json
import logging
from pathlib import Path
import re
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the regular expression pattern to match illegal characters in Windows file paths
ILLEGAL_WINDOWS_PATH_CHARACTERS = re.compile(r'[\\/:*?"<>|]')

# Define the directory paths
CODING_DIRECTORY = Path(__file__).parent
HOME_DIRECTORY = Path.home()
BENCHMARK_DIRECTORY = HOME_DIRECTORY / "MRC" / "MRC - MI9050_Various_Benchmark"
MAIN_DIRECTORY = BENCHMARK_DIRECTORY / "Main_Directory"
WORKING_DIRECTORY = MAIN_DIRECTORY / "Alınan Veriler/2023/2.Dönem"
TEMPLATE_DIRECTORY = MAIN_DIRECTORY / "Sunum_Şablonları"
GRAPHICS_DIRECTORY = MAIN_DIRECTORY / "Grafikler"
REPORTS_DIRECTORY = MAIN_DIRECTORY / "Raporlar"

# Define the file paths
MASTER_FILE = WORKING_DIRECTORY / "2023_YILLIK_MASTER_DOSYA.xlsx"
YARIYIL_TEMPLATE_PATH = TEMPLATE_DIRECTORY / "Kıyaslama_Çalışması_Yarıyıl_Raporu_Template.pptx"
YILSONU_TEMPLATE_PATH = TEMPLATE_DIRECTORY / "Kıyaslama_Çalışması_Yıl_Sonu_Raporu_Template.pptx"
COMPANY_GROUPS_PATH = CODING_DIRECTORY / "company_groups.json"
PRESENTATION_INTRO_TEMPLATE_PATH = CODING_DIRECTORY / "presentation_intro_template.txt"

with COMPANY_GROUPS_PATH.open() as company_groups_file:
    COMPANY_GROUPS = json.load(company_groups_file)

NUM_OF_COMPANIES = sum(len(companies) for companies in COMPANY_GROUPS.values())
COMPANIES_RANGE = np.arange(1, NUM_OF_COMPANIES + 1)

REPORT_TYPE_CHOICES = "yariyillik", "yillik"
REPORT_TEMPLATE_FILES = {
    "yariyillik": YARIYIL_TEMPLATE_PATH,
    "yillik": YILSONU_TEMPLATE_PATH,
}

SIGMA: int = 3
START_COL = 3
END_COL = START_COL + NUM_OF_COMPANIES

# PLOT PARAMETERS
PRESENTATION_PAGES = {0, 1, 2, 3, 17, 21, 25, 42, 68, 84, 103}
ANNOTATION_OFFSET_PIXELS = -5, 5
HORIZONTAL_MEAN_COLOR = "purple"
HORIZONTAL_MEAN_ALPHA = 0.7
FONT_SIZE = 10.5

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create powerpoint files based on benchmark data')
    parser.add_argument('year', type=int, help='Year of the data (e.g. 2023)')
    parser.add_argument(
        'type', type=str, choices=REPORT_TYPE_CHOICES,
        help=f'Type of the data (must be {REPORT_TYPE_CHOICES[0]} or {REPORT_TYPE_CHOICES[1]})'
    )
    return parser.parse_args()

def generate_company_text(company_list):
    company_lines = []
    for i, company in enumerate(company_list, start=1):
        line = f"- {i} numaralı Şirket, {company}'ı"
        if i == len(company_list):
            line += " temsil etmekte iken"
        company_lines.append(line)
    company_lines.append(f"{num_of_group_companies + 1} - {NUM_OF_COMPANIES} arasında numaralandırılan Şirketler, Kıyaslama Çalışmasına dahil olan diğer Elektrik Dağıtım Şirketlerini temsil etmektedir.")
    return "\n".join(company_lines)

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


def standardgraph(row) -> plt.Figure:
    # Select the relevant columns and transpose the DataFrame and reset the index to companies
    transposable = merged_df[["APG No"] + list(COMPANIES_RANGE)]
    transposed = transposable.set_index("APG No").T.reset_index().rename(columns={"index": "companies"})

    # If next line doesn't exist, the graph will be overwritten by each row. Somehow necessary.
    ax = plt.figure()

    # Create the scatter plot
    ax = sns.scatterplot(
        data=transposed,
        x="companies",
        y=row['APG No'],
        legend=False,
        hue=company_color_indicator
    )
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.grid(axis='y')
    ax.set_xticks(COMPANIES_RANGE)
    ax.set_xticklabels(COMPANIES_RANGE)
    ax.set(title=row['APG Full Name'])

    # annotate points on the graph
    for company_index in COMPANIES_RANGE:
        value = row[company_index]
        formatted_value = format_percentage(value) if row["Birim"] == "%" else str(value)
        plt.annotate(
            text=formatted_value,
            xy=(company_index, value),
            xytext=ANNOTATION_OFFSET_PIXELS,
            textcoords="offset pixels"
        )

    # set the y-label to indicate that the corresponding KPI is in millions TL
    if row["Birim"] == "TL":  # MILYON TL???
        ax.set(ylabel='Milyon TL')

    # Set y-axis tick labels if needed
    if row["Birim"] == "%":
        ax.set_yticklabels(map(format_percentage, ax.get_yticks()))

    # draw horizontal mean value
    ax.axhline(
        y=row["filtered_mean"],
        color=HORIZONTAL_MEAN_COLOR,
        alpha=HORIZONTAL_MEAN_ALPHA,
    )

    return ax.get_figure()


def create_powerpoint():
    template_file = REPORT_TEMPLATE_FILES[report_type]
    graphics_save_directory = GRAPHICS_DIRECTORY / f'{report_year}_{report_type}' / company_group
    # Create the directories if they don't exist
    graphics_save_directory.mkdir(parents=True, exist_ok=True)
    presentation = Presentation(template_file)

    for _, row in merged_df.iterrows():
        slide = presentation.slides[row["Sayfa"]]
        left, top, height, width = row["Left"], row["Top"], row["Height"], row["Width"]
        grafik_tipi = row["Grafik_tipi"]

        create_figure_function_mapping = {
            "standard": standardgraph,
            # "stacked": stackedgraph,
            # "overlayed": overlayedgraph,
            # "e316": e316
        }
        create_figure_function = create_figure_function_mapping[grafik_tipi]
        fig = create_figure_function(row)

        # save figure to local directory
        apg_pic_name = f'{row["APG Full Name"]}.png'
        # Replace illegal characters in the file name with underscores
        clean_apg_path = ILLEGAL_WINDOWS_PATH_CHARACTERS.sub('_', apg_pic_name)
        pic_path = graphics_save_directory / clean_apg_path
        # Create the directories if they don't exist
        pic_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the figure to the local directory
        fig.savefig(pic_path, bbox_inches='tight')
        logger.info(f"Saved the figure for {row['APG No']} to {pic_path.relative_to(MAIN_DIRECTORY)}")

        # Add the figure to the slide
        slide.shapes.add_picture(str(pic_path), left, top, width, height)

    bulgu_shapes = [
        shape for slide in presentation.slides
        for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX and shape.text.startswith("Bulgu")
    ]

    for idx, shape in enumerate(bulgu_shapes):
        filtered_mean_value = merged_df.iloc[idx]["filtered_mean"]
        shape.text = f'Bulgu\n{NUM_OF_COMPANIES} şirketin ortalaması {filtered_mean_value} olarak tespit edilmiştir.'
        paragraphs = shape.text_frame.paragraphs
        paragraphs[0].font.bold = True
        for paragraph in paragraphs:
            paragraph.font.size = Pt(FONT_SIZE)

    presentation_pages = set(PRESENTATION_PAGES)  # sunum başlıkları ve APG başlıklarının olduğu sayfalar
    unique_graph_pages = set(merged_df.Sayfa)
    presentation_pages.update(unique_graph_pages)  # Add the graph pages to the set

    # removing the pages that are not selected
    max_page_num = merged_df.Sayfa.max()
    for slide_num in range(max_page_num, 0, -1):
        if slide_num not in presentation_pages:
            xml_slides = presentation.slides._sldIdLst
            slides = list(xml_slides)
            xml_slides.remove(slides[slide_num])

    # ilk ve ikinci slaytda yılı ve şirket ismini değiştirme
    ay = 6 if report_type == REPORT_TYPE_CHOICES[0] else 12
    presentation.slides[0].shapes[3].text = f'{report_year} Yılı {ay} Aylık Döneme Ait Performans Göstergesi Sonuçları'

    presentation_text_template = PRESENTATION_INTRO_TEMPLATE_PATH.read_text()
    company_enumeration_text = generate_company_text(company_list)
    formatted_intro_text = presentation_text_template.format(company_text=company_enumeration_text, company_group=company_group, num_of_APG=unique_categories_amount)
    presentation.slides[1].shapes[3].text = formatted_intro_text
    presentation_path = REPORTS_DIRECTORY / f'{report_year}_{report_type}' / company_group / f'Kıyaslama Raporu {report_year}_{report_type}_{company_group}.pptx'
    # Create the directories if they don't exist
    presentation_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(presentation_path)

    logger.info(f"Saved the presentation for {company_group} to {presentation_path.relative_to(MAIN_DIRECTORY)}")

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
    unique_categories_amount = df['Category No'].nunique()

    # Add a new column to the DataFrame that concatenates the APG No and APG İsmi
    df['APG Full Name'] = df.apply(lambda row: f'{row["APG No"]}-{row["APG İsmi"]}', axis=1)

    for company_group, company_list in COMPANY_GROUPS.items():
        # Shuffle columns for each group and store in a list
        shuffled_groups = []
        for _, group in df.groupby('Category No'):
            shuffled_group = shuffle_columns(group, company_list)
            shuffled_group.columns.values[START_COL:END_COL] = COMPANIES_RANGE
            shuffled_groups.append(shuffled_group)

        # Merge the shuffled groups back together and reset the column names
        merged_df = pd.concat(shuffled_groups).reset_index(drop=True)

        # Apply the filtered_mean function to each row and create a new column
        merged_df['filtered_mean'] = merged_df.apply(filtered_mean, axis=1)
        merged_df = pd.merge(merged_df, pptx_layout, left_on='APG No', right_on='APG Kodu', how='left')

        # merged_df.to_excel(REPORTS_DIRECTORY / f'{report_year}_{report_type}' / company_group / f"{report_year}_{report_type}_{company_group}_Shuffled.xlsx", index=False)

        # Create a list to indicate a company group (0) or their rivals (1) for each company group
        num_of_group_companies = len(company_list)
        num_of_other_companies = NUM_OF_COMPANIES - num_of_group_companies
        company_color_indicator = [0] * num_of_group_companies + [1] * num_of_other_companies

        create_powerpoint()
