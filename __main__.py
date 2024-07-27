#! /usr/bin/env python

# Standard library imports
import logging
from pathlib import Path
import re

# Third-party library imports
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE
import seaborn as sns
import yaml

# Configure seaborn
sns.set_style(style="whitegrid")
sns.set_palette("muted")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the regular expression pattern to match illegal characters in Windows file paths
ILLEGAL_WINDOWS_PATH_CHARACTERS = re.compile(r'[\\/:*?"<>|\n]')
ENCODING = 'utf-8'

# Define the directory paths
CODING_DIRECTORY = Path(__file__).parent
MAIN_DIRECTORY = CODING_DIRECTORY.parent
GRAPHICS_DIRECTORY = MAIN_DIRECTORY / "Grafikler"
REPORTS_DIRECTORY = MAIN_DIRECTORY / "Raporlar"

# Define the file paths
CONFIG_PATH = CODING_DIRECTORY / "config.yaml"
PRESENTATION_INTRO_TEMPLATE_PATH = CODING_DIRECTORY / "presentation_intro_template.txt"

# Load YAML configuration
with CONFIG_PATH.open(encoding=ENCODING) as config_file:
    config = yaml.safe_load(config_file)

MASTER_FILE = MAIN_DIRECTORY / config['MASTER_FILE']
TEMPLATE_PATH = MAIN_DIRECTORY / config['TEMPLATE_PATH']

COMPANY_GROUPS = config['COMPANY_GROUPS']
COMPANY_GROUPS_EXCLUDED_FROM_REPORT = config['COMPANY_GROUPS_EXCLUDED_FROM_REPORT']

# Check if the companies in COMPANY_GROUPS_EXCLUDED_FROM_REPORT are also in COMPANY_GROUPS
if set(COMPANY_GROUPS_EXCLUDED_FROM_REPORT) - set(COMPANY_GROUPS):
    raise ValueError("COMPANY_GROUPS_EXCLUDED_FROM_REPORT contains companies that are not in COMPANY_GROUPS. Please check the config.yaml file.")

NUM_OF_COMPANIES = sum(len(companies) for companies in COMPANY_GROUPS.values())
COMPANIES_RANGE = np.arange(1, NUM_OF_COMPANIES + 1)

REPORT_TYPE_CHOICES = "yariyillik", "yillik"
REPORT_YEAR = config['REPORT_YEAR']
REPORT_TYPE = config['REPORT_TYPE']

SIGMA: int = config['SIGMA']
START_COL = 3
END_COL = START_COL + NUM_OF_COMPANIES
GROUP_COMPANY_INDICATOR = 0
RIVAL_COMPANY_INDICATOR = 1

# PLOT PARAMETERS
PRESENTATION_PAGES = config['PRESENTATION_PAGES']
ANNOTATION_OFFSET_PIXELS = tuple(config['ANNOTATION_OFFSET_PIXELS'])
HORIZONTAL_MEAN_COLOR = config['HORIZONTAL_MEAN_COLOR']
HORIZONTAL_MEAN_ALPHA = config['HORIZONTAL_MEAN_ALPHA']
FONT_SIZE = config['FONT_SIZE']


def generate_presentation_intro_text(company_list) -> str:
    presentation_text_template = PRESENTATION_INTRO_TEMPLATE_PATH.read_text(encoding=ENCODING)

    company_lines = []
    for i, company in enumerate(company_list, start=1):
        line = f"- {i} numaralı Şirket, {company}'ı"
        if i == len(company_list):
            line += " temsil etmekte iken"
        company_lines.append(line)

    company_lines.append(f"{num_of_group_companies + 1} - {NUM_OF_COMPANIES}")
    company_enumeration_text = "\n".join(company_lines)
    return presentation_text_template.format(
        company_text=company_enumeration_text,
        company_group=company_group,
        num_of_APG=unique_categories_amount
    )


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

    group_df = df[new_column_order]
    group_df.columns.values[START_COL:END_COL] = COMPANIES_RANGE

    return group_df


def standardgraph(row) -> plt.Figure:
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
    if row["Birim"] == "TL":
        ax.set(ylabel='TL')

    # Set y-axis tick labels if needed
    if row["Birim"] == "%":
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(map(format_percentage, ax.get_yticks()))

    # draw horizontal mean value
    ax.axhline(
        y=row["filtered_mean"],
        color=HORIZONTAL_MEAN_COLOR,
        alpha=HORIZONTAL_MEAN_ALPHA,
    )

    return ax.get_figure()

def stackedgraph(row):
    stacked_apg_nos = category_to_apg_dict[row["Category No"]]
    ax = plt.figure()
    ax = transposed[stacked_apg_nos].plot(kind='bar', stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xticks(COMPANIES_RANGE)
    ax.set_xticklabels(COMPANIES_RANGE)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(map(format_percentage, ax.get_yticks()))

    return ax.get_figure()


def overlayedgraph(row):
    apg = row["APG Group"]
    alt_bilgi = f"{apg} EK"

    # create a custom colored map object
    custom_color_map = ListedColormap(['cornflowerblue', 'coral'])

    # bar plot (sub-info)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xticks(COMPANIES_RANGE)
    ax1.set_xticklabels(COMPANIES_RANGE)
    ax1.bar(x=transposed.companies,
            height=transposed[alt_bilgi],
            color='#CCFFCC',
            width=0.4,
            zorder=2)
    ax1.grid(visible=False)
    ax1.set(ylabel=f"{alt_bilgi}\n(Çubuk Gösterim)")

    for x, y in zip(transposed.companies, transposed[alt_bilgi]):
        formatted_text = format_percentage(y) if row["Birim"] == "%" else str(y)
        plt.text(x,
                 0,
                 formatted_text,
                 horizontalalignment='center',
                 verticalalignment='bottom')

        # create the second (scatter) graph
        ax2 = ax1.twinx()
        ax2.scatter(
            x=transposed.companies,
            y=transposed[row['APG No']],
            c=company_color_indicator,
            cmap=custom_color_map,
            zorder=3
        )
        ax2.grid(axis='y')
        ax2.set(ylabel=row["APG Group"])

        for company_index in COMPANIES_RANGE:
            value = row[company_index]
            formatted_value = format_percentage(value) if row["Birim"] == "%" else str(value)
            plt.annotate(
                text=formatted_value,
                xy=(company_index, value),
                xytext=(2,5),
                textcoords="offset pixels",
                zorder=4
            )

    return fig

def create_powerpoint():
    graphics_save_directory = GRAPHICS_DIRECTORY / f'{REPORT_YEAR}_{REPORT_TYPE}' / company_group
    # Create the directories if they don't exist
    graphics_save_directory.mkdir(parents=True, exist_ok=True)
    presentation = Presentation(TEMPLATE_PATH)

    for _, row in shuffled_df.iterrows():
        slide = presentation.slides[row["Sayfa"]]
        left, top, height, width = row["Left"], row["Top"], row["Height"], row["Width"]
        grafik_tipi = row["Grafik_tipi"]

        # Call the appropriate function based on grafik_tipi
        if grafik_tipi == "standard":
            fig = standardgraph(row)
        elif grafik_tipi == "stacked":
            fig = stackedgraph(row)
        elif grafik_tipi == "overlayed":
            fig = overlayedgraph(row)
        else:
            raise ValueError(f"Unknown grafik_tipi: {grafik_tipi}")

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
        # Close the figure to free up resources
        plt.close(fig)

        # Add the figure to the slide
        slide.shapes.add_picture(str(pic_path), left, top, width, height)

    bulgu_shapes = [
        shape for slide in presentation.slides
        for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX and shape.text.startswith("Bulgu")
    ]

    for idx, shape in enumerate(bulgu_shapes):
        filtered_mean_value = shuffled_df.iloc[idx]["filtered_mean"]
        shape.text = f'Bulgu\n{NUM_OF_COMPANIES} şirketin ortalaması {filtered_mean_value:.2f} olarak tespit edilmiştir.'
        paragraphs = shape.text_frame.paragraphs
        paragraphs[0].font.bold = True
        for paragraph in paragraphs:
            paragraph.font.size = Pt(FONT_SIZE)

    presentation_pages = set(PRESENTATION_PAGES)  # sunum başlıkları ve APG başlıklarının olduğu sayfalar
    unique_graph_pages = set(shuffled_df.Sayfa)
    presentation_pages.update(unique_graph_pages)  # Add the graph pages to the set

    # removing the pages that are not selected
    max_page_num = shuffled_df.Sayfa.max()
    for slide_num in range(max_page_num, 0, -1):
        if slide_num not in presentation_pages:
            xml_slides = presentation.slides._sldIdLst
            slides = list(xml_slides)
            xml_slides.remove(slides[slide_num])

    # ilk ve ikinci slaytda yılı ve şirket ismini değiştirme
    ay = 6 if REPORT_TYPE == REPORT_TYPE_CHOICES[0] else 12
    presentation.slides[0].shapes[3].text = f'{REPORT_YEAR} Yılı {ay} Aylık Döneme Ait Performans Göstergesi Sonuçları'
    presentation.slides[1].shapes[3].text = generate_presentation_intro_text(company_list)
    presentation_path = REPORTS_DIRECTORY / f'{REPORT_YEAR}_{REPORT_TYPE}' / company_group / f'Kıyaslama Raporu {REPORT_YEAR}_{REPORT_TYPE}_{company_group}.pptx'
    # Create the directories if they don't exist
    presentation_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(presentation_path)

    logger.info(f"Saved the presentation for {company_group} to {presentation_path.relative_to(MAIN_DIRECTORY)}")

# This line ensures that the code is only run when the script is executed directly, not when it is imported as a module.
if __name__ == "__main__":
    # Read the data from the Excel file and divide it into two DataFrames for the data and the layout
    dataframe_dict = pd.read_excel(MASTER_FILE, sheet_name=[f"{REPORT_YEAR}_Total_Veriler", "pptx_layout"])
    df = dataframe_dict[f"{REPORT_YEAR}_Total_Veriler"]
    pptx_layout = dataframe_dict["pptx_layout"]

    merged_df = pd.merge(df, pptx_layout, left_on='APG No', right_on='APG Kodu', how='left')
    merged_df.Sayfa = merged_df.Sayfa.astype(int)

    # Add Category from APG No
    merged_df['Category No'] = merged_df['APG No'].str.split('.').str[0]
    # Extract subcategory using regex to group EK APG No's into a common category of APG No's
    merged_df['APG Group'] = merged_df['APG No'].str.extract(r'(\w+\.\d+)')[0]
    unique_categories_amount = merged_df['Category No'].nunique()

    stacked_df = merged_df[merged_df.Grafik_tipi == "stacked"][["Category No", "APG No"]]

    # Creating the dictionary
    category_to_apg_dict = stacked_df.groupby('Category No')['APG No'].apply(list).to_dict()

    # Add a new column to the DataFrame that concatenates the APG No and APG İsmi
    merged_df['APG Full Name'] = merged_df.apply(lambda row: f'{row["APG No"]}-{row["APG İsmi"]}', axis=1)

    for company_group, company_list in COMPANY_GROUPS.items():
        if company_group in COMPANY_GROUPS_EXCLUDED_FROM_REPORT:
            continue
        # Create a list to indicate a company group (0) or their rivals (1) for each company group
        num_of_group_companies = len(company_list)
        num_of_other_companies = NUM_OF_COMPANIES - num_of_group_companies
        company_color_indicator = [GROUP_COMPANY_INDICATOR] * num_of_group_companies + [RIVAL_COMPANY_INDICATOR] * num_of_other_companies

        # Shuffle columns for each group and store in a list
        shuffled_groups = [
            shuffle_columns(group, company_list)
            for _, group
            in merged_df.groupby('Category No')
        ]

        # Merge the shuffled groups back together and reset the column names
        shuffled_df = pd.concat(shuffled_groups).reset_index(drop=True)

        # Apply the filtered_mean function to each row and create a new column
        shuffled_df['filtered_mean'] = shuffled_df.apply(filtered_mean, axis=1)
        # shuffled_df.to_excel(REPORTS_DIRECTORY / f'{REPORT_YEAR}_{REPORT_TYPE}' / company_group / f"{REPORT_YEAR}_{REPORT_TYPE}_{company_group}_Shuffled.xlsx", index=False)

        # Select the relevant columns and transpose the DataFrame and reset the index to companies
        transposable = shuffled_df[["APG No"] + list(COMPANIES_RANGE)]
        transposed = transposable.set_index("APG No").T.reset_index().rename(columns={"index": "companies"})

        create_powerpoint()
