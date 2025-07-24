"""
Constants and shared configuration values for the Electric Distribution Dashboard.

This module contains all the constants that are used across different modules
of the dashboard application.
"""

import numpy as np
from matplotlib.colors import ListedColormap

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
PAGE_TITLE = "Electric Distribution Dashboard"
PAGE_ICON = ":bar_chart:"
LAYOUT = "wide"
DASHBOARD_TITLE = " :bar_chart: Electric Distribution Companies Performance Dashboard"

# ============================================================================
# SESSION STATE KEYS
# ============================================================================
SESSION_STATE_KEYS = {
    'QUICK_SELECT_APGS': 'quick_select_apgs',
    'QUICK_SELECT_APGS_CUMULATIVE': 'quick_select_apgs_cumulative',
    'SELECTED_CATEGORIES': 'selected_categories',
    'SELECTED_SUBCATEGORIES': 'selected_subcategories',
    'PREV_CATEGORIES': 'prev_categories',
    'PREV_SUBCATEGORIES': 'prev_subcategories',
    'COMPANY_GROUP': 'company_group'
}

# ============================================================================
# FILE PATHS AND CONFIGURATION
# ============================================================================
CONFIG_FILENAME = "config.yaml"
DEFAULT_MASTER_FILE = 'data.xlsx'

# ============================================================================
# COMPANY CONFIGURATION
# ============================================================================
# All company names in the order they appear in Excel
ALL_COMPANIES = ['AYEDAŞ', 'Başkent', 'Toroslar', 'ADM', 'GDZ', 'Aras', 'SEDAŞ', 'UEDAŞ', 'YEDAŞ']

# Company name mapping from config names to Excel names
COMPANY_NAME_MAPPING = {
    'AYEDAŞ': 'AYEDAŞ',
    'BAŞKENT EDAŞ': 'Başkent', 
    'TOROSLAR EDAŞ': 'Toroslar',
    'ADM EDAŞ': 'ADM',
    'GDZ EDAŞ': 'GDZ', 
    'ARAS EDAŞ': 'Aras',
    'SEDAŞ': 'SEDAŞ',
    'UEDAŞ': 'UEDAŞ',
    'YEDAŞ': 'YEDAŞ',
    'TREDAŞ': 'TREDAŞ',
    'VEDAŞ': 'VEDAŞ'
}

# ============================================================================
# DATA STRUCTURE CONSTANTS
# ============================================================================
START_COL = 3
NUM_OF_COMPANIES = len(ALL_COMPANIES)
COMPANIES_RANGE = np.arange(1, NUM_OF_COMPANIES + 1)
END_COL = START_COL + NUM_OF_COMPANIES

# ============================================================================
# DEFAULT CONFIGURATION VALUES
# ============================================================================
DEFAULT_REPORT_YEAR = 2024
DEFAULT_REPORT_TYPE = 'yillik'

# Outlier detection default parameters
DEFAULT_SIGMA = 2
DEFAULT_IQR_FACTOR = 1.5
DEFAULT_MAD_THRESHOLD = 3.5

# Display formatting
DEFAULT_DECIMAL_DIGITS = 2
DEFAULT_ANNOTATION_FONT_SIZE = 10
DEFAULT_CHART_HEIGHT = 500

# Chart appearance defaults
DEFAULT_HORIZONTAL_MEAN_COLOR = 'purple'
DEFAULT_HORIZONTAL_MEAN_ALPHA = 0.7
DEFAULT_OVERLAY_GRAPH_BAR_COLOR = 'blue'
DEFAULT_OVERLAY_GRAPH_COLOR_MAP = ['#FF0000', '#00FF00']

# ============================================================================
# CUMULATIVE ANALYSIS DEFAULTS
# ============================================================================
DEFAULT_CUMULATIVE_ENABLED = True
DEFAULT_CUMULATIVE_SHEET = 'Kümülatif'
DEFAULT_CUMULATIVE_YEARS = ['2021', '2022', '2023', '20242']

# ============================================================================
# CHART COLORS AND STYLING
# ============================================================================
# Color schemes for different chart types
COMPANY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Outlier visualization colors
OUTLIER_MARKER_COLOR = 'rgba(255, 165, 0, 0)'
OUTLIER_BORDER_COLOR = 'orange'
OUTLIER_BORDER_WIDTH = 3

# Company group colors (red for selected, green for others)
SELECTED_COMPANY_COLOR = '#FF0000'  # Red
OTHER_COMPANY_COLOR = '#00FF00'     # Green
COMPANY_COLORSCALE = [SELECTED_COMPANY_COLOR, OTHER_COMPANY_COLOR]

# Benchmark line colors
BENCHMARK_COLORS = {
    'MAX': '#e74c3c',
    'MIN': '#3498db', 
    'AVG': '#27ae60'
}

# Method comparison colors
COMPARISON_COLORS = ['red', 'blue', 'green']

# ============================================================================
# OUTLIER DETECTION METHODS
# ============================================================================
OUTLIER_METHODS = {
    'STD': 'std',
    'IQR': 'iqr', 
    'MAD': 'mad'
}

OUTLIER_METHOD_LABELS = {
    'std': 'STD (Standard Deviation)',
    'iqr': 'IQR (Interquartile Range)',
    'mad': 'MAD (Median Absolute Deviation)'
}

# ============================================================================
# CHART CONFIGURATION
# ============================================================================
CHART_TYPES = {
    'STANDARD': 'standard',
    'STACKED': 'stacked',
    'OVERLAYED': 'overlayed'
}

# Plot template
PLOT_TEMPLATE = 'plotly_white'

# Legend configuration
LEGEND_CONFIG = {
    'orientation': "v",
    'yanchor': "top",
    'y': 0.98,
    'xanchor': "left", 
    'x': 1.02,
    'font': {'size': 10, 'color': 'white'},
    'bgcolor': 'rgba(52, 58, 64, 0.95)',
    'bordercolor': 'rgba(255,255,255,0.3)',
    'borderwidth': 1,
    'itemsizing': 'constant'
}

# Margin configuration
CHART_MARGINS = {
    't': 100,
    'b': 50, 
    'l': 50,
    'r': 120
}

# ============================================================================
# TEXT AND FORMATTING
# ============================================================================
# Performance analysis text
PERFORMANCE_WARNING_THRESHOLD = 50
PERFORMANCE_WARNING_TEXT = "⚠️ {} APGs selected. This may affect performance. Consider selecting fewer APGs for better responsiveness."

# Filter section labels
FILTER_LABELS = {
    'CATEGORIES': "📂 APG Category & Subcategory Selection",
    'SPECIFIC_FILTERS': "🔍 APG-Specific Filters",
    'SELECTION_CONTROLS': "⚙️ Selection Controls",
    'IMPORT_EXPORT': "📤 Import/Export Selection"
}

# Analysis section labels
ANALYSIS_LABELS = {
    'STANDARD_TITLE': "📊 Analysis Options:",
    'CUMULATIVE_TITLE': "📊 Outlier Analysis Options:",
    'OUTLIER_SCOPE': "Outlier Detection Scope:",
    'OUTLIER_METHOD': "Outlier Detection Method:",
    'YEAR_SELECTION': "📅 Year Selection:"
}

# ============================================================================
# DATA VALIDATION AND ERROR MESSAGES
# ============================================================================
ERROR_MESSAGES = {
    'CONFIG_NOT_FOUND': "Configuration file (config.yaml) not found. Please ensure it exists in the same directory.",
    'DATA_LOAD_ERROR': "❌ Could not load data for year {}: {}",
    'LAYOUT_LOAD_ERROR': "Could not load pptx_layout sheet: {}",
    'NO_COMPANIES_FOUND': "No companies found for the selected group.",
    'NO_DATA_FOR_APGS': "❌ No data found for selected APGs",
    'NO_APGS_FOUND': "❌ No APGs found for the selected companies",
    'CUMULATIVE_DATA_ERROR': "Could not load cumulative data from Kümülatif sheet"
}

SUCCESS_MESSAGES = {
    'DATA_LOADED': "✅ Successfully loaded {} data: {} records",
    'CUMULATIVE_LOADED': "✅ Loaded data for {} APGs and {} companies"
}

WARNING_MESSAGES = {
    'GROUP_EXCLUDED': "{} is excluded from reports.",
    'NO_CATEGORIES': "⚠️ No categories found in APG data.",
    'NO_SUBCATEGORIES': "⚠️ No subcategories selected. Showing all APGs in selected categories.",
    'CUMULATIVE_DISABLED': "Cumulative charts are disabled in configuration."
}

# ============================================================================
# BENCHMARK CALCULATION CONSTANTS
# ============================================================================
# Column positions for cumulative benchmark data
BENCHMARK_COLUMN_POSITIONS = {
    '2021': {'min': 12, 'max': 13, 'avg': 14},
    '2022': {'min': 24, 'max': 25, 'avg': 26},
    '2023': {'min': 36, 'max': 37, 'avg': 38},
    '2024': {'min': 48, 'max': 49, 'avg': 50}
}

# Company column positions for cumulative data
def get_company_column_positions():
    """Get company column positions for cumulative data"""
    return {
        '2021': {company: 3 + i for i, company in enumerate(ALL_COMPANIES)},
        '2022': {company: 15 + i for i, company in enumerate(ALL_COMPANIES)},
        '2023': {company: 27 + i for i, company in enumerate(ALL_COMPANIES)},
        '2024': {company: 39 + i for i, company in enumerate(ALL_COMPANIES)}
    }

# ============================================================================
# UTILITY CONSTANTS
# ============================================================================
# Text wrapping
DEFAULT_WORD_WRAP_WIDTH = 50

# Percentage conversion threshold
PERCENTAGE_CONVERSION_THRESHOLD = 1

# Invalid data indicators
INVALID_DATA_VALUES = ['#div/0!', '', 'nan', 'n/a', 'null']

# File export constants
EXPORT_FILE_FORMATS = {
    'CSV': 'text/csv',
    'JSON': 'application/json'
}

# Dashboard modes
DASHBOARD_MODES = [
    "Standard Performance Analysis",
    "Cumulative Year-over-Year Analysis"
]

# Available years for analysis
AVAILABLE_YEARS = [2021, 2022, 2023, 20242]

# ============================================================================
# REGEX PATTERNS
# ============================================================================
APG_GROUP_PATTERN = r'(\w+\.\d+)'

# ============================================================================
# MATHEMATICAL CONSTANTS
# ============================================================================
# MAD calculation constant
MAD_CONSTANT = 0.6745

# Minimum data points for statistical calculations
MIN_DATA_POINTS_IQR = 4
MIN_DATA_POINTS_GENERAL = 2

# Statistical percentiles
Q1_PERCENTILE = 25
Q3_PERCENTILE = 75

# ============================================================================
# HELPER FUNCTIONS FOR CONSTANTS
# ============================================================================
def get_default_color_map():
    """Get default color map as ListedColormap"""
    return ListedColormap(DEFAULT_OVERLAY_GRAPH_COLOR_MAP)

def get_companies_range():
    """Get companies range array"""
    return COMPANIES_RANGE

def get_all_companies():
    """Get list of all companies"""
    return ALL_COMPANIES.copy()

def get_company_name_mapping():
    """Get company name mapping dictionary"""
    return COMPANY_NAME_MAPPING.copy()