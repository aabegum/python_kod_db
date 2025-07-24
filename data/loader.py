# loader.py
"""
Data loading utilities for the Electric Distribution Dashboard.

This module handles loading data from Excel files, processing cumulative data,
and preparing data structures for analysis.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional, Union

from utils.constants import (
    ALL_COMPANIES,
    COMPANY_NAME_MAPPING,
    START_COL,
    END_COL,
    NUM_OF_COMPANIES,
    COMPANIES_RANGE,
    BENCHMARK_COLUMN_POSITIONS,
    get_company_column_positions,
    PERCENTAGE_CONVERSION_THRESHOLD,
    INVALID_DATA_VALUES,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES
)


class DataLoader:
    """
    Data loading and processing utilities for the dashboard.
    
    Handles loading Excel files, processing data structures, and preparing
    data for analysis and visualization.
    """
    
    def __init__(self, settings):
        """
        Initialize DataLoader with settings.
        
        Args:
            settings: DashboardSettings instance
        """
        self.settings = settings
    
    def get_excel_company_names(self, config_company_list: List[str]) -> List[str]:
        """
        Convert config company names to Excel company names.
        
        Args:
            config_company_list: List of company names from config
            
        Returns:
            List of Excel-compatible company names
        """
        excel_names = []
        for config_name in config_company_list:
            if config_name in COMPANY_NAME_MAPPING:
                excel_names.append(COMPANY_NAME_MAPPING[config_name])
            else:
                excel_names.append(config_name)
        return excel_names
    
    def load_main_data(self, file_path: Union[str, object], year: int) -> pd.DataFrame:
        """
        Load main performance data for a specific year.
        
        Args:
            file_path: Path to Excel file or file-like object
            year: Year to load data for
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            Exception: If data loading fails
        """
        sheet_name = self.settings.get_sheet_name(year)
        
        try:
            if hasattr(file_path, 'read'):
                # File-like object (uploaded file)
                if hasattr(file_path, 'name') and file_path.name.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding="ISO-8859-1")
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                # File path string
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding="ISO-8859-1")
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            return df
            
        except Exception as e:
            error_msg = ERROR_MESSAGES['DATA_LOAD_ERROR'].format(year, str(e))
            st.error(error_msg)
            
            # Try to show available sheets for debugging
            try:
                if hasattr(file_path, 'read'):
                    excel_file = pd.ExcelFile(file_path)
                else:
                    excel_file = pd.ExcelFile(file_path)
                st.info(f"Available sheets: {excel_file.sheet_names}")
            except:
                pass
            
            raise Exception(error_msg)
    
    def load_layout_data(self, file_path: Union[str, object]) -> pd.DataFrame:
        """
        Load PowerPoint layout configuration data.
        
        Args:
            file_path: Path to Excel file or file-like object
            
        Returns:
            DataFrame containing layout configuration
            
        Raises:
            Exception: If layout data loading fails
        """
        try:
            if hasattr(file_path, 'read'):
                layout_df = pd.read_excel(file_path, sheet_name="pptx_layout")
            else:
                layout_df = pd.read_excel(file_path, sheet_name="pptx_layout")
            
            return layout_df
            
        except Exception as e:
            error_msg = ERROR_MESSAGES['LAYOUT_LOAD_ERROR'].format(str(e))
            st.error(error_msg)
            raise Exception(error_msg)
    
    def process_main_data(self, df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge main data with layout configuration.
        
        Args:
            df: Main performance data
            layout_df: Layout configuration data
            
        Returns:
            Processed and merged DataFrame
        """
        # Merge main data with layout
        merged_df = pd.merge(df, layout_df, left_on='APG No', right_on='APG Kodu', how='left')
        
        # Process data types and create derived columns
        merged_df['Sayfa'] = merged_df['Sayfa'].astype(int, errors='ignore')
        merged_df['Category No'] = merged_df['APG No'].str.split('.').str[0]
        merged_df['APG Full Name'] = merged_df.apply(
            lambda row: f"{row['APG No']}-{row['APG İsmi']}", axis=1
        )
        merged_df['APG Group'] = merged_df['APG No'].str.extract(r'(\w+\.\d+)')
        
        return merged_df
    
    def shuffle_columns(self, df: pd.DataFrame, company_list: List[str]) -> pd.DataFrame:
        """
        Shuffle columns while keeping selected companies in order.
        
        Args:
            df: DataFrame to shuffle
            company_list: List of companies to keep in order
            
        Returns:
            DataFrame with shuffled columns
        """
        # Get fixed columns (before and after company data)
        fixed_columns = (df.columns[:START_COL].tolist() + 
                        df.columns[END_COL:].tolist())
        
        # Get columns to shuffle (excluding selected companies)
        columns_to_shuffle = [col for col in df.columns[START_COL:END_COL] 
                             if col not in company_list]
        
        # Shuffle the columns
        shuffled_columns = np.random.permutation(columns_to_shuffle)
        
        # Create new column order
        new_column_order = (fixed_columns[:START_COL] + 
                           company_list + 
                           shuffled_columns.tolist() + 
                           fixed_columns[START_COL:])
        
        # Reorder DataFrame
        group_df = df[new_column_order]
        
        # Rename company columns to numeric range
        group_df.columns.values[START_COL:END_COL] = COMPANIES_RANGE
        
        return group_df
    
    def load_cumulative_data(self, file_path: Union[str, object], 
                           sheet_name: str = 'Kümülatif') -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Load and process cumulative data from the Kümülatif sheet.
        
        Args:
            file_path: Path to Excel file or file-like object
            sheet_name: Name of the sheet containing cumulative data
            
        Returns:
            Tuple of (processed_dataframe, benchmark_data) or (None, None) if failed
        """
        try:
            # Load the sheet
            if hasattr(file_path, 'read'):
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            years = self.settings.cumulative_years
            companies = ALL_COMPANIES
            
            # Get column positions for companies and benchmarks
            company_column_positions = get_company_column_positions()
            benchmark_columns = BENCHMARK_COLUMN_POSITIONS
            
            # Extract benchmark data
            benchmark_data = self._extract_benchmark_data(df, years, benchmark_columns)
            
            # Extract company performance data
            all_data = self._extract_company_data(df, years, companies, company_column_positions)
            
            # Create result DataFrame
            result_df = pd.DataFrame(all_data)
            
            num_apgs = len(result_df['APG No'].unique())
            num_companies = len(result_df['Company'].unique())
            
            return result_df, benchmark_data
            
        except Exception as e:
            st.error(f"Error loading cumulative data: {e}")
            return None, None
    
    def _extract_benchmark_data(self, df: pd.DataFrame, years: List[str], 
                               benchmark_columns: Dict) -> Dict:
        """
        Extract benchmark data (min, max, avg) for each year.
        
        Args:
            df: Source DataFrame
            years: List of years to process
            benchmark_columns: Column positions for benchmark data
            
        Returns:
            Dictionary containing benchmark data by year
        """
        benchmark_data = {}
        
        for year in years:
            if year in benchmark_columns:
                min_pos = benchmark_columns[year]['min']
                max_pos = benchmark_columns[year]['max']
                avg_pos = benchmark_columns[year]['avg']
                
                # Check if columns exist
                if all(pos < len(df.columns) for pos in [min_pos, max_pos, avg_pos]):
                    try:
                        # Try to find valid benchmark values in first 10 rows
                        for idx in range(min(10, len(df))):
                            min_val = pd.to_numeric(df.iloc[idx, min_pos], errors='coerce')
                            max_val = pd.to_numeric(df.iloc[idx, max_pos], errors='coerce')
                            avg_val = pd.to_numeric(df.iloc[idx, avg_pos], errors='coerce')
                            
                            if not (pd.isna(min_val) or pd.isna(max_val) or pd.isna(avg_val)):
                                # Convert to percentage if needed
                                if abs(min_val) <= PERCENTAGE_CONVERSION_THRESHOLD:
                                    min_val *= 100
                                if abs(max_val) <= PERCENTAGE_CONVERSION_THRESHOLD:
                                    max_val *= 100
                                if abs(avg_val) <= PERCENTAGE_CONVERSION_THRESHOLD:
                                    avg_val *= 100
                                
                                benchmark_data[year] = {
                                    'min': min_val,
                                    'max': max_val,
                                    'avg': avg_val
                                }
                                break
                    except Exception as e:
                        print(f"Error extracting benchmark for {year}: {e}")
        
        return benchmark_data
    
    def _extract_company_data(self, df: pd.DataFrame, years: List[str], 
                             companies: List[str], 
                             company_column_positions: Dict) -> List[Dict]:
        """
        Extract company performance data for all years.
        
        Args:
            df: Source DataFrame
            years: List of years to process
            companies: List of company names
            company_column_positions: Column positions for company data
            
        Returns:
            List of dictionaries containing company performance data
        """
        apg_no_col = df.columns[0]
        apg_name_col = df.columns[1]
        
        all_data = []
        processed_apgs = 0
        
        for idx, row in df.iterrows():
            apg_no = str(row[apg_no_col])
            apg_name = str(row[apg_name_col])
            
            # Skip invalid rows
            if (apg_no in ['nan', 'APG No', ''] or 
                apg_name in ['nan', 'APG Adı', ''] or
                pd.isna(row[apg_no_col]) or 
                pd.isna(row[apg_name_col])):
                continue
            
            processed_apgs += 1
            
            # Process each company
            for company in companies:
                company_performance = {}
                
                # Extract performance data for each year
                for year in years:
                    performance_value = self._extract_company_year_value(
                        row, year, company, company_column_positions, df.columns
                    )
                    company_performance[year] = performance_value
                
                # Add record to results
                all_data.append({
                    'APG No': apg_no,
                    'APG Name': apg_name,
                    'Company': company,
                    'APG Full Name': f"{apg_no} - {apg_name}",
                    'APG Company Full Name': f"{apg_no} - {apg_name} ({company})",
                    **company_performance
                })
        
        return all_data
    
    def _extract_company_year_value(self, row: pd.Series, year: str, company: str,
                                   company_column_positions: Dict, 
                                   df_columns: pd.Index) -> float:
        """
        Extract and clean a single company's performance value for a specific year.
        
        Args:
            row: DataFrame row
            year: Year to extract
            company: Company name
            company_column_positions: Column positions mapping
            df_columns: DataFrame columns
            
        Returns:
            Cleaned performance value
        """
        if year in company_column_positions and company in company_column_positions[year]:
            col_pos = company_column_positions[year][company]
            
            if col_pos < len(df_columns):
                value = row.iloc[col_pos]
                return self._clean_performance_value(value, year, company)
            else:
                return 0
        else:
            return 0
    
    def _clean_performance_value(self, value: Any, year: str = None, 
                                company: str = None) -> float:
        """
        Clean and convert performance value to float.
        
        Args:
            value: Raw value to clean
            year: Year for debugging (optional)
            company: Company for debugging (optional)
            
        Returns:
            Cleaned float value
        """
        if pd.isna(value):
            return 0
        
        if isinstance(value, str):
            # Clean string values
            cleaned_value = value.replace('%', '').replace(',', '.').strip()
            
            if cleaned_value.lower() in INVALID_DATA_VALUES:
                return 0
            
            try:
                float_value = float(cleaned_value)
                # Convert to percentage if needed
                if abs(float_value) <= PERCENTAGE_CONVERSION_THRESHOLD and float_value != 0:
                    float_value *= 100
                return float_value
            except ValueError:
                if year and company:
                    print(f"Warning: Could not convert value '{value}' for company {company}, year {year}. Setting to 0.")
                return 0
        else:
            try:
                float_value = float(value)
                # Convert to percentage if needed
                if abs(float_value) <= PERCENTAGE_CONVERSION_THRESHOLD and float_value != 0:
                    float_value *= 100
                return float_value
            except (ValueError, TypeError):
                if year and company:
                    print(f"Warning: Invalid value '{value}' for company {company}, year {year}. Setting to 0.")
                return 0
    
    def create_transposed_data(self, shuffled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transposed data for chart plotting.
        
        Args:
            shuffled_df: Shuffled DataFrame with company columns
            
        Returns:
            Transposed DataFrame ready for plotting
        """
        transposable = shuffled_df[["APG No"] + list(COMPANIES_RANGE)]
        transposed = (transposable.set_index("APG No")
                     .T.reset_index()
                     .rename(columns={"index": "companies"}))
        return transposed
    
    def create_category_mappings(self, shuffled_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Create category mappings for stacked charts.
        
        Args:
            shuffled_df: Processed DataFrame
            
        Returns:
            Tuple of (category_to_apg_dict, category_to_apg_full_name_dict)
        """
        stacked_df = shuffled_df[shuffled_df['Grafik_tipi'] == "stacked"][
            ["Category No", "APG No", "APG Full Name"]
        ]
        
        category_to_apg_dict = stacked_df.groupby('Category No')['APG No'].apply(list).to_dict()
        category_to_apg_full_name_dict = stacked_df.groupby('Category No')['APG Full Name'].apply(list).to_dict()
        
        return category_to_apg_dict, category_to_apg_full_name_dict
    
    def get_available_apgs(self, merged_df: pd.DataFrame) -> List[str]:
        """
        Get list of available APGs from the data.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Sorted list of available APG numbers
        """
        return sorted(merged_df['APG No'].dropna().unique())
    
    def filter_data_by_apgs(self, merged_df: pd.DataFrame, selected_apgs: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to only include selected APGs.
        
        Args:
            merged_df: Source DataFrame
            selected_apgs: List of APG numbers to include
            
        Returns:
            Filtered DataFrame
        """
        return merged_df[merged_df['APG No'].isin(selected_apgs)]
    
    def get_apg_name_from_df(self, apg_no: str, df: pd.DataFrame) -> str:
        """
        Get APG name from DataFrame for a specific APG number.
        
        Args:
            apg_no: APG number to look up
            df: DataFrame containing APG data
            
        Returns:
            APG name if found, empty string otherwise
        """
        apg_rows = df[df['APG No'] == apg_no]
        if len(apg_rows) > 0:
            return apg_rows.iloc[0]['APG İsmi']
        return ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_process_standard_data(settings, file_path: Union[str, object], 
                                  year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process standard analysis data.
    
    Args:
        settings: DashboardSettings instance
        file_path: Path to data file or file-like object
        year: Year to load data for
        
    Returns:
        Tuple of (main_data, layout_data)
    """
    loader = DataLoader(settings)
    
    # Load main data
    main_df = loader.load_main_data(file_path, year)
    
    # Load layout data
    layout_df = loader.load_layout_data(file_path)
    
    # Process and merge data
    processed_df = loader.process_main_data(main_df, layout_df)
    
    return processed_df, layout_df

def load_and_process_cumulative_data(settings, file_path: Union[str, object]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load and process cumulative analysis data.
    
    Args:
        settings: DashboardSettings instance
        file_path: Path to data file or file-like object
        
    Returns:
        Tuple of (cumulative_data, benchmark_data) or (None, None) if failed
    """
    loader = DataLoader(settings)
    return loader.load_cumulative_data(file_path, settings.cumulative_sheet)

def prepare_shuffled_data(settings, processed_df: pd.DataFrame, 
                         company_list: List[str]) -> pd.DataFrame:
    """
    Prepare shuffled data for analysis.
    
    Args:
        settings: DashboardSettings instance
        processed_df: Processed DataFrame
        company_list: List of companies for the selected group
        
    Returns:
        Shuffled DataFrame ready for analysis
    """
    loader = DataLoader(settings)
    
    # Group by category and shuffle each group
    shuffled_groups = []
    for _, group in processed_df.groupby('Category No', sort=False):
        shuffled_group = loader.shuffle_columns(group, company_list)
        shuffled_groups.append(shuffled_group)
    
    # Concatenate all shuffled groups
    shuffled_df = pd.concat(shuffled_groups).reset_index(drop=True)
    
    return shuffled_df