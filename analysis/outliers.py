"""
Outlier detection and analysis for the Electric Distribution Dashboard.

This module contains all outlier detection algorithms (STD, IQR, MAD) and
related analysis functions, matching the PowerPoint code implementation exactly.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional

from utils.constants import (
    COMPANIES_RANGE,
    NUM_OF_COMPANIES,
    DEFAULT_SIGMA,
    DEFAULT_IQR_FACTOR,
    DEFAULT_MAD_THRESHOLD,
    MAD_CONSTANT,
    MIN_DATA_POINTS_IQR,
    MIN_DATA_POINTS_GENERAL,
    Q1_PERCENTILE,
    Q3_PERCENTILE
)


class OutlierDetector:
    """
    Comprehensive outlier detection using multiple statistical methods.
    
    Implements Standard Deviation (STD), Interquartile Range (IQR), and
    Median Absolute Deviation (MAD) methods for outlier detection.
    """
    
    def __init__(self, settings=None):
        """
        Initialize OutlierDetector.
        
        Args:
            settings: DashboardSettings instance for configuration
        """
        self.settings = settings
        self.sigma = settings.sigma if settings else DEFAULT_SIGMA
        self.iqr_factor = settings.iqr_factor if settings else DEFAULT_IQR_FACTOR
        self.mad_threshold = settings.mad_threshold if settings else DEFAULT_MAD_THRESHOLD
    
    def filtered_mean_with_outliers_std(self, data: List[float], sigma: float = None) -> Dict[str, Any]:
        """
        Calculate filtered mean and detect outliers using Standard Deviation method.
        Matches the PowerPoint code's STD outlier detection exactly.
        
        Args:
            data: List of numeric values
            sigma: Number of standard deviations for outlier threshold
            
        Returns:
            Dictionary containing filtered mean, outliers, and statistics
        """
        if sigma is None:
            sigma = self.sigma
        
        data_array = np.array(data, dtype=float)
        valid_data = data_array[~np.isnan(data_array)]
        
        if len(valid_data) < MIN_DATA_POINTS_GENERAL:
            return {
                'filtered_mean': np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        mean = valid_data.mean()
        std = valid_data.std()
        
        if std == 0 or np.isnan(std):
            return {
                'filtered_mean': mean if not np.isnan(mean) else np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        lower_bound = mean - sigma * std
        upper_bound = mean + sigma * std
        
        outliers = []
        non_outliers = []
        outlier_indices = []
        
        for i, val in enumerate(data_array):
            if not np.isnan(val):
                if val < lower_bound or val > upper_bound:
                    outliers.append(val)
                    outlier_indices.append(i)
                else:
                    non_outliers.append(val)
        
        # Calculate filtered mean (excluding outliers)
        filtered_data = np.array(non_outliers)
        if len(filtered_data) == 0:
            filtered_mean = mean  # Fallback to original mean if all filtered out
        else:
            filtered_mean = filtered_data.mean()
        
        return {
            'filtered_mean': filtered_mean,
            'outliers': outliers,
            'non_outliers': non_outliers,
            'outlier_count': len(outliers),
            'outlier_indices': outlier_indices,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'statistics': {'mean': mean, 'std': std, 'sigma': sigma}
        }
    
    def filtered_mean_with_outliers_iqr(self, data: List[float], iqr_factor: float = None) -> Dict[str, Any]:
        """
        Calculate filtered mean and detect outliers using IQR method.
        Matches the PowerPoint code's IQR outlier detection exactly.
        
        Args:
            data: List of numeric values
            iqr_factor: Multiplier for IQR bounds
            
        Returns:
            Dictionary containing filtered mean, outliers, and statistics
        """
        if iqr_factor is None:
            iqr_factor = self.iqr_factor
        
        data_array = np.array(data, dtype=float)
        valid_data = data_array[~np.isnan(data_array)]
        
        if len(valid_data) < MIN_DATA_POINTS_IQR:
            return {
                'filtered_mean': np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        q1 = np.percentile(valid_data, Q1_PERCENTILE)
        q3 = np.percentile(valid_data, Q3_PERCENTILE)
        iqr = q3 - q1
        
        if iqr == 0 or np.isnan(iqr):
            return {
                'filtered_mean': valid_data.mean() if len(valid_data) > 0 else np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        
        outliers = []
        non_outliers = []
        outlier_indices = []
        
        for i, val in enumerate(data_array):
            if not np.isnan(val):
                if val < lower_bound or val > upper_bound:
                    outliers.append(val)
                    outlier_indices.append(i)
                else:
                    non_outliers.append(val)
        
        # Calculate filtered mean (excluding outliers)
        filtered_data = np.array(non_outliers)
        if len(filtered_data) == 0:
            filtered_mean = valid_data.mean()  # Fallback to original mean
        else:
            filtered_mean = filtered_data.mean()
        
        return {
            'filtered_mean': filtered_mean,
            'outliers': outliers,
            'non_outliers': non_outliers,
            'outlier_count': len(outliers),
            'outlier_indices': outlier_indices,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'statistics': {'q1': q1, 'q3': q3, 'iqr': iqr, 'iqr_factor': iqr_factor}
        }
    
    def filtered_mean_with_outliers_mad(self, data: List[float], mad_threshold: float = None) -> Dict[str, Any]:
        """
        Calculate filtered mean and detect outliers using MAD method.
        Matches the PowerPoint code's MAD outlier detection exactly.
        
        Args:
            data: List of numeric values
            mad_threshold: Threshold for modified Z-score
            
        Returns:
            Dictionary containing filtered mean, outliers, and statistics
        """
        if mad_threshold is None:
            mad_threshold = self.mad_threshold
        
        data_array = np.array(data, dtype=float)
        valid_data = data_array[~np.isnan(data_array)]
        
        if len(valid_data) < MIN_DATA_POINTS_GENERAL:
            return {
                'filtered_mean': np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        
        if mad == 0 or np.isnan(mad):
            return {
                'filtered_mean': valid_data.mean() if len(valid_data) > 0 else np.nan,
                'outliers': [],
                'non_outliers': valid_data.tolist(),
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        outliers = []
        non_outliers = []
        outlier_indices = []
        
        for i, val in enumerate(data_array):
            if not np.isnan(val):
                # Calculate modified Z-score
                modified_z = MAD_CONSTANT * (val - median) / mad
                
                if abs(modified_z) > mad_threshold:
                    outliers.append(val)
                    outlier_indices.append(i)
                else:
                    non_outliers.append(val)
        
        # Calculate filtered mean (excluding outliers)
        filtered_data = np.array(non_outliers)
        if len(filtered_data) == 0:
            filtered_mean = valid_data.mean()  # Fallback to original mean
        else:
            filtered_mean = filtered_data.mean()
        
        return {
            'filtered_mean': filtered_mean,
            'outliers': outliers,
            'non_outliers': non_outliers,
            'outlier_count': len(outliers),
            'outlier_indices': outlier_indices,
            'statistics': {'median': median, 'mad': mad, 'mad_threshold': mad_threshold}
        }
    
    def compute_outliers_for_dataframe(self, df: pd.DataFrame, company_list: List[str], 
                                     method: str = "std", threshold: float = None, 
                                     compare_all: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Compute outliers for the entire dataframe using specified method(s).
        
        Args:
            df: DataFrame containing company performance data
            company_list: List of companies in the selected group
            method: Primary outlier detection method ('std', 'iqr', 'mad')
            threshold: Threshold value for the primary method
            compare_all: Whether to compute all methods for comparison
            
        Returns:
            Tuple of (outlier_results_dict, updated_dataframe)
        """
        from data.loader import DataLoader
        loader = DataLoader(self.settings)
        excel_company_names = loader.get_excel_company_names(company_list)
        
        # Create display companies list
        display_companies = excel_company_names + [
            str(i) for i in range(len(excel_company_names) + 1, NUM_OF_COMPANIES + 1)
        ]
        display_companies = display_companies[:NUM_OF_COMPANIES]
        
        # Determine which methods to compute
        methods_to_compute = ['std', 'iqr', 'mad'] if compare_all else [method]
        outlier_results = {}
        
        # Initialize columns for each method
        for current_method in methods_to_compute:
            df[f'filtered_mean_{current_method}'] = np.nan
            df[f'outlier_count_{current_method}'] = 0
            df[f'outliers_{current_method}'] = None
            df[f'outlier_indices_{current_method}'] = None
        
        # Process each method
        for current_method in methods_to_compute:
            apgs_with_outliers = []
            outlier_details = []
            outlier_summary = {"total": 0, "by_apg": {}, "by_company": {}}
            
            # Process each row (APG)
            for idx, row in df.iterrows():
                values = row[COMPANIES_RANGE].values
                
                # Apply appropriate outlier detection method
                if current_method == 'std':
                    method_threshold = threshold if current_method == method else self.sigma
                    result = self.filtered_mean_with_outliers_std(values, sigma=method_threshold)
                elif current_method == 'iqr':
                    method_threshold = threshold if current_method == method else self.iqr_factor
                    result = self.filtered_mean_with_outliers_iqr(values, iqr_factor=method_threshold)
                else:  # mad
                    method_threshold = threshold if current_method == method else self.mad_threshold
                    result = self.filtered_mean_with_outliers_mad(values, mad_threshold=method_threshold)
                
                # Update dataframe with results
                df.at[idx, f'filtered_mean_{current_method}'] = result['filtered_mean']
                df.at[idx, f'outlier_count_{current_method}'] = result['outlier_count']
                df.at[idx, f'outliers_{current_method}'] = str(result['outliers'])
                df.at[idx, f'outlier_indices_{current_method}'] = str(result['outlier_indices'])
                
                # Process outliers for selected companies
                if result['outliers']:
                    outlier_summary["total"] += len(result['outliers'])
                    outlier_summary["by_apg"][row['APG No']] = result['outlier_count']
                    
                    for idx_outlier in result['outlier_indices']:
                        if idx_outlier < len(display_companies):
                            company_name = display_companies[idx_outlier]
                            if company_name in excel_company_names:
                                apgs_with_outliers.append(row['APG No'])
                                outlier_details.append({
                                    'APG No': row['APG No'],
                                    'APG Name': row['APG Full Name'],
                                    'Company': company_name,
                                    'Value': values[idx_outlier],
                                    'Birim': row['Birim'],
                                    'Method': current_method.upper()
                                })
                            
                            company_idx = idx_outlier + 1
                            if company_idx not in outlier_summary["by_company"]:
                                outlier_summary["by_company"][company_idx] = 0
                            outlier_summary["by_company"][company_idx] += 1
            
            outlier_results[current_method] = {
                'apgs_with_outliers': sorted(list(set(apgs_with_outliers))),
                'outlier_details': outlier_details,
                'outlier_summary': outlier_summary
            }
        
        return outlier_results, df


class OutlierStatistics:
    """
    Outlier statistics and analysis utilities.
    """
    
    @staticmethod
    def parse_outlier_indices(outlier_indices_str: str) -> List[int]:
        """
        Parse string representation of outlier indices back to list.
        
        Args:
            outlier_indices_str: String representation of indices
            
        Returns:
            List of integer indices
        """
        if pd.isna(outlier_indices_str) or outlier_indices_str == 'None' or outlier_indices_str == '[]':
            return []
        
        try:
            # Remove brackets and split by comma
            indices_str = outlier_indices_str.strip('[]')
            if not indices_str:
                return []
            
            # Split and convert to integers
            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
            return indices
        except:
            return []
    
    @staticmethod
    def display_outlier_statistics(outlier_results: Dict[str, Any], outlier_method_standard: str, 
                                 threshold_value: float, detect_outliers_scope: str, 
                                 compare_methods: bool = False) -> None:
        """
        Display comprehensive outlier statistics in Streamlit.
        
        Args:
            outlier_results: Results from outlier detection
            outlier_method_standard: Primary outlier detection method
            threshold_value: Threshold value used
            detect_outliers_scope: Scope of outlier detection
            compare_methods: Whether method comparison is enabled
        """
        method_key = 'std' if 'STD' in outlier_method_standard else ('iqr' if 'IQR' in outlier_method_standard else 'mad')
        primary_results = outlier_results[method_key]
        outlier_summary = primary_results['outlier_summary']
        outlier_details = primary_results['outlier_details']
        
        if outlier_summary["total"] > 0:
            expander_title = f"🔍 Outlier Analysis Summary - {outlier_method_standard} - Scope: {detect_outliers_scope}"
            
            with st.expander(expander_title, expanded=False):
                # Primary method statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Outliers", outlier_summary["total"])
                    st.metric("APGs with Outliers", len(outlier_summary["by_apg"]))
                
                with col2:
                    st.metric("Companies with Outliers", len(outlier_summary["by_company"]))
                    threshold_label = {
                        'std': f"Sigma = {threshold_value}",
                        'iqr': f"IQR Factor = {threshold_value}",
                        'mad': f"MAD Threshold = {threshold_value}"
                    }[method_key]
                    st.metric("Threshold", threshold_label)
                
                with col3:
                    if outlier_summary["by_apg"]:
                        top_apg = max(outlier_summary["by_apg"], key=outlier_summary["by_apg"].get)
                        st.metric("Most Outliers (APG)", f"{top_apg}: {outlier_summary['by_apg'][top_apg]}")
                    
                    if outlier_summary["by_company"]:
                        top_company = max(outlier_summary["by_company"], key=outlier_summary["by_company"].get)
                        st.metric("Most Outliers (Company)", f"#{top_company}: {outlier_summary['by_company'][top_company]}")
                
                # Method comparison if enabled
                if compare_methods and len(outlier_results) > 1:
                    st.markdown("---")
                    st.markdown("**📊 Method Comparison:**")
                    
                    comparison_data = []
                    for method, results in outlier_results.items():
                        comparison_data.append({
                            'Method': method.upper(),
                            'Total Outliers': results['outlier_summary']['total'],
                            'APGs Affected': len(results['outlier_summary']['by_apg']),
                            'Companies Affected': len(results['outlier_summary']['by_company']),
                            'Threshold': {
                                'std': f"σ = {DEFAULT_SIGMA}",
                                'iqr': f"factor = {DEFAULT_IQR_FACTOR}",
                                'mad': f"threshold = {DEFAULT_MAD_THRESHOLD}"
                            }[method]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Show method agreement
                    all_apgs = set()
                    for results in outlier_results.values():
                        all_apgs.update(results['apgs_with_outliers'])
                    
                    agreement_data = []
                    for apg in sorted(all_apgs):
                        methods_detecting = [
                            method.upper() for method, results in outlier_results.items()
                            if apg in results['apgs_with_outliers']
                        ]
                        agreement_data.append({
                            'APG': apg,
                            'Methods Detecting': ', '.join(methods_detecting),
                            'Agreement Level': f"{len(methods_detecting)}/3"
                        })
                    
                    if agreement_data:
                        st.markdown("**🤝 Method Agreement:**")
                        agreement_df = pd.DataFrame(agreement_data)
                        st.dataframe(agreement_df, use_container_width=True)
                
                st.markdown("---")
                st.markdown("**📖 Outlier Detection Legend:**")
                st.markdown("- 🔴 **Red dots**: Companies in selected group")
                st.markdown("- 🟢 **Green dots**: Other companies")
                st.markdown("- 🔶 **Orange diamonds**: Outlier values")
                st.markdown("- 📊 **Dashed line**: Filtered mean (excluding outliers)")
                
                if compare_methods:
                    st.markdown("- 📊 **Dotted lines**: Comparison method means")
                
                # Show outlier details
                if outlier_details:
                    st.markdown("**📋 Outlier Details for Selected Companies:**")
                    outlier_df = pd.DataFrame(outlier_details)
                    
                    # Format the Value column based on Birim
                    def format_value(row):
                        if row['Birim'] == '%':
                            return f"{row['Value']:.2f}%"
                        return f"{row['Value']:.2f}"
                    
                    outlier_df['Formatted Value'] = outlier_df.apply(format_value, axis=1)
                    display_cols = ['APG No', 'APG Name', 'Company', 'Formatted Value']
                    
                    if compare_methods:
                        display_cols.append('Method')
                    
                    st.dataframe(outlier_df[display_cols], use_container_width=True)
    
    @staticmethod
    def create_outlier_comparison_table(row: pd.Series, methods: List[str] = ['std', 'iqr', 'mad']) -> pd.DataFrame:
        """
        Create a comparison table for different outlier detection methods.
        
        Args:
            row: DataFrame row containing outlier analysis results
            methods: List of methods to compare
            
        Returns:
            DataFrame containing method comparison
        """
        comparison_data = []
        
        for method in methods:
            mean_col = f'filtered_mean_{method}'
            count_col = f'outlier_count_{method}'
            
            if mean_col in row and count_col in row:
                comparison_data.append({
                    'Method': method.upper(),
                    'Filtered Mean': f"{row[mean_col]:.3f}" if not pd.isna(row[mean_col]) else "N/A",
                    'Outlier Count': int(row[count_col]) if not pd.isna(row[count_col]) else 0,
                    'Threshold': {
                        'std': f"σ = {DEFAULT_SIGMA}",
                        'iqr': f"factor = {DEFAULT_IQR_FACTOR}",
                        'mad': f"threshold = {DEFAULT_MAD_THRESHOLD}"
                    }[method]
                })
        
        return pd.DataFrame(comparison_data)


# ============================================================================
# COMPATIBILITY FUNCTIONS FOR EXISTING CODE
# ============================================================================

def detect_outliers_iqr(data: List[float], k: float = None) -> Tuple[List[float], List[float]]:
    """
    Wrapper function to maintain compatibility with existing code.
    
    Args:
        data: List of numeric values
        k: IQR factor (multiplier)
        
    Returns:
        Tuple of (outliers_list, non_outliers_list)
    """
    detector = OutlierDetector()
    result = detector.filtered_mean_with_outliers_iqr(data, iqr_factor=k)
    return result['outliers'], result['non_outliers']

def detect_outliers_mad(data: List[float], threshold: float = None) -> Tuple[List[float], List[float]]:
    """
    Wrapper function to maintain compatibility with existing code.
    
    Args:
        data: List of numeric values
        threshold: MAD threshold value
        
    Returns:
        Tuple of (outliers_list, non_outliers_list)
    """
    detector = OutlierDetector()
    result = detector.filtered_mean_with_outliers_mad(data, mad_threshold=threshold)
    return result['outliers'], result['non_outliers']

def detect_outliers_std(data: List[float], sigma: float = None) -> Tuple[List[float], List[float]]:
    """
    Function for STD outlier detection (matching PowerPoint code).
    
    Args:
        data: List of numeric values
        sigma: Number of standard deviations
        
    Returns:
        Tuple of (outliers_list, non_outliers_list)
    """
    detector = OutlierDetector()
    result = detector.filtered_mean_with_outliers_std(data, sigma=sigma)
    return result['outliers'], result['non_outliers']

# Legacy function for backwards compatibility
def filtered_mean(row: pd.Series, sigma: float = None) -> float:
    """
    Calculate mean after removing outliers (legacy function).
    
    Args:
        row: DataFrame row containing company data
        sigma: Standard deviation multiplier
        
    Returns:
        Filtered mean value
    """
    if sigma is None:
        sigma = DEFAULT_SIGMA
    
    from utils.constants import START_COL, END_COL
    data = row[START_COL:END_COL]
    
    detector = OutlierDetector()
    result = detector.filtered_mean_with_outliers_std(data.values, sigma=sigma)
    
    return result['filtered_mean'] if not pd.isna(result['filtered_mean']) else data.mean()