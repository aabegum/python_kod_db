"""
Reusable UI components for the Electric Distribution Dashboard.

This module contains reusable UI components like performance summaries,
data tables, download sections, and metric displays.
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go

from utils.constants import (
    COMPANIES_RANGE,
    DEFAULT_DECIMAL_DIGITS,
    PERFORMANCE_WARNING_THRESHOLD,
    PERFORMANCE_WARNING_TEXT
)


class MetricDisplay:
    """
    Component for displaying performance metrics and statistics.
    """
    
    @staticmethod
    def create_metric_columns(metrics_data: Dict[str, Any], columns: int = 3) -> None:
        """
        Create metric columns layout.
        
        Args:
            metrics_data: Dictionary containing metric data
            columns: Number of columns to create
        """
        cols = st.columns(columns)
        
        metric_items = list(metrics_data.items())
        for i, (label, value) in enumerate(metric_items):
            with cols[i % columns]:
                if isinstance(value, dict):
                    # Handle metric with delta
                    st.metric(
                        label=label,
                        value=value.get('value', 'N/A'),
                        delta=value.get('delta', None),
                        help=value.get('help', None)
                    )
                else:
                    # Simple metric
                    st.metric(label=label, value=value)
    
    @staticmethod
    def create_performance_summary_metrics(summary_data: Dict[str, Any]) -> None:
        """
        Create performance summary metrics display.
        
        Args:
            summary_data: Dictionary containing performance summary data
        """
        if 'average_performance' in summary_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Performance", f"{summary_data['average_performance']:.2f}")
                st.metric("Best Performance", f"{summary_data['best_performance']:.2f}")
            
            with col2:
                st.metric("Worst Performance", f"{summary_data['worst_performance']:.2f}")
                st.metric("Performance Range", f"{summary_data['performance_range']:.2f}")
            
            with col3:
                if 'above_average_count' in summary_data:
                    # Standard mode metrics
                    above_avg = summary_data['above_average_count']
                    total = summary_data['total_data_points']
                    st.metric("Above Average", f"{above_avg}/{total}")
                    st.metric("Success Rate", f"{summary_data['success_rate']:.1f}%")
                else:
                    # Cumulative mode metrics
                    positive = summary_data.get('positive_performances', 0)
                    total = summary_data['total_data_points']
                    st.metric("Positive Performances", f"{positive}/{total}")
                    st.metric("Success Rate", f"{summary_data['success_rate']:.1f}%")
        
        # APG coverage metrics
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Total APGs Analyzed", f"{summary_data['total_apgs_analyzed']}")
            st.metric("Available APGs", f"{summary_data['available_apgs']}")
        
        with col5:
            apg_coverage = summary_data['apg_coverage']
            st.metric("APG Coverage", f"{apg_coverage:.1f}%")
            if summary_data['total_apgs_analyzed'] < summary_data['available_apgs']:
                st.info("🎯 Custom APG selection active")
    
    @staticmethod
    def create_outlier_summary_metrics(outlier_summary: Dict[str, Any], 
                                     threshold_info: Dict[str, Any]) -> None:
        """
        Create outlier analysis summary metrics.
        
        Args:
            outlier_summary: Dictionary containing outlier summary data
            threshold_info: Dictionary containing threshold information
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Outliers", outlier_summary["total"])
            st.metric("APGs with Outliers", len(outlier_summary["by_apg"]))
        
        with col2:
            st.metric("Companies with Outliers", len(outlier_summary["by_company"]))
            st.metric("Threshold", threshold_info.get('label', 'N/A'))
        
        with col3:
            if outlier_summary["by_apg"]:
                top_apg = max(outlier_summary["by_apg"], key=outlier_summary["by_apg"].get)
                st.metric("Most Outliers (APG)", f"{top_apg}: {outlier_summary['by_apg'][top_apg]}")
            
            if outlier_summary["by_company"]:
                top_company = max(outlier_summary["by_company"], key=outlier_summary["by_company"].get)
                st.metric("Most Outliers (Company)", f"#{top_company}: {outlier_summary['by_company'][top_company]}")


class ExpandableSection:
    """
    Component for creating expandable sections with consistent styling.
    """
    
    @staticmethod
    def create_performance_summary_section(summary_data: Dict[str, Any], 
                                         title: str, 
                                         expanded: bool = False) -> None:
        """
        Create an expandable performance summary section.
        
        Args:
            summary_data: Dictionary containing performance summary data
            title: Title for the expandable section
            expanded: Whether the section should be expanded by default
        """
        with st.expander(f"📊 {title}", expanded=expanded):
            MetricDisplay.create_performance_summary_metrics(summary_data)
    
    @staticmethod
    def create_outlier_analysis_section(outlier_results: Dict[str, Any],
                                      method_info: Dict[str, Any],
                                      expanded: bool = False) -> None:
        """
        Create an expandable outlier analysis section.
        
        Args:
            outlier_results: Dictionary containing outlier analysis results
            method_info: Dictionary containing method information
            expanded: Whether the section should be expanded by default
        """
        method_name = method_info.get('name', 'Unknown')
        scope = method_info.get('scope', 'Unknown')
        
        with st.expander(f"🔍 Outlier Analysis Summary - {method_name} - Scope: {scope}", expanded=expanded):
            primary_method = method_info.get('primary_method', 'std')
            if primary_method in outlier_results:
                outlier_summary = outlier_results[primary_method]['outlier_summary']
                threshold_info = method_info.get('threshold_info', {})
                
                MetricDisplay.create_outlier_summary_metrics(outlier_summary, threshold_info)
                
                # Method comparison if available
                if method_info.get('compare_methods', False) and len(outlier_results) > 1:
                    ExpandableSection._create_method_comparison_section(outlier_results, method_info)
                
                # Outlier legend
                ExpandableSection._create_outlier_legend(method_info.get('compare_methods', False))
                
                # Outlier details
                if outlier_results[primary_method]['outlier_details']:
                    ExpandableSection._create_outlier_details_section(
                        outlier_results[primary_method]['outlier_details'],
                        method_info.get('compare_methods', False)
                    )
    
    @staticmethod
    def _create_method_comparison_section(outlier_results: Dict[str, Any],
                                        method_info: Dict[str, Any]) -> None:
        """Create method comparison section."""
        st.markdown("---")
        st.markdown("**📊 Method Comparison:**")
        
        comparison_data = []
        for method, results in outlier_results.items():
            comparison_data.append({
                'Method': method.upper(),
                'Total Outliers': results['outlier_summary']['total'],
                'APGs Affected': len(results['outlier_summary']['by_apg']),
                'Companies Affected': len(results['outlier_summary']['by_company']),
                'Threshold': method_info.get('thresholds', {}).get(method, 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Method agreement analysis
        all_apgs = set()
        for results in outlier_results.values():
            all_apgs.update(results['apgs_with_outliers'])
        
        if all_apgs:
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
            
            st.markdown("**🤝 Method Agreement:**")
            agreement_df = pd.DataFrame(agreement_data)
            st.dataframe(agreement_df, use_container_width=True)
    
    @staticmethod
    def _create_outlier_legend(compare_methods: bool = False) -> None:
        """Create outlier detection legend."""
        st.markdown("---")
        st.markdown("**📖 Outlier Detection Legend:**")
        st.markdown("- 🔴 **Red dots**: Companies in selected group")
        st.markdown("- 🟢 **Green dots**: Other companies")
        st.markdown("- 🔶 **Orange diamonds**: Outlier values")
        st.markdown("- 📊 **Dashed line**: Filtered mean (excluding outliers)")
        
        if compare_methods:
            st.markdown("- 📊 **Dotted lines**: Comparison method means")
    
    @staticmethod
    def _create_outlier_details_section(outlier_details: List[Dict[str, Any]],
                                      compare_methods: bool = False) -> None:
        """Create outlier details section."""
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


class DataTable:
    """
    Component for displaying data tables with styling and functionality.
    """
    
    @staticmethod
    def create_styled_dataframe(df: pd.DataFrame, 
                               style_columns: Optional[List[str]] = None,
                               color_map: str = "Blues") -> None:
        """
        Create a styled dataframe display.
        
        Args:
            df: DataFrame to display
            style_columns: Columns to apply gradient styling to
            color_map: Color map for styling
        """
        if style_columns:
            available_style_cols = [col for col in style_columns if col in df.columns]
            if available_style_cols:
                st.dataframe(
                    df.style.background_gradient(cmap=color_map, subset=available_style_cols),
                    use_container_width=True
                )
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def create_pivot_table_display(df: pd.DataFrame, 
                                  index_cols: List[str],
                                  columns_col: str,
                                  values_cols: List[str],
                                  highlight_outliers: bool = False,
                                  outliers: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Create and display a pivot table.
        
        Args:
            df: Source DataFrame
            index_cols: Columns to use as index
            columns_col: Column to use as columns
            values_cols: Columns to use as values
            highlight_outliers: Whether to highlight outlier values
            outliers: List of outlier values to highlight
            
        Returns:
            Created pivot table DataFrame
        """
        pivot_table = df.pivot_table(
            index=index_cols,
            columns=columns_col,
            values=values_cols,
            aggfunc='first'
        )
        
        # Flatten column names
        pivot_table.columns = [f"{col[1]}_{col[0]}" for col in pivot_table.columns]
        
        if highlight_outliers and outliers:
            st.info("🔍 Cells highlighted in red indicate outlier values")
            
            def highlight_outlier_cells(val):
                if pd.isna(val):
                    return ''
                if val in outliers:
                    return 'background-color: #ffcccc; font-weight: bold'
                return ''
            
            st.dataframe(
                pivot_table.style.applymap(highlight_outlier_cells)
                .background_gradient(cmap="RdYlGn", axis=1, vmin=-10, vmax=10),
                use_container_width=True
            )
        else:
            st.dataframe(
                pivot_table.style.background_gradient(cmap="RdYlGn", axis=1),
                use_container_width=True
            )
        
        return pivot_table


class DownloadSection:
    """
    Component for creating download sections with various export formats.
    """
    
    @staticmethod
    def create_data_download_section(df: pd.DataFrame, 
                                   metadata: Dict[str, Any],
                                   base_columns: List[str],
                                   outlier_columns: Optional[List[str]] = None) -> None:
        """
        Create a comprehensive data download section.
        
        Args:
            df: DataFrame to download
            metadata: Dictionary containing metadata to add to download
            base_columns: Base columns to include in download
            outlier_columns: Additional outlier-related columns
        """
        # Prepare columns for display and download
        if outlier_columns:
            available_outlier_cols = [col for col in outlier_columns if col in df.columns]
            report_columns = base_columns + available_outlier_cols
        else:
            report_columns = base_columns
            available_outlier_cols = []
        
        available_columns = [col for col in report_columns if col in df.columns]
        
        # Add metadata columns
        df_copy = df.copy()
        for key, value in metadata.items():
            df_copy[key] = value
        
        # Display data table
        DataTable.create_styled_dataframe(df_copy[available_columns], 
                                        style_columns=base_columns[2:])  # Skip APG No and Name
        
        # Prepare download
        download_columns = list(metadata.keys()) + available_columns
        csv_data = df_copy[download_columns].to_csv(index=False).encode('utf-8')
        
        # Generate filename
        filename = DownloadSection._generate_filename(metadata)
        
        st.download_button(
            "📥 Download Data",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    
    @staticmethod
    def create_dual_download_section(detailed_df: pd.DataFrame,
                                   summary_df: pd.DataFrame,
                                   metadata: Dict[str, Any]) -> None:
        """
        Create a download section with both detailed and summary options.
        
        Args:
            detailed_df: Detailed data DataFrame
            summary_df: Summary data DataFrame
            metadata: Metadata for filename generation
        """
        col1, col2 = st.columns(2)
        
        with col1:
            # Detailed download
            detailed_csv = detailed_df.to_csv(index=False).encode('utf-8')
            detailed_filename = DownloadSection._generate_filename(metadata, suffix="detailed")
            
            st.download_button(
                "📥 Download Detailed Data",
                data=detailed_csv,
                file_name=detailed_filename,
                mime="text/csv"
            )
        
        with col2:
            # Summary download
            summary_csv = summary_df.to_csv().encode('utf-8')
            summary_filename = DownloadSection._generate_filename(metadata, suffix="summary")
            
            st.download_button(
                "📥 Download Summary Pivot",
                data=summary_csv,
                file_name=summary_filename,
                mime="text/csv"
            )
    
    @staticmethod
    def _generate_filename(metadata: Dict[str, Any], suffix: str = "") -> str:
        """Generate filename based on metadata."""
        year = metadata.get('Analysis_Year', 'data')
        company_group = metadata.get('Company_Group', 'group')
        apg_count = metadata.get('Selected_APG_Count', 0)
        total_apgs = metadata.get('Total_APG_Count', 0)
        
        filename_suffix = "AllAPGs" if apg_count == total_apgs else f"{apg_count}APGs"
        
        method = metadata.get('Outlier_Detection_Method', '')
        method_suffix = ""
        if method:
            method_key = 'std' if 'STD' in method else ('iqr' if 'IQR' in method else 'mad')
            method_suffix = f"_{method_key}"
        
        if suffix:
            suffix = f"_{suffix}"
        
        return f"{year}_{company_group}_{filename_suffix}{method_suffix}{suffix}_Analysis.csv"


class AlertsAndNotifications:
    """
    Component for displaying alerts, warnings, and notifications.
    """
    
    @staticmethod
    def show_performance_warning(item_count: int, 
                               threshold: int = PERFORMANCE_WARNING_THRESHOLD) -> None:
        """
        Show performance warning if item count exceeds threshold.
        
        Args:
            item_count: Number of items selected
            threshold: Threshold for showing warning
        """
        if item_count > threshold:
            st.warning(PERFORMANCE_WARNING_TEXT.format(item_count))
    
    @staticmethod
    def show_data_info(current_count: int, total_count: int, 
                      item_type: str = "items") -> None:
        """
        Show data information message.
        
        Args:
            current_count: Current number of items
            total_count: Total number of items available
            item_type: Type of items being counted
        """
        if current_count != total_count:
            st.info(f"🎯 Showing {current_count} of {total_count} available {item_type}")
    
    @staticmethod
    def show_analysis_info(apg_count: int, company_group: str, year: int = None) -> None:
        """
        Show analysis information message.
        
        Args:
            apg_count: Number of APGs being analyzed
            company_group: Company group name
            year: Analysis year (optional)
        """
        if year:
            st.info(f"📊 Displaying analysis for {apg_count} APGs in **{company_group}** group for year **{year}**")
        else:
            st.info(f"📊 Displaying analysis for {apg_count} APGs in **{company_group}** group")
    
    @staticmethod
    def show_cumulative_info(apg_count: int, company_count: int, 
                           company_group: str) -> None:
        """
        Show cumulative analysis information.
        
        Args:
            apg_count: Number of APGs being analyzed
            company_count: Number of companies being analyzed
            company_group: Company group name
        """
        st.info(f"📈 Displaying {apg_count} charts - each showing **{company_count} companies** from {company_group} group")


class ComparisonTable:
    """
    Component for creating comparison tables and analysis.
    """
    
    @staticmethod
    def create_method_comparison_table(row: pd.Series, 
                                     methods: List[str] = ['std', 'iqr', 'mad']) -> pd.DataFrame:
        """
        Create a comparison table for different outlier detection methods.
        
        Args:
            row: DataFrame row containing outlier analysis results
            methods: List of methods to compare
            
        Returns:
            DataFrame containing method comparison
        """
        from utils.constants import DEFAULT_SIGMA, DEFAULT_IQR_FACTOR, DEFAULT_MAD_THRESHOLD
        
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
    
    @staticmethod
    def display_method_comparison_tables(shuffled_df: pd.DataFrame, 
                                       max_tables: int = 5) -> None:
        """
        Display method comparison tables for multiple APGs.
        
        Args:
            shuffled_df: DataFrame containing APG data
            max_tables: Maximum number of tables to show
        """
        with st.expander("📊 Method Comparison Tables", expanded=False):
            for idx, (_, row) in enumerate(shuffled_df.iterrows()):
                if idx >= max_tables:
                    break
                
                st.subheader(f"{row['APG No']} - Comparison")
                comparison_table = ComparisonTable.create_method_comparison_table(row)
                st.dataframe(comparison_table, use_container_width=True)


# ============================================================================
# UTILITY FUNCTIONS FOR COMPONENT CREATION
# ============================================================================

def create_section_header(title: str, icon: str = "📊") -> None:
    """
    Create a consistent section header.
    
    Args:
        title: Section title
        icon: Icon to display with title
    """
    st.subheader(f"{icon} {title}")

def create_info_box(message: str, box_type: str = "info") -> None:
    """
    Create an information box with consistent styling.
    
    Args:
        message: Message to display
        box_type: Type of box ("info", "warning", "error", "success")
    """
    if box_type == "info":
        st.info(message)
    elif box_type == "warning":
        st.warning(message)
    elif box_type == "error":
        st.error(message)
    elif box_type == "success":
        st.success(message)
    else:
        st.info(message)

def create_footer() -> None:
    """Create dashboard footer with consistent styling."""
    st.markdown("---")
    st.markdown("**Electric Distribution Companies Performance Dashboard**")
    st.markdown("*Enhanced with STD, IQR, and MAD Outlier Detection Methods*")
    st.markdown("*Powered by Streamlit & Plotly*")

def format_percentage_value(value: float, 
                          decimal_places: int = DEFAULT_DECIMAL_DIGITS) -> str:
    """
    Format a value as percentage with Turkish formatting.
    
    Args:
        value: Value to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"%{value * 100:.{decimal_places}f}".replace('.', ',')

def create_loading_spinner(message: str = "Loading...") -> None:
    """
    Create a loading spinner with message.
    
    Args:
        message: Loading message to display
    """
    with st.spinner(message):
        pass

def validate_dataframe_columns(df: pd.DataFrame, 
                             required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all columns exist, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    return True