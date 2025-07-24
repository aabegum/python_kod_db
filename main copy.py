"""
Electric Distribution Companies Performance Dashboard - Main Application

A comprehensive dashboard for analyzing electric distribution company performance
with advanced outlier detection and multi-year trend analysis.
"""

import streamlit as st
import pandas as pd
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modular components
from config.settings import get_settings, validate_and_display_config
from data.loader import (
    load_and_process_standard_data, 
    load_and_process_cumulative_data,
    prepare_shuffled_data
)
from analysis.outliers import OutlierDetector, OutlierStatistics
from visualization.plots import ChartBuilder, create_chart_by_type
from ui.sidebar import SidebarManager, APGFilterManager, render_dashboard_mode_selection
from utils.constants import (
    PAGE_TITLE, 
    PAGE_ICON, 
    LAYOUT, 
    DASHBOARD_TITLE,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    WARNING_MESSAGES,
    COMPANIES_RANGE
)


def initialize_streamlit():
    """Initialize Streamlit page configuration and title."""
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON, 
        layout=LAYOUT
    )
    
    st.title(DASHBOARD_TITLE)
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)


def initialize_application():
    """Initialize application settings and validate configuration."""
    # Load settings
    settings = get_settings()
    
    # Validate configuration
    if not validate_and_display_config(settings):
        st.stop()
    
    return settings


# Replace your create_performance_summary_section function
from ui.components import ExpandableSection

def create_performance_summary_section(summary_data, title):
    ExpandableSection.create_performance_summary_section(summary_data, title)

# Replace your data download sections
from ui.components import DownloadSection

def create_data_download_section(df, year, company_group, selected_apgs, available_apgs, show_outliers, method):
    metadata = {
        'Analysis_Year': year,
        'Company_Group': company_group,
        'Selected_APG_Count': len(selected_apgs),
        'Total_APG_Count': len(available_apgs),
        'Outlier_Detection_Method': method if show_outliers else None
    }
    
    base_columns = ["APG No", "APG İsmi", "Birim"] + list(COMPANIES_RANGE)
    outlier_columns = [f'filtered_mean_{method}', f'outlier_count_{method}'] if show_outliers else None
    
    DownloadSection.create_data_download_section(df, metadata, base_columns, outlier_columns)

def run_standard_analysis(settings, sidebar_manager, apg_filter, file_path):
    """Run standard performance analysis mode."""
    # Get sidebar inputs
    company_group, company_list = sidebar_manager.render_company_group_selection()
    selected_year = sidebar_manager.render_year_selection("Standard Performance Analysis")
    
    # Get analysis options
    (show_outliers, detect_outliers_scope, outlier_method_standard, 
     threshold_value, hide_outliers, compare_methods) = sidebar_manager.render_standard_analysis_options()
    
    # Load and process data
    try:
        merged_df, layout_df = load_and_process_standard_data(settings, file_path, selected_year)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    # Get available APGs
    from data.loader import DataLoader
    loader = DataLoader(settings)
    available_apgs = loader.get_available_apgs(merged_df)
    
    # Compute outliers based on scope
    outlier_results = None
    apgs_with_outliers = None
    
    if show_outliers and detect_outliers_scope == "All APGs":
        # Compute outliers before filtering APGs
        shuffled_df_all = prepare_shuffled_data(settings, merged_df, company_list)
        
        # Determine method key
        method_key = 'std' if 'STD' in outlier_method_standard else ('iqr' if 'IQR' in outlier_method_standard else 'mad')
        
        # Compute outliers
        detector = OutlierDetector(settings)
        outlier_results, shuffled_df_all = detector.compute_outliers_for_dataframe(
            shuffled_df_all, 
            company_list, 
            method=method_key, 
            threshold=threshold_value,
            compare_all=compare_methods
        )
        
        # Get results for primary method
        primary_results = outlier_results[method_key]
        apgs_with_outliers = primary_results['apgs_with_outliers']
    
    # APG filtering (right sidebar)
    selected_apgs = apg_filter.create_apg_filter_right_sidebar(
        available_apgs, 
        apgs_with_outliers if show_outliers else None, 
        merged_df
    )
    
    if not selected_apgs:
        return
    
    # Performance warning
    if len(selected_apgs) > 50:
        st.warning(f"⚠️ {len(selected_apgs)} APGs selected. This may affect performance.")
    
    # Filter data to selected APGs
    filtered_df = loader.filter_data_by_apgs(merged_df, selected_apgs)
    
    if len(filtered_df) == 0:
        st.error(ERROR_MESSAGES['NO_DATA_FOR_APGS'])
        return
    
    st.info(f"📊 Displaying analysis for {len(selected_apgs)} APGs in **{company_group}** group for year **{selected_year}**")
    
    # Prepare data for visualization
    shuffled_df = prepare_shuffled_data(settings, filtered_df, company_list)
    
    # Recompute outliers if scope is "Selected APGs Only"
    if show_outliers and detect_outliers_scope == "Selected APGs Only":
        method_key = 'std' if 'STD' in outlier_method_standard else ('iqr' if 'IQR' in outlier_method_standard else 'mad')
        
        detector = OutlierDetector(settings)
        outlier_results, shuffled_df = detector.compute_outliers_for_dataframe(
            shuffled_df, 
            company_list, 
            method=method_key, 
            threshold=threshold_value,
            compare_all=compare_methods
        )
    
    # Create supporting data structures
    transposed = loader.create_transposed_data(shuffled_df)
    category_to_apg_dict, category_to_apg_full_name_dict = loader.create_category_mappings(shuffled_df)
    
    # Display outlier statistics
    if show_outliers and outlier_results:
        OutlierStatistics.display_outlier_statistics(
            outlier_results, 
            outlier_method_standard, 
            threshold_value, 
            detect_outliers_scope, 
            compare_methods
        )
    
    # Performance summary
    from visualization.plots import create_performance_summary_data
    summary_data = create_performance_summary_data(shuffled_df, company_list, selected_apgs, available_apgs)
    create_performance_summary_section(
        summary_data, 
        f"{selected_year} Performance Summary for {company_group} ({len(selected_apgs)} APGs)"
    )
    
    # Visualization section
    st.subheader(f"📈 {selected_year} Performance Comparison - {company_group} Group ({len(selected_apgs)} APGs)")
    
    if len(selected_apgs) != len(available_apgs):
        st.info(f"🎯 Showing {len(selected_apgs)} of {len(available_apgs)} available APGs")
    
    # Create charts
    chart_builder = ChartBuilder(settings)
    method_key = 'std' if 'STD' in outlier_method_standard else ('iqr' if 'IQR' in outlier_method_standard else 'mad')
    
    col1, col2 = st.columns(2)
    
    for idx, (_, row) in enumerate(shuffled_df.iterrows()):
        grafik_tipi = row.get("Grafik_tipi", "standard")
        
        with col1 if idx % 2 == 0 else col2:
            st.subheader(row["APG Full Name"])
            
            # Chart creation arguments
            chart_kwargs = {
                'show_outliers': show_outliers,
                'outlier_method': method_key,
                'threshold': threshold_value,
                'hide_outliers': hide_outliers,
                'compare_methods': compare_methods
            }
            
            # Create chart based on type
            try:
                if grafik_tipi == "standard":
                    fig = chart_builder.plot_standard_enhanced_v2(
                        row, selected_year, company_list, **chart_kwargs
                    )
                elif grafik_tipi == "stacked":
                    fig = chart_builder.plot_stacked_enhanced(
                        row, transposed, category_to_apg_dict, 
                        category_to_apg_full_name_dict, selected_year
                    )
                elif grafik_tipi == "overlayed":
                    fig = chart_builder.plot_overlayed_enhanced(
                        row, transposed, shuffled_df, selected_year,
                        company_list, **chart_kwargs
                    )
                else:
                    # Default to standard
                    fig = chart_builder.plot_standard_enhanced_v2(
                        row, selected_year, company_list, **chart_kwargs
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart for {row['APG Full Name']}: {e}")
    
    # Data table and download
    create_data_download_section(
        shuffled_df, selected_year, company_group, selected_apgs, 
        available_apgs, show_outliers, outlier_method_standard
    )


def run_cumulative_analysis(settings, sidebar_manager, apg_filter, file_path):
    """Run cumulative year-over-year analysis mode."""
    st.header("📈 Year-over-Year Cumulative Performance Analysis")
    
    # Get sidebar inputs
    company_group, company_list = sidebar_manager.render_company_group_selection()
    
    if not settings.cumulative_enabled:
        st.warning(WARNING_MESSAGES['CUMULATIVE_DISABLED'])
        return
    
    st.markdown(f"Compare **{company_group}** performance against market benchmarks across years 2021-2024")
    
    # Get analysis options
    show_cumulative_outliers, outlier_method, threshold_value = sidebar_manager.render_cumulative_analysis_options()
    
    # Load cumulative data
    company_df, benchmark_data = load_and_process_cumulative_data(settings, file_path)
    
    if company_df is None:
        st.error(ERROR_MESSAGES['CUMULATIVE_DATA_ERROR'])
        return
    
    # Debug options
    show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
    
    # Filter data by company group
    from data.loader import DataLoader
    loader = DataLoader(settings)
    excel_company_names = loader.get_excel_company_names(company_list)
    
    if excel_company_names:
        filtered_company_df = company_df[company_df['Company'].isin(excel_company_names)]
        st.info(f"📊 Showing performance for **{company_group}** companies: {', '.join(excel_company_names)}")
        
        if show_debug:
            st.write(f"🔍 Debug: Config companies: {company_list}")
            st.write(f"🔍 Debug: Excel companies: {excel_company_names}")
            st.write(f"🔍 Debug: Available companies: {sorted(company_df['Company'].unique())}")
            st.write(f"🔍 Debug: Filtered data shape: {filtered_company_df.shape}")
    else:
        st.warning(WARNING_MESSAGES['NO_COMPANIES_FOUND'])
        return
    
    if len(filtered_company_df) == 0:
        st.error(f"❌ No data found for companies: {excel_company_names}")
        st.info("Available companies in the dataset:")
        st.write(sorted(company_df['Company'].unique()))
        return
    
    # Get available APGs
    available_apgs = sorted(filtered_company_df['APG No'].unique())
    
    if len(available_apgs) == 0:
        st.error(ERROR_MESSAGES['NO_APGS_FOUND'])
        return
    
    # Create temporary dataframe for APG names
    apg_names_df = filtered_company_df[['APG No', 'APG Name']].drop_duplicates()
    apg_names_df.columns = ['APG No', 'APG İsmi']
    
    # APG filtering (right sidebar)
    selected_apgs = apg_filter.create_apg_filter_right_sidebar(available_apgs, merged_df=apg_names_df)
    
    if not selected_apgs:
        return
    
    # Performance warning
    if len(selected_apgs) > 50:
        st.warning(f"⚠️ {len(selected_apgs)} APGs selected. This may affect performance.")
    
    final_filtered_df = filtered_company_df[filtered_company_df['APG No'].isin(selected_apgs)]
    
    st.info(f"📈 Displaying {len(selected_apgs)} charts - each showing **{len(excel_company_names)} companies** from {company_group} group")
    
    if len(selected_apgs) != len(available_apgs):
        st.info(f"🎯 Showing {len(selected_apgs)} of {len(available_apgs)} available APGs for cumulative analysis")
    
    # Outlier statistics for cumulative mode
    if show_cumulative_outliers:
        all_performance_values = []
        for year in settings.cumulative_years:
            year_values = final_filtered_df[year].dropna().values
            all_performance_values.extend(year_values)
        
        if len(all_performance_values) > 0:
            from analysis.outliers import detect_outliers_std, detect_outliers_iqr, detect_outliers_mad
            
            method_key = 'std' if 'STD' in outlier_method else ('iqr' if 'IQR' in outlier_method else 'mad')
            
            if method_key == 'std':
                outliers, non_outliers = detect_outliers_std(all_performance_values, sigma=threshold_value)
            elif method_key == 'iqr':
                outliers, non_outliers = detect_outliers_iqr(all_performance_values, k=threshold_value)
            else:  # mad
                outliers, non_outliers = detect_outliers_mad(all_performance_values, threshold=threshold_value)
            
            outlier_percentage = (len(outliers) / len(all_performance_values)) * 100 if all_performance_values else 0
            
            with st.expander(f"🔍 Outlier Analysis Summary ({outlier_method})", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data Points", len(all_performance_values))
                    st.metric("Outliers Detected", len(outliers))
                
                with col2:
                    st.metric("Outlier Percentage", f"{outlier_percentage:.1f}%")
                    threshold_label = {
                        'std': f"σ = {threshold_value}",
                        'iqr': f"IQR Factor = {threshold_value}",
                        'mad': f"MAD Threshold = {threshold_value}"
                    }[method_key]
                    st.metric("Threshold", threshold_label)
                
                with col3:
                    if len(outliers) > 0:
                        st.metric("Outlier Range", f"{min(outliers):.1f}% to {max(outliers):.1f}%")
                    if len(non_outliers) > 0:
                        st.metric("Normal Range", f"{min(non_outliers):.1f}% to {max(non_outliers):.1f}%")
    
    # Performance summary
    from visualization.plots import create_cumulative_summary_data
    summary_data = create_cumulative_summary_data(
        final_filtered_df, excel_company_names, selected_apgs, 
        available_apgs, settings.cumulative_years
    )
    create_performance_summary_section(
        summary_data,
        f"Cumulative Performance Summary for {company_group} ({len(selected_apgs)} APGs)"
    )
    
    # Display charts
    chart_builder = ChartBuilder(settings)
    method_key = 'std' if 'STD' in outlier_method else ('iqr' if 'IQR' in outlier_method else 'mad')
    
    col1, col2 = st.columns(2)
    
    for idx, apg_no in enumerate(selected_apgs):
        apg_data = final_filtered_df[final_filtered_df['APG No'] == apg_no]
        
        if len(apg_data) == 0:
            continue
            
        with col1 if idx % 2 == 0 else col2:
            try:
                fig = chart_builder.plot_cumulative_chart_for_company_group(
                    apg_data, 
                    benchmark_data, 
                    company_group, 
                    company_list,
                    show_outliers=show_cumulative_outliers,
                    outlier_method=method_key,
                    threshold=threshold_value
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cumulative chart for APG {apg_no}: {e}")
    
    # Data table and download for cumulative mode
    with st.expander("📋 View Cumulative Data"):
        display_columns = ['APG No', 'APG Name', 'Company'] + settings.cumulative_years
        
        st.subheader(f"📊 {company_group} Group Performance Summary")
        
        # Create pivot table
        summary_pivot = final_filtered_df.pivot_table(
            index=['APG No', 'APG Name'], 
            columns='Company', 
            values=settings.cumulative_years,
            aggfunc='first'
        )
        
        summary_pivot.columns = [f"{col[1]}_{col[0]}" for col in summary_pivot.columns]
        
        # Display with styling
        if show_cumulative_outliers and 'outliers' in locals():
            st.info("🔍 Cells highlighted in red indicate outlier values")
            
            def highlight_outliers(val):
                if pd.isna(val) or 'outliers' not in locals():
                    return ''
                if val in outliers:
                    return 'background-color: #ffcccc; font-weight: bold'
                return ''
            
            st.dataframe(
                summary_pivot.style.applymap(highlight_outliers).background_gradient(cmap="RdYlGn", axis=1, vmin=-10, vmax=10),
                use_container_width=True
            )
        else:
            st.dataframe(
                summary_pivot.style.background_gradient(cmap="RdYlGn", axis=1),
                use_container_width=True
            )
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Detailed download
            download_df = final_filtered_df[display_columns].copy()
            download_df['Analysis_Mode'] = 'Cumulative'
            download_df['Company_Group'] = company_group
            download_df['Selected_APG_Count'] = len(selected_apgs)
            download_df['Total_APG_Count'] = len(available_apgs)
            
            if show_cumulative_outliers and 'outliers' in locals():
                download_df['Outlier_Method'] = outlier_method
                download_df['Threshold_Value'] = threshold_value
                
                for year in settings.cumulative_years:
                    download_df[f"{year}_IsOutlier"] = download_df[year].apply(
                        lambda x: "Yes" if not pd.isna(x) and x in outliers else "No"
                    )
            
            csv_detailed = download_df.to_csv(index=False).encode('utf-8')
            filename_suffix = "AllAPGs" if len(selected_apgs) == len(available_apgs) else f"{len(selected_apgs)}APGs"
            method_suffix = f"_{method_key}" if show_cumulative_outliers else ""
            
            st.download_button(
                "📥 Download Detailed Data",
                data=csv_detailed,
                file_name=f"cumulative_detailed_{company_group}_{filename_suffix}{method_suffix}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary download
            csv_summary = summary_pivot.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Summary Pivot",
                data=csv_summary,
                file_name=f"cumulative_summary_{company_group}_{filename_suffix}.csv",
                mime="text/csv"
            )


def main():
    """Main application entry point."""
    # Initialize Streamlit
    initialize_streamlit()
    
    # Initialize application
    settings = initialize_application()
    
    # Initialize managers
    sidebar_manager = SidebarManager(settings)
    apg_filter = APGFilterManager(settings)
    
    # Dashboard mode selection
    dashboard_mode = render_dashboard_mode_selection()
    
    # File uploader
    uploaded_file = sidebar_manager.render_file_uploader()
    
    # Determine file path
    if uploaded_file is not None:
        file_path = uploaded_file
    else:
        file_path = os.path.join(os.path.dirname(__file__), settings.master_file)
        if not os.path.exists(file_path):
            st.error(f"Default data file {settings.master_file} not found.")
            st.stop()
    
    # Run analysis based on selected mode
    try:
        if dashboard_mode == "Standard Performance Analysis":
            run_standard_analysis(settings, sidebar_manager, apg_filter, file_path)
        elif dashboard_mode == "Cumulative Year-over-Year Analysis":
            run_cumulative_analysis(settings, sidebar_manager, apg_filter, file_path)
    except Exception as e:
        st.error(f"Application error: {e}")
        if st.sidebar.checkbox("Show error details", value=False):
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("**Electric Distribution Companies Performance Dashboard**")
    st.markdown("*Enhanced with STD, IQR, and MAD Outlier Detection Methods*")
    st.markdown("*Powered by Streamlit & Plotly*")


if __name__ == "__main__":
    main()