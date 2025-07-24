"""
Sidebar UI components for the Electric Distribution Dashboard.

This module handles all sidebar functionality including company group selection,
analysis options, APG filtering, and configuration controls.
"""

import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any

from utils.constants import (
    SESSION_STATE_KEYS,
    PERFORMANCE_WARNING_THRESHOLD,
    PERFORMANCE_WARNING_TEXT,
    FILTER_LABELS,
    ANALYSIS_LABELS,
    WARNING_MESSAGES,
    OUTLIER_METHODS,
    OUTLIER_METHOD_LABELS,
    AVAILABLE_YEARS
)


def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


class SidebarManager:
    """
    Manages all sidebar components and interactions.
    
    Handles company selection, analysis options, APG filtering,
    and configuration management.
    """
    
    def __init__(self, settings):
        """
        Initialize SidebarManager.
        
        Args:
            settings: DashboardSettings instance
        """
        self.settings = settings
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        for key in SESSION_STATE_KEYS.values():
            if key not in st.session_state:
                if key == SESSION_STATE_KEYS['QUICK_SELECT_APGS']:
                    st.session_state[key] = []
                elif key == SESSION_STATE_KEYS['QUICK_SELECT_APGS_CUMULATIVE']:
                    st.session_state[key] = []
                elif key == SESSION_STATE_KEYS['SELECTED_CATEGORIES']:
                    st.session_state[key] = []
                elif key == SESSION_STATE_KEYS['SELECTED_SUBCATEGORIES']:
                    st.session_state[key] = []
                elif key == SESSION_STATE_KEYS['PREV_CATEGORIES']:
                    st.session_state[key] = []
                elif key == SESSION_STATE_KEYS['PREV_SUBCATEGORIES']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
    
    def render_file_uploader(self) -> Optional[object]:
        """
        Render file uploader section.
        
        Returns:
            Uploaded file object or None
        """
        fl = st.file_uploader(":file_folder: Upload a file", type=["xlsx", "csv"])
        if fl is not None:
            filename = fl.name
            st.write(f"Uploaded file: {filename}")
        return fl
    
    def render_company_group_selection(self) -> Tuple[str, List[str]]:
        """
        Render company group selection section.
        
        Returns:
            Tuple of (selected_group, company_list)
        """
        st.sidebar.header("Choose your filter:")
        
        company_groups = self.settings.company_groups
        company_group = st.sidebar.selectbox("Select Company Group", list(company_groups.keys()))
        company_list = company_groups.get(company_group, [])
        
        # Store in session state
        st.session_state[SESSION_STATE_KEYS['COMPANY_GROUP']] = company_group
        
        # Show warning if group is excluded
        if self.settings.is_group_excluded(company_group):
            st.sidebar.warning(WARNING_MESSAGES['GROUP_EXCLUDED'].format(company_group))
        
        return company_group, company_list
    
    def render_year_selection(self, dashboard_mode: str) -> int:
        """
        Render year selection for standard analysis.
        
        Args:
            dashboard_mode: Current dashboard mode
            
        Returns:
            Selected year
        """
        if dashboard_mode == "Standard Performance Analysis":
            st.sidebar.markdown("---")
            st.sidebar.subheader(ANALYSIS_LABELS['YEAR_SELECTION'])
            
            available_years = AVAILABLE_YEARS
            default_index = available_years.index(2024) if 2024 in available_years else len(available_years)-1
            
            selected_year = st.sidebar.selectbox(
                "Select Year for Analysis:",
                options=available_years,
                index=default_index
            )
            
            return selected_year
        
        return self.settings.report_year
    
    def render_standard_analysis_options(self) -> Tuple[bool, str, str, float, bool, bool]:
        """
        Render analysis options for standard mode.
        
        Returns:
            Tuple of (show_outliers, detect_outliers_scope, outlier_method_standard, 
                     threshold_value, hide_outliers, compare_methods)
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader(ANALYSIS_LABELS['STANDARD_TITLE'])
        show_outliers = st.sidebar.checkbox("Show outlier analysis", value=True)
        
        if show_outliers:
            with st.sidebar.expander("Advanced Outlier Options", expanded=False):
                # Outlier detection scope
                detect_outliers_scope = st.radio(
                    ANALYSIS_LABELS['OUTLIER_SCOPE'],
                    ["All APGs", "Selected APGs Only"],
                    help="Choose whether to detect outliers across all data or just selected APGs",
                    key="outlier_scope"
                )
                
                # Outlier detection method
                outlier_method_standard = st.radio(
                    ANALYSIS_LABELS['OUTLIER_METHOD'],
                    [
                        OUTLIER_METHOD_LABELS['std'],
                        OUTLIER_METHOD_LABELS['iqr'], 
                        OUTLIER_METHOD_LABELS['mad']
                    ],
                    index=0,
                    key="standard_outlier_method",
                    help="STD: Mean ± sigma * std\nIQR: Q1/Q3 ± factor * IQR\nMAD: Modified Z-score with threshold"
                )
                
                # Method-specific parameters
                if outlier_method_standard == OUTLIER_METHOD_LABELS['std']:
                    threshold_value = st.slider(
                        "Sigma Value:",
                        min_value=1.0,
                        max_value=3.0,
                        value=float(self.settings.sigma),
                        step=0.1,
                        key="std_sigma",
                        help="Number of standard deviations from mean"
                    )
                elif outlier_method_standard == OUTLIER_METHOD_LABELS['iqr']:
                    threshold_value = st.slider(
                        "IQR Factor:",
                        min_value=1.0,
                        max_value=3.0,
                        value=float(self.settings.iqr_factor),
                        step=0.1,
                        key="iqr_factor",
                        help="Multiplier for IQR bounds"
                    )
                else:  # MAD
                    threshold_value = st.slider(
                        "MAD Threshold:",
                        min_value=2.0,
                        max_value=5.0,
                        value=float(self.settings.mad_threshold),
                        step=0.1,
                        key="mad_threshold",
                        help="Threshold for modified Z-score"
                    )
                
                hide_outliers = st.checkbox("Hide outliers from charts", value=False)
                
                # Multi-method comparison option
                compare_methods = st.checkbox(
                    "Compare All Methods",
                    value=False,
                    help="Show comparison of STD, IQR, and MAD methods"
                )
        else:
            outlier_method_standard = OUTLIER_METHOD_LABELS['std']
            threshold_value = float(self.settings.sigma)
            hide_outliers = False
            detect_outliers_scope = "All APGs"
            compare_methods = False
        
        return show_outliers, detect_outliers_scope, outlier_method_standard, threshold_value, hide_outliers, compare_methods
    
    def render_cumulative_analysis_options(self) -> Tuple[bool, str, float]:
        """
        Render analysis options for cumulative mode.
        
        Returns:
            Tuple of (show_cumulative_outliers, outlier_method, threshold_value)
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader(ANALYSIS_LABELS['CUMULATIVE_TITLE'])
        
        show_cumulative_outliers = st.sidebar.checkbox(
            "Show Outlier Analysis", 
            value=True, 
            key="cumulative_outliers_checkbox"
        )
        
        if show_cumulative_outliers:
            with st.sidebar.expander("Advanced Outlier Options", expanded=False):
                outlier_method = st.radio(
                    ANALYSIS_LABELS['OUTLIER_METHOD'],
                    [
                        OUTLIER_METHOD_LABELS['std'],
                        OUTLIER_METHOD_LABELS['iqr'], 
                        OUTLIER_METHOD_LABELS['mad']
                    ],
                    index=1,  # Default to IQR for cumulative
                    key="cumulative_outlier_method",
                    help="STD: Mean ± sigma * std\nIQR: Q1/Q3 ± factor * IQR\nMAD: Modified Z-score with threshold"
                )
                
                if outlier_method == OUTLIER_METHOD_LABELS['std']:
                    threshold_value = st.slider(
                        "Sigma Value:",
                        min_value=1.0,
                        max_value=3.0,
                        value=float(self.settings.sigma),
                        step=0.1,
                        key="cumulative_std_sigma",
                        help="Number of standard deviations from mean"
                    )
                elif outlier_method == OUTLIER_METHOD_LABELS['iqr']:
                    threshold_value = st.slider(
                        "IQR Factor:",
                        min_value=1.0,
                        max_value=3.0,
                        value=float(self.settings.iqr_factor),
                        step=0.1,
                        key="cumulative_iqr_factor",
                        help="Multiplier for IQR bounds"
                    )
                else:  # MAD
                    threshold_value = st.slider(
                        "MAD Threshold:",
                        min_value=2.0,
                        max_value=5.0,
                        value=float(self.settings.mad_threshold),
                        step=0.1,
                        key="cumulative_mad_threshold",
                        help="Threshold for modified Z-score"
                    )
        else:
            outlier_method = OUTLIER_METHOD_LABELS['iqr']
            threshold_value = float(self.settings.iqr_factor)
        
        return show_cumulative_outliers, outlier_method, threshold_value


class APGFilterManager:
    """
    Manages APG filtering functionality with hierarchical category selection.
    """
    
    def __init__(self, settings):
        """
        Initialize APGFilterManager.
        
        Args:
            settings: DashboardSettings instance
        """
        self.settings = settings
    
    def create_apg_filter_right_sidebar(self, available_apgs: List[str], 
                                       apgs_with_outliers: Optional[List[str]] = None, 
                                       merged_df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Enhanced APG filtering with hierarchical category selection for RIGHT SIDEBAR.
        
        Args:
            available_apgs: List of available APG numbers
            apgs_with_outliers: List of APGs with outliers (optional)
            merged_df: DataFrame containing APG data (optional)
            
        Returns:
            List of selected APG numbers
        """
        with st.sidebar:
            st.markdown("---")
            st.subheader("🎯 APG Selection")
            
            if st.button("Reset Filters"):
                reset_apg_selection()
                st.rerun()
            
            # Combined expandable section for all APG filters
            with st.expander("APG Filters", expanded=True):
                categories = sorted(
                    set([
                        re.match(r'([A-Z]+)', apg).group(1) 
                        for apg in available_apgs if re.match(r'([A-Z]+)', apg)
                    ]), 
                    key=natural_key
                )
                
                if not categories:
                    st.warning(WARNING_MESSAGES['NO_CATEGORIES'])
                    return []
                
                # Category selection (as dropdown - single select)
                selected_category = st.selectbox(
                    "Category:",
                    options=["All"] + categories,
                    index=0,
                    help="Select a top-level category (e.g., A, B, C)"
                )
                
                if selected_category == "All":
                    filtered_apgs = available_apgs
                    selected_categories = categories
                else:
                    selected_categories = [selected_category]
                    filtered_apgs = [apg for apg in available_apgs if re.match(r'([A-Z]+)', apg).group(1) in selected_categories]
                
                # Subcategory selection (conditional)
                if selected_category != "All":
                    subcategories = sorted(
                        set([
                            re.match(r'([A-Z]+\d+)', apg).group(1) 
                            for apg in filtered_apgs if re.match(r'([A-Z]+\d+)', apg)
                        ]), 
                        key=natural_key
                    )
                    
                    selected_subcategory = st.selectbox(
                        "Subcategory:",
                        options=["All"] + subcategories,
                        index=0,
                        help="Select a subcategory (e.g., A1, A2, B1)"
                    )
                    
                    if selected_subcategory == "All":
                        selected_subcategories = subcategories
                    else:
                        selected_subcategories = [selected_subcategory]
                        filtered_apgs = [apg for apg in filtered_apgs if re.match(r'([A-Z]+\d+)', apg).group(1) in selected_subcategories]
                    
                    # APG selection (conditional on subcategory)
                    if selected_subcategory != "All":
                        sorted_apgs = sorted(filtered_apgs, key=natural_key)
                        selected_apg = st.selectbox(
                            "APG:",
                            options=["All"] + sorted_apgs,
                            index=0,
                            help="Select a specific APG (e.g., A1.1)"
                        )
                        
                        if selected_apg == "All":
                            selected_apgs = filtered_apgs
                        else:
                            selected_apgs = [selected_apg]
                    else:
                        selected_apgs = filtered_apgs
                else:
                    selected_apgs = filtered_apgs
                
                # Outlier filter
                show_only_outlier_apgs = False
                if apgs_with_outliers is not None:
                    show_only_outlier_apgs = st.checkbox(
                        "Only Outlier APGs",
                        value=False,
                        key="show_only_outlier_apgs",
                        help="Show only APGs with outliers"
                    )
                    if show_only_outlier_apgs:
                        filtered_apgs = [apg for apg in filtered_apgs if apg in apgs_with_outliers]
                        selected_apgs = filtered_apgs
                
                # Search (always available)
                apg_search = st.text_input(
                    "🔍 Search APGs:",
                    placeholder="e.g., A1.1 or APG name",
                    key="apg_search",
                    help="Filter by APG number or name"
                )
                
                if apg_search and merged_df is not None:
                    filtered_apgs = [
                        apg for apg in filtered_apgs 
                        if apg_search.upper() in apg.upper() or 
                           apg_search.upper() in self._get_apg_name_from_df(apg, merged_df).upper()
                    ]
                    selected_apgs = filtered_apgs
                elif apg_search:
                    filtered_apgs = [apg for apg in filtered_apgs if apg_search.upper() in apg.upper()]
                    selected_apgs = filtered_apgs
                
                # Performance warning if applicable
                if len(selected_apgs) > PERFORMANCE_WARNING_THRESHOLD:
                    st.warning(PERFORMANCE_WARNING_TEXT.format(len(selected_apgs)))
            
            if not selected_apgs:
                st.warning("⚠️ Select at least one APG.")
                return []
            
            return selected_apgs
    
    def _get_apg_name_from_df(self, apg_no: str, df: pd.DataFrame) -> str:
        """Get APG name from dataframe."""
        apg_rows = df[df['APG No'] == apg_no]
        if len(apg_rows) > 0:
            return apg_rows.iloc[0]['APG İsmi']
        return ""


# ============================================================================
# UTILITY FUNCTIONS FOR SIDEBAR COMPONENTS
# ============================================================================

def render_dashboard_mode_selection() -> str:
    """
    Render dashboard mode selection radio button.
    
    Returns:
        Selected dashboard mode
    """
    dashboard_mode = st.sidebar.radio(
        "Select Dashboard Mode:",
        ["Standard Performance Analysis", "Cumulative Year-over-Year Analysis"],
        index=0
    )
    return dashboard_mode

def render_debug_options(settings) -> bool:
    """
    Render debug options if enabled.
    
    Args:
        settings: DashboardSettings instance
        
    Returns:
        Whether debug mode is enabled
    """
    if hasattr(settings, 'debug_mode') and settings.debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🐛 Debug Options")
        show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
        return show_debug
    return False

def create_sidebar_state_summary() -> Dict[str, Any]:
    """
    Create a summary of current sidebar state for debugging.
    
    Returns:
        Dictionary containing sidebar state information
    """
    state_summary = {}
    
    for key_name, session_key in SESSION_STATE_KEYS.items():
        if session_key in st.session_state:
            value = st.session_state[session_key]
            if isinstance(value, list):
                state_summary[key_name] = f"List with {len(value)} items"
            else:
                state_summary[key_name] = str(value)[:50]  # Truncate long values
        else:
            state_summary[key_name] = "Not set"
    
    return state_summary

def reset_apg_selection():
    """Reset APG selection in session state."""
    keys_to_reset = [
        SESSION_STATE_KEYS['QUICK_SELECT_APGS'],
        SESSION_STATE_KEYS['QUICK_SELECT_APGS_CUMULATIVE'],
        SESSION_STATE_KEYS['SELECTED_CATEGORIES'],
        SESSION_STATE_KEYS['SELECTED_SUBCATEGORIES']
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = []

def validate_apg_selection(selected_apgs: List[str], available_apgs: List[str]) -> bool:
    """
    Validate that selected APGs are valid.
    
    Args:
        selected_apgs: List of selected APG numbers
        available_apgs: List of available APG numbers
        
    Returns:
        True if selection is valid, False otherwise
    """
    if not selected_apgs:
        return False
    
    invalid_apgs = [apg for apg in selected_apgs if apg not in available_apgs]
    if invalid_apgs:
        st.error(f"Invalid APGs selected: {invalid_apgs}")
        return False
    
    return True