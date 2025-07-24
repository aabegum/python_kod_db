# settings.py
"""
Configuration settings management for the Electric Distribution Dashboard.

This module handles loading and validating configuration from YAML files,
merging with default constants, and providing a unified settings interface.
"""

import os
import yaml
import streamlit as st
from typing import Dict, Any, List, Optional
from matplotlib.colors import ListedColormap

from utils.constants import (
    CONFIG_FILENAME,
    DEFAULT_MASTER_FILE,
    DEFAULT_REPORT_YEAR,
    DEFAULT_REPORT_TYPE,
    DEFAULT_SIGMA,
    DEFAULT_IQR_FACTOR,
    DEFAULT_MAD_THRESHOLD,
    DEFAULT_DECIMAL_DIGITS,
    DEFAULT_ANNOTATION_FONT_SIZE,
    DEFAULT_HORIZONTAL_MEAN_COLOR,
    DEFAULT_HORIZONTAL_MEAN_ALPHA,
    DEFAULT_OVERLAY_GRAPH_BAR_COLOR,
    DEFAULT_OVERLAY_GRAPH_COLOR_MAP,
    DEFAULT_CHART_HEIGHT,
    DEFAULT_CUMULATIVE_ENABLED,
    DEFAULT_CUMULATIVE_SHEET,
    DEFAULT_CUMULATIVE_YEARS,
    ERROR_MESSAGES
)


class DashboardSettings:
    """
    Configuration settings manager for the dashboard application.
    
    Loads configuration from YAML file and provides access to all settings
    with proper defaults and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings manager.
        
        Args:
            config_path: Path to configuration file. If None, looks for config.yaml
                        in the current directory.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config_data = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        coding_directory = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(coding_directory, CONFIG_FILENAME)
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with error handling."""
        try:
            with open(self.config_path, encoding='utf-8') as config_file:
                self.config_data = yaml.safe_load(config_file) or {}
        except FileNotFoundError:
            st.error(ERROR_MESSAGES['CONFIG_NOT_FOUND'])
            st.stop()
        except yaml.YAMLError as e:
            st.error(f"Error parsing configuration file: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading configuration: {e}")
            st.stop()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config_data.get(key, default)
    
    # ============================================================================
    # FILE PATHS AND BASIC CONFIGURATION
    # ============================================================================
    
    @property
    def master_file(self) -> str:
        """Get master data file path."""
        return self.get('MASTER_FILE', DEFAULT_MASTER_FILE)
    
    @property
    def template_path(self) -> str:
        """Get PowerPoint template path."""
        return self.get('TEMPLATE_PATH', '')
    
    @property
    def report_year(self) -> int:
        """Get report year."""
        year = self.get('REPORT_YEAR', DEFAULT_REPORT_YEAR)
        # Handle string format like "2024_2"
        if isinstance(year, str):
            return int(year.split('_')[0])
        return year
    
    @property
    def report_type(self) -> str:
        """Get report type."""
        return self.get('REPORT_TYPE', DEFAULT_REPORT_TYPE)
    
    # ============================================================================
    # COMPANY GROUPS CONFIGURATION
    # ============================================================================
    
    @property
    def company_groups(self) -> Dict[str, List[str]]:
        """Get company groups mapping."""
        return self.get('COMPANY_GROUPS', {})
    
    @property
    def company_groups_excluded(self) -> List[str]:
        """Get list of excluded company groups."""
        return self.get('COMPANY_GROUPS_EXCLUDED_FROM_REPORT', [])
    
    def get_company_list(self, group_name: str) -> List[str]:
        """
        Get list of companies for a specific group.
        
        Args:
            group_name: Name of the company group
            
        Returns:
            List of company names in the group
        """
        return self.company_groups.get(group_name, [])
    
    def is_group_excluded(self, group_name: str) -> bool:
        """
        Check if a company group is excluded from reports.
        
        Args:
            group_name: Name of the company group
            
        Returns:
            True if group is excluded, False otherwise
        """
        return group_name in self.company_groups_excluded
    
    # ============================================================================
    # OUTLIER DETECTION PARAMETERS
    # ============================================================================
    
    @property
    def sigma(self) -> float:
        """Get sigma value for standard deviation outlier detection."""
        return float(self.get('SIGMA', DEFAULT_SIGMA))
    
    @property
    def iqr_factor(self) -> float:
        """Get IQR factor for interquartile range outlier detection."""
        return float(self.get('IQR_FACTOR', DEFAULT_IQR_FACTOR))
    
    @property
    def mad_threshold(self) -> float:
        """Get MAD threshold for median absolute deviation outlier detection."""
        return float(self.get('MAD_THRESHOLD', DEFAULT_MAD_THRESHOLD))
    
    # ============================================================================
    # DISPLAY AND FORMATTING SETTINGS
    # ============================================================================
    
    @property
    def decimal_digits(self) -> int:
        """Get number of decimal digits for display."""
        return int(self.get('DEFAULT_DECIMAL_DIGITS', DEFAULT_DECIMAL_DIGITS))
    
    @property
    def annotation_font_size(self) -> int:
        """Get font size for chart annotations."""
        return int(self.get('ANNOTATION_FONT_SIZE', DEFAULT_ANNOTATION_FONT_SIZE))
    
    @property
    def word_wrap_limit(self) -> int:
        """Get word wrap limit for chart titles."""
        return int(self.get('WORD_WRAP_LIMIT', 50))
    
    @property
    def font_size(self) -> float:
        """Get general font size."""
        return float(self.get('FONT_SIZE', 10.5))
    
    # ============================================================================
    # CHART APPEARANCE SETTINGS
    # ============================================================================
    
    @property
    def horizontal_mean_color(self) -> str:
        """Get color for horizontal mean lines."""
        return self.get('HORIZONTAL_MEAN_COLOR', DEFAULT_HORIZONTAL_MEAN_COLOR)
    
    @property
    def horizontal_mean_alpha(self) -> float:
        """Get alpha (transparency) for horizontal mean lines."""
        return float(self.get('HORIZONTAL_MEAN_ALPHA', DEFAULT_HORIZONTAL_MEAN_ALPHA))
    
    @property
    def overlay_graph_bar_color(self) -> str:
        """Get bar color for overlay graphs."""
        return self.get('OVERLAY_GRAPH_BAR_COLOR', DEFAULT_OVERLAY_GRAPH_BAR_COLOR)
    
    @property
    def overlay_graph_color_map(self) -> ListedColormap:
        """Get color map for overlay graphs."""
        colors = self.get('OVERLAY_GRAPH_COLOR_MAP', DEFAULT_OVERLAY_GRAPH_COLOR_MAP)
        return ListedColormap(colors)
    
    @property
    def annotation_offset_pixels(self) -> List[int]:
        """Get annotation offset in pixels."""
        return self.get('ANNOTATION_OFFSET_PIXELS', [-5, 5])
    
    # ============================================================================
    # CUMULATIVE CHARTS CONFIGURATION
    # ============================================================================
    
    @property
    def cumulative_config(self) -> Dict[str, Any]:
        """Get cumulative charts configuration."""
        return self.get('CUMULATIVE_CHARTS', {})
    
    @property
    def cumulative_enabled(self) -> bool:
        """Check if cumulative charts are enabled."""
        return self.cumulative_config.get('ENABLED', DEFAULT_CUMULATIVE_ENABLED)
    
    @property
    def cumulative_sheet(self) -> str:
        """Get cumulative data sheet name."""
        return self.cumulative_config.get('SHEET_NAME', DEFAULT_CUMULATIVE_SHEET)
    
    @property
    def cumulative_years(self) -> List[str]:
        """Get list of years for cumulative analysis."""
        return self.cumulative_config.get('YEARS', DEFAULT_CUMULATIVE_YEARS)
    
    @property
    def cumulative_default_apgs(self) -> int:
        """Get default number of APGs to show in cumulative analysis."""
        return self.cumulative_config.get('DEFAULT_APGS_TO_SHOW', 10)
    
    @property
    def cumulative_chart_height(self) -> int:
        """Get chart height for cumulative charts."""
        return self.cumulative_config.get('CHART_HEIGHT', DEFAULT_CHART_HEIGHT)
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the loaded configuration and return any issues found.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required fields
        if not self.company_groups:
            errors.append("COMPANY_GROUPS configuration is missing or empty")
        
        # Validate numeric values
        try:
            float(self.sigma)
            if self.sigma <= 0:
                errors.append("SIGMA must be a positive number")
        except (ValueError, TypeError):
            errors.append("SIGMA must be a valid number")
        
        try:
            float(self.iqr_factor)
            if self.iqr_factor <= 0:
                errors.append("IQR_FACTOR must be a positive number")
        except (ValueError, TypeError):
            errors.append("IQR_FACTOR must be a valid number")
        
        try:
            float(self.mad_threshold)
            if self.mad_threshold <= 0:
                errors.append("MAD_THRESHOLD must be a positive number")
        except (ValueError, TypeError):
            errors.append("MAD_THRESHOLD must be a valid number")
        
        # Validate file paths
        if self.master_file and not self.master_file.endswith(('.xlsx', '.csv')):
            errors.append("MASTER_FILE must have .xlsx or .csv extension")
        
        # Validate cumulative years
        for year in self.cumulative_years:
            try:
                int(year.replace('_', '').replace('2', ''))  # Handle formats like "20242"
            except ValueError:
                errors.append(f"Invalid year format in CUMULATIVE_CHARTS.YEARS: {year}")
        
        return errors
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_sheet_name(self, year: int) -> str:
        """
        Get sheet name for a specific year.
        
        Args:
            year: Year for the sheet
            
        Returns:
            Sheet name for the year
        """
        return f"{year}_Total_Veriler"
    
    def get_all_company_names(self) -> List[str]:
        """
        Get all unique company names from all groups.
        
        Returns:
            List of all company names
        """
        all_companies = []
        for companies in self.company_groups.values():
            all_companies.extend(companies)
        return list(set(all_companies))
    
    def get_group_by_company(self, company_name: str) -> Optional[str]:
        """
        Find which group a company belongs to.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Group name if found, None otherwise
        """
        for group_name, companies in self.company_groups.items():
            if company_name in companies:
                return group_name
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all settings to a dictionary for debugging/export.
        
        Returns:
            Dictionary containing all configuration values
        """
        return {
            'master_file': self.master_file,
            'template_path': self.template_path,
            'report_year': self.report_year,
            'report_type': self.report_type,
            'company_groups': self.company_groups,
            'company_groups_excluded': self.company_groups_excluded,
            'sigma': self.sigma,
            'iqr_factor': self.iqr_factor,
            'mad_threshold': self.mad_threshold,
            'decimal_digits': self.decimal_digits,
            'annotation_font_size': self.annotation_font_size,
            'word_wrap_limit': self.word_wrap_limit,
            'font_size': self.font_size,
            'horizontal_mean_color': self.horizontal_mean_color,
            'horizontal_mean_alpha': self.horizontal_mean_alpha,
            'overlay_graph_bar_color': self.overlay_graph_bar_color,
            'cumulative_enabled': self.cumulative_enabled,
            'cumulative_sheet': self.cumulative_sheet,
            'cumulative_years': self.cumulative_years,
            'cumulative_default_apgs': self.cumulative_default_apgs,
            'cumulative_chart_height': self.cumulative_chart_height
        }
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return f"DashboardSettings(config_path='{self.config_path}')"


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

# Create a global settings instance that can be imported by other modules
settings = None

def get_settings(config_path: Optional[str] = None) -> DashboardSettings:
    """
    Get the global settings instance, creating it if necessary.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Global settings instance
    """
    global settings
    if settings is None or config_path is not None:
        settings = DashboardSettings(config_path)
    return settings

def reload_settings(config_path: Optional[str] = None) -> DashboardSettings:
    """
    Force reload of settings configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Reloaded settings instance
    """
    global settings
    settings = DashboardSettings(config_path)
    return settings

# ============================================================================
# CONFIGURATION VALIDATION FUNCTION
# ============================================================================

def validate_and_display_config(settings_instance: DashboardSettings) -> bool:
    """
    Validate configuration and display any errors in Streamlit.
    
    Args:
        settings_instance: Settings instance to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = settings_instance.validate_configuration()
    
    if errors:
        st.error("❌ Configuration validation failed:")
        for error in errors:
            st.error(f"• {error}")
        return False
    else:
        return True