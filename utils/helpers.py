"""
Helper utilities for the Electric Distribution Dashboard.

This module contains utility functions for text processing, data formatting,
validation, and other common operations used throughout the application.
"""

import pandas as pd
import numpy as np
import streamlit as st
import re
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

from utils.constants import (
    DEFAULT_DECIMAL_DIGITS,
    DEFAULT_WORD_WRAP_WIDTH,
    PERCENTAGE_CONVERSION_THRESHOLD,
    INVALID_DATA_VALUES,
    APG_GROUP_PATTERN
)


class TextProcessor:
    """
    Text processing and formatting utilities.
    """
    
    @staticmethod
    def wrap_text(text: str, width: int = None) -> str:
        """
        Wrap text to specified width for display purposes.
        
        Args:
            text: Text to wrap
            width: Maximum width in characters
            
        Returns:
            HTML-formatted wrapped text with <br> tags
        """
        if width is None:
            width = DEFAULT_WORD_WRAP_WIDTH
        
        if not text or len(text) <= width:
            return text
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Word is longer than width, force break
                    lines.append(word)
                    current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        return "<br>".join(lines)
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with optional suffix.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text).strip())
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        
        return text
    
    @staticmethod
    def extract_apg_components(apg_no: str) -> Dict[str, str]:
        """
        Extract components from APG number (e.g., "A1.1.1" -> {"category": "A", "subcategory": "A1", "group": "A1.1"}).
        
        Args:
            apg_no: APG number string
            
        Returns:
            Dictionary containing APG components
        """
        if not apg_no or not isinstance(apg_no, str):
            return {"category": "", "subcategory": "", "group": "", "full": ""}
        
        parts = apg_no.split('.')
        
        result = {
            "category": parts[0] if len(parts) > 0 else "",
            "subcategory": '.'.join(parts[:2]) if len(parts) > 1 else "",
            "group": '.'.join(parts[:3]) if len(parts) > 2 else "",
            "full": apg_no
        }
        
        return result
    
    @staticmethod
    def format_apg_display_name(apg_no: str, apg_name: str, max_name_length: int = 25) -> str:
        """
        Format APG display name with consistent formatting.
        
        Args:
            apg_no: APG number
            apg_name: APG name
            max_name_length: Maximum length for name part
            
        Returns:
            Formatted display name
        """
        if not apg_no:
            return apg_name or ""
        
        if not apg_name:
            return apg_no
        
        truncated_name = TextProcessor.truncate_text(apg_name, max_name_length)
        return f"{apg_no} - {truncated_name}"


class DataFormatter:
    """
    Data formatting and conversion utilities.
    """
    
    @staticmethod
    def format_percentage(value: Union[float, int], 
                         decimal_digits: int = None,
                         use_turkish_format: bool = True) -> str:
        """
        Format value as percentage with localization support.
        
        Args:
            value: Numeric value to format
            decimal_digits: Number of decimal places
            use_turkish_format: Whether to use Turkish number formatting
            
        Returns:
            Formatted percentage string
        """
        if decimal_digits is None:
            decimal_digits = DEFAULT_DECIMAL_DIGITS
        
        if pd.isna(value):
            return "N/A"
        
        try:
            formatted = f"{value * 100:.{decimal_digits}f}"
            if use_turkish_format:
                formatted = formatted.replace('.', ',')
            return f"%{formatted}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_number(value: Union[float, int], 
                     decimal_digits: int = 2,
                     use_turkish_format: bool = True,
                     thousands_separator: bool = True) -> str:
        """
        Format number with localization support.
        
        Args:
            value: Numeric value to format
            decimal_digits: Number of decimal places
            use_turkish_format: Whether to use Turkish number formatting
            thousands_separator: Whether to include thousands separator
            
        Returns:
            Formatted number string
        """
        if pd.isna(value):
            return "N/A"
        
        try:
            if thousands_separator:
                formatted = f"{value:,.{decimal_digits}f}"
            else:
                formatted = f"{value:.{decimal_digits}f}"
            
            if use_turkish_format:
                # Turkish formatting: comma as decimal separator, dot as thousands separator
                formatted = formatted.replace(',', '|').replace('.', ',').replace('|', '.')
            
            return formatted
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def clean_numeric_value(value: Any) -> float:
        """
        Clean and convert value to float, handling various edge cases.
        
        Args:
            value: Value to clean and convert
            
        Returns:
            Cleaned float value or 0 if conversion fails
        """
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        if isinstance(value, str):
            # Clean string value
            cleaned = value.replace('%', '').replace(',', '.').strip()
            
            # Check for invalid indicators
            if cleaned.lower() in INVALID_DATA_VALUES:
                return 0.0
            
            try:
                float_val = float(cleaned)
                # Convert to percentage if needed
                if abs(float_val) <= PERCENTAGE_CONVERSION_THRESHOLD and float_val != 0:
                    float_val *= 100
                return float_val
            except ValueError:
                return 0.0
        
        return 0.0
    
    @staticmethod
    def format_currency(value: Union[float, int], 
                       currency_symbol: str = "₺",
                       decimal_digits: int = 2) -> str:
        """
        Format value as currency.
        
        Args:
            value: Numeric value
            currency_symbol: Currency symbol to use
            decimal_digits: Number of decimal places
            
        Returns:
            Formatted currency string
        """
        if pd.isna(value):
            return "N/A"
        
        try:
            formatted = DataFormatter.format_number(value, decimal_digits)
            return f"{formatted} {currency_symbol}"
        except:
            return "N/A"


class DataValidator:
    """
    Data validation and quality checking utilities.
    """
    
    @staticmethod
    def validate_dataframe_structure(df: pd.DataFrame, 
                                   required_columns: List[str],
                                   min_rows: int = 1) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if df is None:
            errors.append("DataFrame is None")
            return False, errors
        
        if len(df) < min_rows:
            errors.append(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_apg_numbers(apg_list: List[str]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate APG number format and structure.
        
        Args:
            apg_list: List of APG numbers to validate
            
        Returns:
            Tuple of (is_valid, valid_apgs, invalid_apgs)
        """
        valid_apgs = []
        invalid_apgs = []
        
        apg_pattern = re.compile(r'^[A-Z]\d+(\.\d+)*$')
        
        for apg in apg_list:
            if not apg or not isinstance(apg, str):
                invalid_apgs.append(apg)
                continue
            
            if apg_pattern.match(apg.strip()):
                valid_apgs.append(apg.strip())
            else:
                invalid_apgs.append(apg)
        
        return len(invalid_apgs) == 0, valid_apgs, invalid_apgs
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, 
                                numeric_columns: List[str],
                                allow_negative: bool = True) -> Dict[str, List[int]]:
        """
        Validate numeric columns for data quality issues.
        
        Args:
            df: DataFrame to validate
            numeric_columns: List of numeric column names
            allow_negative: Whether negative values are allowed
            
        Returns:
            Dictionary mapping column names to lists of problematic row indices
        """
        issues = {}
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            problematic_rows = []
            
            for idx, value in df[col].items():
                # Check for non-numeric values
                if not pd.isna(value):
                    try:
                        num_val = float(value)
                        if np.isnan(num_val) or np.isinf(num_val):
                            problematic_rows.append(idx)
                        elif not allow_negative and num_val < 0:
                            problematic_rows.append(idx)
                    except (ValueError, TypeError):
                        problematic_rows.append(idx)
            
            if problematic_rows:
                issues[col] = problematic_rows
        
        return issues
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame, 
                               important_columns: List[str]) -> Dict[str, float]:
        """
        Check data completeness for important columns.
        
        Args:
            df: DataFrame to check
            important_columns: List of important column names
            
        Returns:
            Dictionary mapping column names to completeness percentages
        """
        completeness = {}
        
        for col in important_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                completeness[col] = (non_null_count / total_count) * 100 if total_count > 0 else 0
            else:
                completeness[col] = 0
        
        return completeness


class FileHandler:
    """
    File handling and path utilities.
    """
    
    @staticmethod
    def get_safe_filename(filename: str, 
                         max_length: int = 100,
                         invalid_chars: str = r'[<>:"/\\|?*]') -> str:
        """
        Create a safe filename by removing invalid characters.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
            invalid_chars: Regex pattern for invalid characters
            
        Returns:
            Safe filename string
        """
        if not filename:
            return "file"
        
        # Remove invalid characters
        safe_name = re.sub(invalid_chars, '_', filename)
        
        # Remove multiple consecutive underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Truncate if too long
        if len(safe_name) > max_length:
            name_part, ext = os.path.splitext(safe_name)
            safe_name = name_part[:max_length - len(ext)] + ext
        
        return safe_name
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get file extension in lowercase.
        
        Args:
            filename: Filename to process
            
        Returns:
            File extension (e.g., '.xlsx', '.csv')
        """
        if not filename:
            return ""
        
        return os.path.splitext(filename)[1].lower()
    
    @staticmethod
    def is_excel_file(filename: str) -> bool:
        """
        Check if file is an Excel file.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if Excel file, False otherwise
        """
        excel_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
        return FileHandler.get_file_extension(filename) in excel_extensions
    
    @staticmethod
    def is_csv_file(filename: str) -> bool:
        """
        Check if file is a CSV file.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if CSV file, False otherwise
        """
        return FileHandler.get_file_extension(filename) == '.csv'


class SessionStateManager:
    """
    Streamlit session state management utilities.
    """
    
    @staticmethod
    def initialize_session_state(key: str, default_value: Any) -> Any:
        """
        Initialize session state key with default value if not exists.
        
        Args:
            key: Session state key
            default_value: Default value to set
            
        Returns:
            Current value in session state
        """
        if key not in st.session_state:
            st.session_state[key] = default_value
        return st.session_state[key]
    
    @staticmethod
    def update_session_state(key: str, value: Any) -> None:
        """
        Update session state key with new value.
        
        Args:
            key: Session state key
            value: New value to set
        """
        st.session_state[key] = value
    
    @staticmethod
    def get_session_state(key: str, default: Any = None) -> Any:
        """
        Get value from session state with optional default.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def clear_session_state_keys(keys: List[str]) -> None:
        """
        Clear multiple session state keys.
        
        Args:
            keys: List of session state keys to clear
        """
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def get_session_state_summary() -> Dict[str, str]:
        """
        Get summary of current session state for debugging.
        
        Returns:
            Dictionary with session state summary
        """
        summary = {}
        for key, value in st.session_state.items():
            if isinstance(value, (list, dict)):
                summary[key] = f"{type(value).__name__} with {len(value)} items"
            else:
                summary[key] = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        return summary


class PerformanceMonitor:
    """
    Performance monitoring and optimization utilities.
    """
    
    @staticmethod
    def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Time function execution.
        
        Args:
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def memory_usage_check() -> Dict[str, Any]:
        """
        Check current memory usage (if psutil is available).
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            if df_optimized[col].min() >= 0:
                if df_optimized[col].max() < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif df_optimized[col].max() < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif df_optimized[col].max() < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if df_optimized[col].min() > -128 and df_optimized[col].max() < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif df_optimized[col].min() > -32768 and df_optimized[col].max() < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif df_optimized[col].min() > -2147483648 and df_optimized[col].max() < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Optimize string columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized


class DebugHelper:
    """
    Debugging and development utilities.
    """
    
    @staticmethod
    def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Print comprehensive DataFrame information for debugging.
        
        Args:
            df: DataFrame to analyze
            name: Name for the DataFrame
        """
        if st.sidebar.checkbox(f"Show {name} Debug Info", value=False):
            st.subheader(f"🔍 {name} Debug Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Info:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {len(df.columns)}")
                st.write(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
            
            with col2:
                st.write("**Data Types:**")
                st.write(df.dtypes.value_counts())
            
            st.write("**Column List:**")
            st.write(list(df.columns))
            
            st.write("**Sample Data:**")
            st.write(df.head())
            
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            if missing_data.any():
                st.write(missing_data[missing_data > 0])
            else:
                st.write("No missing values")
    
    @staticmethod
    def log_performance_metrics(operation: str, execution_time: float, 
                               data_size: int = None) -> None:
        """
        Log performance metrics for debugging.
        
        Args:
            operation: Name of the operation
            execution_time: Execution time in seconds
            data_size: Size of data processed (optional)
        """
        if st.sidebar.checkbox("Show Performance Metrics", value=False):
            st.write(f"⏱️ {operation}: {execution_time:.3f}s")
            if data_size:
                st.write(f"📊 Data size: {data_size} items")
                st.write(f"🚀 Processing rate: {data_size/execution_time:.0f} items/sec")
    
    @staticmethod
    def create_debug_expander(title: str, content: Any) -> None:
        """
        Create a debug expander with formatted content.
        
        Args:
            title: Title for the expander
            content: Content to display
        """
        with st.expander(f"🐛 Debug: {title}", expanded=False):
            if isinstance(content, dict):
                st.json(content)
            elif isinstance(content, pd.DataFrame):
                st.dataframe(content)
            elif isinstance(content, (list, tuple)):
                st.write(content)
            else:
                st.write(str(content))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def format_display_value(value: Any, value_type: str = "number", 
                        decimal_places: int = 2) -> str:
    """
    Format value for display based on type.
    
    Args:
        value: Value to format
        value_type: Type of formatting ("number", "percentage", "currency")
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if value_type == "percentage":
        return DataFormatter.format_percentage(value, decimal_places)
    elif value_type == "currency":
        return DataFormatter.format_currency(value, decimal_digits=decimal_places)
    else:
        return DataFormatter.format_number(value, decimal_places)

def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for division by zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default

def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract numeric value from text string.
    
    Args:
        text: Text containing numeric value
        
    Returns:
        Extracted numeric value or None
    """
    if not text:
        return None
    
    # Find numeric pattern in text
    numeric_pattern = r'[-+]?[\d,]*\.?\d+'
    match = re.search(numeric_pattern, str(text))
    
    if match:
        try:
            return float(match.group().replace(',', ''))
        except ValueError:
            return None
    
    return None

def create_hierarchical_index(items: List[str], separator: str = '.') -> Dict[str, List[str]]:
    """
    Create hierarchical index from flat list of items.
    
    Args:
        items: List of hierarchical items (e.g., ["A1.1", "A1.2", "B1.1"])
        separator: Separator used in hierarchy
        
    Returns:
        Dictionary mapping parent items to children
    """
    hierarchy = {}
    
    for item in items:
        parts = item.split(separator)
        
        for i in range(len(parts)):
            parent = separator.join(parts[:i+1])
            if parent not in hierarchy:
                hierarchy[parent] = []
            
            if i < len(parts) - 1:
                child = separator.join(parts[:i+2])
                if child not in hierarchy[parent]:
                    hierarchy[parent].append(child)
    
    return hierarchy

def get_timestamp_string(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: Format string for timestamp
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)