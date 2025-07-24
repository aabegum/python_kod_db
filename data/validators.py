# data/validators.py
"""
Data validation functions for the Electric Distribution Dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from utils.constants import (
    DATA_QUALITY_CHECKS, VALIDATION_RULES, ERROR_MESSAGES,
    SUPPORTED_FILE_TYPES, COMPANIES_RANGE
)


class DataValidator:
    """Handles data validation operations"""
    
    def __init__(self):
        self.quality_checks = DATA_QUALITY_CHECKS
        self.validation_rules = VALIDATION_RULES
    
    def validate_dataframe(self, df: pd.DataFrame, context: str = "standard") -> Tuple[bool, List[str]]:
        """
        Comprehensive dataframe validation
        
        Args:
            df: DataFrame to validate
            context: Context of validation ('standard', 'cumulative', 'layout')
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Basic structure checks
        if df is None:
            issues.append("DataFrame is None")
            return False, issues
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Context-specific validation
        if context == "standard":
            issues.extend(self._validate_standard_data(df))
        elif context == "cumulative":
            issues.extend(self._validate_cumulative_data(df))
        elif context == "layout":
            issues.extend(self._validate_layout_data(df))
        
        # Common validations
        issues.extend(self._validate_data_quality(df))
        issues.extend(self._validate_data_types(df))
        
        return len(issues) == 0, issues
    
    def _validate_standard_data(self, df: pd.DataFrame) -> List[str]:
        """Validate standard performance data"""
        issues = []
        
        # Check required columns
        required_cols = ['APG No', 'APG İsmi', 'Birim']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check company columns (numeric indices)
        company_cols = [col for col in df.columns if str(col).isdigit()]
        if len(company_cols) == 0:
            issues.append("No company columns found")
        
        # Validate APG No format
        if 'APG No' in df.columns:
            invalid_apgs = df[df['APG No'].isna() | (df['APG No'] == '')]['APG No'].count()
            if invalid_apgs > 0:
                issues.append(f"Found {invalid_apgs} invalid APG numbers")
        
        # Check for duplicate APG numbers
        if 'APG No' in df.columns:
            duplicates = df.duplicated(subset=['APG No']).sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate APG numbers")
        
        return issues
    
    def _validate_cumulative_data(self, df: pd.DataFrame) -> List[str]:
        """Validate cumulative performance data"""
        issues = []
        
        # Check minimum required columns
        if len(df.columns) < 50:  # Rough check for cumulative data structure
            issues.append("Insufficient columns for cumulative data")
        
        # Check for APG identifiers
        if df.columns[0] not in ['APG No', 'APG_No'] and df.columns[1] not in ['APG Adı', 'APG Name']:
            issues.append("First two columns should be APG No and APG Name")
        
        return issues
    
    def _validate_layout_data(self, df: pd.DataFrame) -> List[str]:
        """Validate PowerPoint layout data"""
        issues = []
        
        # Check required columns for layout
        layout_required = ['APG Kodu', 'Grafik_tipi', 'Sayfa']
        missing_layout = [col for col in layout_required if col not in df.columns]
        if missing_layout:
            issues.append(f"Missing layout columns: {missing_layout}")
        
        # Validate graph types
        if 'Grafik_tipi' in df.columns:
            valid_types = ['standard', 'stacked', 'overlayed']
            invalid_types = df[~df['Grafik_tipi'].isin(valid_types)]['Grafik_tipi'].unique()
            if len(invalid_types) > 0:
                issues.append(f"Invalid graph types: {invalid_types}")
        
        return issues
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate data quality metrics"""
        issues = []
        
        # Check null percentage
        null_percentages = df.isnull().sum() / len(df)
        high_null_cols = null_percentages[null_percentages > self.quality_checks['MAX_NULL_PERCENTAGE']]
        
        if len(high_null_cols) > 0:
            issues.append(f"High null percentage in columns: {high_null_cols.index.tolist()}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()]
        if len(empty_cols) > 0:
            issues.append(f"Completely empty columns: {empty_cols.tolist()}")
        
        return issues
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate data types"""
        issues = []
        
        # Check numeric columns
        numeric_cols = [col for col in df.columns if str(col).isdigit()]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to convert
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except:
                    issues.append(f"Column {col} cannot be converted to numeric")
        
        # Check percentage columns
        if 'Birim' in df.columns:
            non_percent_rows = df[~df['Birim'].isin(['%', 'Percent', 'percent'])]
            if len(non_percent_rows) > 0 and len(non_percent_rows) < len(df):
                issues.append("Mixed units found in Birim column")
        
        return issues
    
    def validate_apg_selection(self, selected_apgs: List[str], available_apgs: List[str]) -> Tuple[bool, List[str]]:
        """Validate APG selection"""
        issues = []
        
        if not selected_apgs:
            issues.append("No APGs selected")
            return False, issues
        
        if len(selected_apgs) < self.validation_rules['MIN_APGS']:
            issues.append(f"Minimum {self.validation_rules['MIN_APGS']} APGs required")
        
        if len(selected_apgs) > self.validation_rules['MAX_APGS']:
            issues.append(f"Maximum {self.validation_rules['MAX_APGS']} APGs allowed")
        
        # Check if selected APGs exist in available APGs
        invalid_apgs = [apg for apg in selected_apgs if apg not in available_apgs]
        if invalid_apgs:
            issues.append(f"Invalid APG selections: {invalid_apgs}")
        
        return len(issues) == 0, issues
    
    def validate_outlier_parameters(self, method: str, threshold: float) -> Tuple[bool, List[str]]:
        """Validate outlier detection parameters"""
        issues = []
        
        if threshold < self.validation_rules['MIN_THRESHOLD']:
            issues.append(f"Threshold too low. Minimum: {self.validation_rules['MIN_THRESHOLD']}")
        
        if threshold > self.validation_rules['MAX_THRESHOLD']:
            issues.append(f"Threshold too high. Maximum: {self.validation_rules['MAX_THRESHOLD']}")
        
        # Method-specific validation
        if method == 'std' and threshold < 0.5:
            issues.append("STD threshold should be at least 0.5")
        elif method == 'iqr' and threshold < 0.5:
            issues.append("IQR factor should be at least 0.5")
        elif method == 'mad' and threshold < 2.0:
            issues.append("MAD threshold should be at least 2.0")
        
        return len(issues) == 0, issues
    
    def validate_year_selection(self, year: int) -> Tuple[bool, List[str]]:
        """Validate year selection"""
        issues = []
        
        if year < self.validation_rules['MIN_YEAR']:
            issues.append(f"Year too early. Minimum: {self.validation_rules['MIN_YEAR']}")
        
        if year > self.validation_rules['MAX_YEAR']:
            issues.append(f"Year too late. Maximum: {self.validation_rules['MAX_YEAR']}")
        
        return len(issues) == 0, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for data validation"""
        if df is None or df.empty:
            return {'error': 'No data available'}
        
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df)).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Add specific summaries for key columns
        if 'APG No' in df.columns:
            summary['unique_apgs'] = df['APG No'].nunique()
            summary['duplicate_apgs'] = df.duplicated(subset=['APG No']).sum()
        
        if 'Birim' in df.columns:
            summary['units'] = df['Birim'].value_counts().to_dict()
        
        # Numeric data statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = {
                'mean': df[numeric_cols].mean().to_dict(),
                'std': df[numeric_cols].std().to_dict(),
                'min': df[numeric_cols].min().to_dict(),
                'max': df[numeric_cols].max().to_dict(),
                'zero_counts': (df[numeric_cols] == 0).sum().to_dict()
            }
        
        return summary


# Utility functions
def validate_file_type(filename: str) -> bool:
    """Validate file type"""
    if not filename:
        return False
    
    file_extension = filename.split('.')[-1].lower()
    return file_extension in SUPPORTED_FILE_TYPES


def validate_dataframe(df: pd.DataFrame, context: str = "standard") -> Tuple[bool, List[str]]:
    """Convenience function for dataframe validation"""
    validator = DataValidator()
    return validator.validate_dataframe(df, context)


def validate_apg_selection(selected_apgs: List[str], available_apgs: List[str]) -> Tuple[bool, List[str]]:
    """Convenience function for APG selection validation"""
    validator = DataValidator()
    return validator.validate_apg_selection(selected_apgs, available_apgs)


def validate_outlier_parameters(method: str, threshold: float) -> Tuple[bool, List[str]]:
    """Convenience function for outlier parameter validation"""
    validator = DataValidator()
    return validator.validate_outlier_parameters(method, threshold)


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for data summary"""
    validator = DataValidator()
    return validator.get_data_summary(df)


def create_data_validator() -> DataValidator:
    """Create and return a DataValidator instance"""
    return DataValidator()