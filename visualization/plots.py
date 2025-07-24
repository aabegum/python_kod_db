"""
Visualization and plotting utilities for the Electric Distribution Dashboard.

This module contains all plotting functions for standard, stacked, overlayed,
and cumulative charts with outlier detection integration.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional, Union, Tuple

from utils.constants import (
    COMPANIES_RANGE,
    NUM_OF_COMPANIES,
    DEFAULT_CHART_HEIGHT,
    COMPANY_COLORS,
    OUTLIER_MARKER_COLOR,
    OUTLIER_BORDER_COLOR,
    OUTLIER_BORDER_WIDTH,
    COMPANY_COLORSCALE,
    BENCHMARK_COLORS,
    COMPARISON_COLORS,
    PLOT_TEMPLATE,
    LEGEND_CONFIG,
    CHART_MARGINS,
    DEFAULT_DECIMAL_DIGITS,
    DEFAULT_ANNOTATION_FONT_SIZE,
    DEFAULT_WORD_WRAP_WIDTH
)
from analysis.outliers import OutlierStatistics
from data.loader import DataLoader


class ChartBuilder:
    """
    Chart building utilities for the dashboard.
    
    Handles creation of various chart types with outlier detection,
    styling, and interactive features.
    """
    
    def __init__(self, settings=None):
        """
        Initialize ChartBuilder.
        
        Args:
            settings: DashboardSettings instance
        """
        self.settings = settings
        self.loader = DataLoader(settings) if settings else None
    
    def format_percentage(self, value: float, decimal_digits: int = None) -> str:
        """
        Format value as percentage.
        
        Args:
            value: Numeric value to format
            decimal_digits: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if decimal_digits is None:
            decimal_digits = self.settings.decimal_digits if self.settings else DEFAULT_DECIMAL_DIGITS
        
        return f"%{value * 100:.{decimal_digits}f}".replace('.', ',')
    
    def wrap_text(self, text: str, width: int = None) -> str:
        """
        Wrap text to specified width for chart titles.
        
        Args:
            text: Text to wrap
            width: Maximum width in characters
            
        Returns:
            HTML-formatted wrapped text
        """
        if width is None:
            width = self.settings.word_wrap_limit if self.settings else DEFAULT_WORD_WRAP_WIDTH
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= width:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        
        return "<br>".join(lines)
    
    def get_display_companies(self, company_list: List[str]) -> List[str]:
        """
        Get display company names for charts.
        
        Args:
            company_list: List of companies from config
            
        Returns:
            List of display names for charts
        """
        if not self.loader:
            return [str(i) for i in range(1, NUM_OF_COMPANIES + 1)]
        
        excel_company_names = self.loader.get_excel_company_names(company_list)
        display_companies = excel_company_names + [
            str(i) for i in range(len(excel_company_names) + 1, NUM_OF_COMPANIES + 1)
        ]
        return display_companies[:NUM_OF_COMPANIES]
    
    def plot_standard_enhanced_v2(self, row: pd.Series, selected_year: int, 
                                 company_list: List[str],
                                 show_outliers: bool = True, 
                                 outlier_method: str = 'std', 
                                 threshold: float = 2.0, 
                                 hide_outliers: bool = False, 
                                 compare_methods: bool = False) -> go.Figure:
        """
        Enhanced standard plot with support for all three outlier detection methods.
        
        Args:
            row: DataFrame row containing APG data
            selected_year: Year being analyzed
            company_list: List of companies for the selected group
            show_outliers: Whether to show outlier analysis
            outlier_method: Outlier detection method ('std', 'iqr', 'mad')
            threshold: Threshold value for outlier detection
            hide_outliers: Whether to hide outlier points from chart
            compare_methods: Whether to show comparison lines for all methods
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        display_companies = self.get_display_companies(company_list)
        y_values = row[COMPANIES_RANGE].values
        x_values = COMPANIES_RANGE
        
        # Get outlier information
        outlier_indices = []
        filtered_mean_col = f'filtered_mean_{outlier_method}'
        outlier_indices_col = f'outlier_indices_{outlier_method}'
        
        if show_outliers and outlier_indices_col in row:
            outlier_indices = OutlierStatistics.parse_outlier_indices(row[outlier_indices_col])
        
        # Prepare data for plotting
        if hide_outliers and outlier_indices:
            mask = [i not in outlier_indices for i in range(len(y_values))]
            plot_x = [x for i, x in enumerate(x_values) if mask[i]]
            plot_y = [y for i, y in enumerate(y_values) if mask[i]]
            plot_colors = [0 if i < len(company_list) else 1 for i in range(len(plot_x))]
            plot_companies = [display_companies[i] for i in range(len(x_values)) if mask[i]]
        else:
            plot_x = x_values
            plot_y = y_values
            plot_colors = [0] * len(company_list) + [1] * (NUM_OF_COMPANIES - len(company_list))
            plot_companies = display_companies
        
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=plot_x,
            y=plot_y,
            mode='markers+text',
            marker=dict(
                color=plot_colors,
                colorscale=COMPANY_COLORSCALE,
                size=10,
                line=dict(width=0)
            ),
            text=[self.format_percentage(val) if row["Birim"] == "%" else str(val) for val in plot_y],
            textposition="top center",
            textfont=dict(size=self.settings.annotation_font_size if self.settings else DEFAULT_ANNOTATION_FONT_SIZE),
            name="Company Performance",
            showlegend=False,
            hovertemplate='<b>Company: %{customdata}</b><br>Value: %{text}<extra></extra>',
            customdata=plot_companies
        ))
        
        # Add outlier markers
        if show_outliers and outlier_indices and not hide_outliers:
            outlier_x = [x_values[i] for i in outlier_indices if i < len(x_values)]
            outlier_y = [y_values[i] for i in outlier_indices if i < len(y_values)]
            
            if outlier_x and outlier_y:
                fig.add_trace(go.Scatter(
                    x=outlier_x,
                    y=outlier_y,
                    mode='markers',
                    marker=dict(
                        size=18,
                        color=OUTLIER_MARKER_COLOR,
                        line=dict(color=OUTLIER_BORDER_COLOR, width=OUTLIER_BORDER_WIDTH),
                        symbol='diamond'
                    ),
                    name=f"Outliers ({outlier_method.upper()})",
                    showlegend=True,
                    hovertemplate='<b>OUTLIER</b><br>Company: %{customdata}<br>Value: %{y}<extra></extra>',
                    customdata=[display_companies[i] for i in outlier_indices if i < len(display_companies)]
                ))
        
        # Add filtered mean line
        if filtered_mean_col in row and not pd.isna(row[filtered_mean_col]):
            mean_color = self.settings.horizontal_mean_color if self.settings else 'purple'
            fig.add_hline(
                y=row[filtered_mean_col], 
                line=dict(color=mean_color, dash='dash'),
                annotation_text=f"Filtered Mean ({outlier_method.upper()}): {row[filtered_mean_col]:.2f}",
                annotation_position="right"
            )
        
        # Add comparison lines if comparing methods
        if compare_methods:
            methods = ['std', 'iqr', 'mad']
            
            for i, method in enumerate(methods):
                if method != outlier_method:
                    mean_col = f'filtered_mean_{method}'
                    if mean_col in row and not pd.isna(row[mean_col]):
                        fig.add_hline(
                            y=row[mean_col],
                            line=dict(color=COMPARISON_COLORS[i], dash='dot', width=2),
                            annotation_text=f"{method.upper()}: {row[mean_col]:.2f}",
                            annotation_position="left" if i % 2 == 0 else "right"
                        )
        
        # Configure title with outlier information
        title_text = f"{row['APG Full Name']} ({selected_year})"
        if show_outliers and outlier_indices:
            outlier_count = len(outlier_indices)
            method_name = outlier_method.upper()
            
            if hide_outliers:
                title_text += f"<br><sub>{outlier_count} outliers hidden ({method_name})</sub>"
            else:
                title_text += f"<br><sub>{outlier_count} outliers detected ({method_name})</sub>"
        
        if compare_methods:
            title_text += "<br><sub>Method comparison enabled</sub>"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=self.wrap_text(title_text, 50),
                font=dict(size=12)
            ),
            xaxis=dict(
                title="Companies",
                tickvals=COMPANIES_RANGE,
                ticktext=display_companies,
                tickfont=dict(size=10),
                tickangle=45
            ),
            yaxis=dict(
                title=row["Birim"] if row["Birim"] != "%" else "Percentage",
                tickformat=".2%" if row["Birim"] == "%" else None,
                tickfont=dict(size=10)
            ),
            showlegend=(show_outliers and len(outlier_indices) > 0 and not hide_outliers) or compare_methods,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            height=DEFAULT_CHART_HEIGHT,
            template=PLOT_TEMPLATE,
            margin=dict(b=100)
        )
        
        return fig
    
    def plot_stacked_enhanced(self, row: pd.Series, transposed: pd.DataFrame, 
                             category_to_apg_dict: Dict, 
                             category_to_apg_full_name_dict: Dict, 
                             selected_year: int) -> go.Figure:
        """
        Enhanced stacked plot for category-based analysis.
        
        Args:
            row: DataFrame row containing APG data
            transposed: Transposed data for plotting
            category_to_apg_dict: Mapping of categories to APG numbers
            category_to_apg_full_name_dict: Mapping of categories to APG full names
            selected_year: Year being analyzed
            
        Returns:
            Plotly Figure object
        """
        stacked_apg_nos = category_to_apg_dict.get(row["Category No"], [])
        fig = go.Figure()
        
        for apg in stacked_apg_nos:
            if apg in transposed.columns:
                fig.add_trace(go.Bar(
                    x=COMPANIES_RANGE,
                    y=transposed[apg],
                    name=apg
                ))
        
        title_text = f"{row['APG Full Name']} ({selected_year})"
        
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=self.wrap_text(title_text, 50),
                font=dict(size=12)
            ),
            xaxis=dict(
                title="Companies", 
                tickvals=COMPANIES_RANGE, 
                ticktext=COMPANIES_RANGE
            ),
            yaxis=dict(
                title="Percentage", 
                tickformat=".2%"
            ),
            legend=dict(x=1, y=0.5),
            height=DEFAULT_CHART_HEIGHT,
            template=PLOT_TEMPLATE
        )
        
        return fig
    
    def plot_overlayed_enhanced(self, row: pd.Series, transposed: pd.DataFrame, 
                               shuffled_df: pd.DataFrame, selected_year: int,
                               company_list: List[str],
                               show_outliers: bool = False, 
                               outlier_method: str = 'std', 
                               threshold: float = 2.0, 
                               hide_outliers: bool = False) -> go.Figure:
        """
        Enhanced overlayed plot combining bar and scatter plots.
        
        Args:
            row: DataFrame row containing APG data
            transposed: Transposed data for plotting
            shuffled_df: Shuffled DataFrame
            selected_year: Year being analyzed
            company_list: List of companies for the selected group
            show_outliers: Whether to show outlier analysis
            outlier_method: Outlier detection method
            threshold: Threshold value for outlier detection
            hide_outliers: Whether to hide outlier points
            
        Returns:
            Plotly Figure object
        """
        apg = row["APG Group"]
        alt_bilgi = f"{apg} EK"
        ara_df = shuffled_df[shuffled_df["APG Group"] == apg]
        main_row = ara_df[ara_df["APG No"] == apg].iloc[0] if len(ara_df[ara_df["APG No"] == apg]) > 0 else row
        ek_rows = ara_df[ara_df["APG No"] == alt_bilgi]
        
        fig = go.Figure()
        display_companies = self.get_display_companies(company_list)
        
        # Add bar chart if EK data exists
        if len(ek_rows) > 0 and alt_bilgi in transposed.columns:
            ek_row = ek_rows.iloc[0]
            bar_color = self.settings.overlay_graph_bar_color if self.settings else 'blue'
            fig.add_trace(go.Bar(
                x=COMPANIES_RANGE,
                y=transposed[alt_bilgi],
                name=ek_row["APG İsmi"],
                marker_color=bar_color,
                opacity=0.6
            ))
        
        # Prepare scatter data
        if main_row["APG No"] in transposed.columns:
            y_values = transposed[main_row["APG No"]].values
        else:
            y_values = [0] * len(COMPANIES_RANGE)
        
        x_values = COMPANIES_RANGE
        
        # Get outlier information
        outlier_indices = []
        filtered_mean_col = f'filtered_mean_{outlier_method}'
        outlier_indices_col = f'outlier_indices_{outlier_method}'
        
        if show_outliers and outlier_indices_col in main_row:
            outlier_indices = OutlierStatistics.parse_outlier_indices(main_row[outlier_indices_col])
        
        # Prepare plot data
        if hide_outliers and outlier_indices:
            mask = [i not in outlier_indices for i in range(len(y_values))]
            plot_x = [x for i, x in enumerate(x_values) if mask[i]]
            plot_y = [y for i, y in enumerate(y_values) if mask[i]]
            plot_colors = [0 if i < len(company_list) else 1 for i in range(len(plot_x))]
            plot_companies = [display_companies[i] for i in range(len(x_values)) if mask[i]]
        else:
            plot_x = x_values
            plot_y = y_values
            plot_colors = [0] * len(company_list) + [1] * (NUM_OF_COMPANIES - len(company_list))
            plot_companies = display_companies
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=plot_x,
            y=plot_y,
            mode='markers+text',
            name=main_row["APG İsmi"],
            marker=dict(
                color=plot_colors, 
                colorscale=COMPANY_COLORSCALE, 
                size=10
            ),
            text=[self.format_percentage(val) if "Birim" in main_row and main_row["Birim"] == "%" else str(val) for val in plot_y] if "Birim" in main_row else ["0"] * len(plot_y),
            textposition="top center",
            textfont=dict(size=self.settings.annotation_font_size if self.settings else DEFAULT_ANNOTATION_FONT_SIZE),
            yaxis='y2',
            hovertemplate='<b>Company: %{customdata}</b><br>Value: %{text}<extra></extra>',
            customdata=plot_companies
        ))
        
        # Add outlier markers
        if show_outliers and outlier_indices and not hide_outliers:
            outlier_x = [x_values[i] for i in outlier_indices if i < len(x_values)]
            outlier_y = [y_values[i] for i in outlier_indices if i < len(y_values)]
            
            if outlier_x and outlier_y:
                fig.add_trace(go.Scatter(
                    x=outlier_x,
                    y=outlier_y,
                    mode='markers',
                    marker=dict(
                        size=18,
                        color=OUTLIER_MARKER_COLOR,
                        line=dict(color=OUTLIER_BORDER_COLOR, width=OUTLIER_BORDER_WIDTH),
                        symbol='diamond'
                    ),
                    name="Outliers",
                    showlegend=True,
                    hovertemplate='<b>OUTLIER</b><br>Company: %{customdata}<br>Value: %{y}<extra></extra>',
                    customdata=[display_companies[i] for i in outlier_indices if i < len(display_companies)],
                    yaxis='y2'
                ))
        
        # Add filtered mean line
        if filtered_mean_col in main_row and not pd.isna(main_row[filtered_mean_col]):
            mean_color = self.settings.horizontal_mean_color if self.settings else 'purple'
            fig.add_hline(
                y=main_row[filtered_mean_col], 
                line=dict(color=mean_color, dash='dash'),
                yref='y2'
            )
        
        # Configure title
        title_text = f"{row['APG Full Name']} ({selected_year})"
        if show_outliers and outlier_indices:
            if hide_outliers:
                title_text += f"<br><sub>{len(outlier_indices)} outliers hidden</sub>"
            else:
                title_text += f"<br><sub>{len(outlier_indices)} outliers detected</sub>"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=self.wrap_text(title_text, 50),
                font=dict(size=12)
            ),
            xaxis=dict(
                title="Companies",
                tickvals=COMPANIES_RANGE,
                ticktext=display_companies,
                tickfont=dict(size=10),
                tickangle=45
            ),
            yaxis=dict(
                title=ek_rows.iloc[0]["APG İsmi"] + " (Bar)" if len(ek_rows) > 0 else "Performance"
            ),
            yaxis2=dict(
                title=main_row["APG İsmi"] + " (Scatter)", 
                overlaying='y', 
                side='right'
            ),
            showlegend=True,
            height=DEFAULT_CHART_HEIGHT,
            template=PLOT_TEMPLATE
        )
        
        return fig
    
    def plot_cumulative_chart_for_company_group(self, apg_data: pd.DataFrame, 
                                               benchmark_data: Dict, 
                                               selected_company_group: str, 
                                               company_list: List[str],
                                               show_outliers: bool = True, 
                                               outlier_method: str = 'iqr', 
                                               threshold: float = 1.5) -> go.Figure:
        """
        Create cumulative chart for company group performance vs market benchmarks.
        
        Args:
            apg_data: DataFrame containing APG performance data
            benchmark_data: Dictionary containing market benchmark data
            selected_company_group: Name of the selected company group
            company_list: List of companies in the group
            show_outliers: Whether to show outlier analysis
            outlier_method: Outlier detection method
            threshold: Threshold value for outlier detection
            
        Returns:
            Plotly Figure object
        """
        from analysis.outliers import detect_outliers_iqr, detect_outliers_mad, detect_outliers_std
        
        fig = go.Figure()
        
        years = self.settings.cumulative_years if self.settings else ['2021', '2022', '2023', '2024']
        excel_company_names = self.loader.get_excel_company_names(company_list) if self.loader else company_list
        
        all_values = []
        company_data_map = {}
        
        # Collect company data
        for company_name in excel_company_names:
            company_row = apg_data[apg_data['Company'] == company_name]
            if len(company_row) > 0:
                company_row = company_row.iloc[0]
                company_values = []
                for year in years:
                    try:
                        val = float(company_row[year])
                        company_values.append(val)
                    except (ValueError, TypeError):
                        company_values.append(0)
                company_data_map[company_name] = company_values
                all_values.extend([v for v in company_values if v != 0])
        
        # Detect outliers
        outliers = []
        outlier_info = {}
        
        if show_outliers and len(all_values) > 0:
            cleaned_values = np.array([float(v) for v in all_values if not pd.isna(v)], dtype=float)
            valid_values = cleaned_values[~np.isnan(cleaned_values)]
            
            if len(valid_values) > 0:
                if outlier_method == 'iqr':
                    outliers, non_outliers = detect_outliers_iqr(valid_values, k=threshold)
                elif outlier_method == 'mad':
                    outliers, non_outliers = detect_outliers_mad(valid_values, threshold=threshold)
                else:  # std
                    outliers, non_outliers = detect_outliers_std(valid_values, sigma=threshold)
                
                # Map outliers to companies and years
                for company_name, values in company_data_map.items():
                    for i, (year, val) in enumerate(zip(years, values)):
                        try:
                            val = float(val)
                            if val in outliers:
                                key = f"{company_name}_{year}"
                                outlier_info[key] = {
                                    'company': company_name,
                                    'year': year,
                                    'value': val,
                                    'index': i
                                }
                        except (ValueError, TypeError):
                            continue
        
        # Add company traces
        companies_found = 0
        for i, company_name in enumerate(excel_company_names):
            if company_name in company_data_map:
                company_values = company_data_map[company_name]
                
                # Prepare marker colors for outliers
                marker_colors = []
                for val in company_values:
                    if val in outliers:
                        base_color = COMPANY_COLORS[i % len(COMPANY_COLORS)]
                        marker_colors.append(base_color)
                    else:
                        marker_colors.append(COMPANY_COLORS[i % len(COMPANY_COLORS)])
                
                fig.add_trace(go.Bar(
                    x=years,
                    y=company_values,
                    name=f'{company_name}',
                    marker=dict(
                        color=marker_colors,
                        line=dict(
                            color=['red' if val in outliers else 'rgba(0,0,0,0)' for val in company_values],
                            width=3
                        )
                    ),
                    opacity=0.8,
                    text=[f'{val:.1f}%{" ⚠️" if val in outliers else ""}' if val != 0 else '' for val in company_values],
                    textposition='outside',
                    textfont=dict(size=9),
                    hovertemplate=f'<b>{company_name}</b><br>%{{x}}: %{{y:.1f}}%' + 
                                 '<br><b>OUTLIER</b>' if any(val in outliers for val in company_values) else '' +
                                 '<extra></extra>',
                    offsetgroup=i
                ))
                companies_found += 1
        
        if companies_found == 0:
            st.error("No data found for any companies in the selected group")
            return go.Figure()
        
        # Add benchmark lines
        if benchmark_data:
            min_values = []
            max_values = []
            avg_values = []
            
            for year in years:
                if year in benchmark_data:
                    min_values.append(benchmark_data[year]['min'])
                    max_values.append(benchmark_data[year]['max'])
                    avg_values.append(benchmark_data[year]['avg'])
                else:
                    min_values.append(0)
                    max_values.append(0)
                    avg_values.append(0)
            
            # Market Max line
            fig.add_trace(go.Scatter(
                x=years,
                y=max_values,
                mode='lines+markers',
                name='Market Max',
                line=dict(color=BENCHMARK_COLORS['MAX'], width=3),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='<b>Market Maximum</b><br>%{x}: %{y:.1f}%<extra></extra>'
            ))
            
            # Market Min line
            fig.add_trace(go.Scatter(
                x=years,
                y=min_values,
                mode='lines+markers',
                name='Market Min',
                line=dict(color=BENCHMARK_COLORS['MIN'], width=3),
                marker=dict(size=7, symbol='square'),
                hovertemplate='<b>Market Minimum</b><br>%{x}: %{y:.1f}%<extra></extra>'
            ))
            
            # Market Average line
            fig.add_trace(go.Scatter(
                x=years,
                y=avg_values,
                mode='lines+markers',
                name='Market Avg',
                line=dict(color=BENCHMARK_COLORS['AVG'], width=3, dash='dash'),
                marker=dict(size=7, symbol='triangle-up'),
                hovertemplate='<b>Market Average</b><br>%{x}: %{y:.1f}%<extra></extra>'
            ))
        
        # Add outlier annotation
        if outlier_info and show_outliers:
            outlier_text = f"Outliers detected: {len(outliers)} values"
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=outlier_text,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        # Configure title and layout
        apg_info = apg_data.iloc[0] if len(apg_data) > 0 else {'APG No': 'Unknown', 'APG Name': 'Unknown'}
        
        title_text = f"{apg_info['APG No']} - {apg_info['APG Name']}<br>{selected_company_group} Group Performance vs Market Benchmarks"
        if show_outliers:
            title_text += f"<br><sub>Outlier Detection: {outlier_method.upper()}" + (f" (k={threshold})" if outlier_method == 'iqr' else f" (threshold={threshold})") + "</sub>"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=13),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(text="Years", font=dict(size=11)),
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=dict(text="Performance (%)", font=dict(size=11)),
                tickfont=dict(size=10),
                tickformat=".1f"
            ),
            showlegend=True,
            legend=LEGEND_CONFIG,
            height=DEFAULT_CHART_HEIGHT,
            template=PLOT_TEMPLATE,
            hovermode='x unified',
            barmode='group',
            margin=CHART_MARGINS
        )
        
        return fig


# ============================================================================
# UTILITY FUNCTIONS FOR CHART CREATION
# ============================================================================

def create_standard_chart(settings, row: pd.Series, selected_year: int, 
                         company_list: List[str], **kwargs) -> go.Figure:
    """
    Create a standard chart with settings.
    
    Args:
        settings: DashboardSettings instance
        row: DataFrame row containing APG data
        selected_year: Year being analyzed
        company_list: List of companies for the selected group
        **kwargs: Additional arguments for chart customization
        
    Returns:
        Plotly Figure object
    """
    builder = ChartBuilder(settings)
    return builder.plot_standard_enhanced_v2(row, selected_year, company_list, **kwargs)

def create_stacked_chart(settings, row: pd.Series, transposed: pd.DataFrame,
                        category_to_apg_dict: Dict, category_to_apg_full_name_dict: Dict,
                        selected_year: int) -> go.Figure:
    """
    Create a stacked chart with settings.
    
    Args:
        settings: DashboardSettings instance
        row: DataFrame row containing APG data
        transposed: Transposed data for plotting
        category_to_apg_dict: Mapping of categories to APG numbers
        category_to_apg_full_name_dict: Mapping of categories to APG full names
        selected_year: Year being analyzed
        
    Returns:
        Plotly Figure object
    """
    builder = ChartBuilder(settings)
    return builder.plot_stacked_enhanced(row, transposed, category_to_apg_dict, 
                                        category_to_apg_full_name_dict, selected_year)

def create_overlayed_chart(settings, row: pd.Series, transposed: pd.DataFrame,
                          shuffled_df: pd.DataFrame, selected_year: int,
                          company_list: List[str], **kwargs) -> go.Figure:
    """
    Create an overlayed chart with settings.
    
    Args:
        settings: DashboardSettings instance
        row: DataFrame row containing APG data
        transposed: Transposed data for plotting
        shuffled_df: Shuffled DataFrame
        selected_year: Year being analyzed
        company_list: List of companies for the selected group
        **kwargs: Additional arguments for chart customization
        
    Returns:
        Plotly Figure object
    """
    builder = ChartBuilder(settings)
    return builder.plot_overlayed_enhanced(row, transposed, shuffled_df, selected_year,
                                          company_list, **kwargs)

def create_cumulative_chart(settings, apg_data: pd.DataFrame, benchmark_data: Dict,
                           selected_company_group: str, company_list: List[str],
                           **kwargs) -> go.Figure:
    """
    Create a cumulative chart with settings.
    
    Args:
        settings: DashboardSettings instance
        apg_data: DataFrame containing APG performance data
        benchmark_data: Dictionary containing market benchmark data
        selected_company_group: Name of the selected company group
        company_list: List of companies in the group
        **kwargs: Additional arguments for chart customization
        
    Returns:
        Plotly Figure object
    """
    builder = ChartBuilder(settings)
    return builder.plot_cumulative_chart_for_company_group(apg_data, benchmark_data,
                                                          selected_company_group, 
                                                          company_list, **kwargs)

# ============================================================================
# CHART TYPE DETECTION AND ROUTING
# ============================================================================

def create_chart_by_type(settings, row: pd.Series, chart_type: str, 
                        selected_year: int, company_list: List[str],
                        transposed: Optional[pd.DataFrame] = None,
                        shuffled_df: Optional[pd.DataFrame] = None,
                        category_mappings: Optional[Tuple[Dict, Dict]] = None,
                        **kwargs) -> go.Figure:
    """
    Create a chart based on the specified type.
    
    Args:
        settings: DashboardSettings instance
        row: DataFrame row containing APG data
        chart_type: Type of chart ('standard', 'stacked', 'overlayed')
        selected_year: Year being analyzed
        company_list: List of companies for the selected group
        transposed: Transposed data for plotting (required for stacked/overlayed)
        shuffled_df: Shuffled DataFrame (required for overlayed)
        category_mappings: Tuple of category mappings (required for stacked)
        **kwargs: Additional arguments for chart customization
        
    Returns:
        Plotly Figure object
    """
    builder = ChartBuilder(settings)
    
    if chart_type == "standard":
        return builder.plot_standard_enhanced_v2(row, selected_year, company_list, **kwargs)
    
    elif chart_type == "stacked":
        if transposed is None or category_mappings is None:
            raise ValueError("Transposed data and category mappings required for stacked charts")
        
        category_to_apg_dict, category_to_apg_full_name_dict = category_mappings
        return builder.plot_stacked_enhanced(row, transposed, category_to_apg_dict,
                                           category_to_apg_full_name_dict, selected_year)
    
    elif chart_type == "overlayed":
        if transposed is None or shuffled_df is None:
            raise ValueError("Transposed data and shuffled DataFrame required for overlayed charts")
        
        return builder.plot_overlayed_enhanced(row, transposed, shuffled_df, selected_year,
                                             company_list, **kwargs)
    
    else:
        # Default to standard chart
        return builder.plot_standard_enhanced_v2(row, selected_year, company_list, **kwargs)

# ============================================================================
# PERFORMANCE SUMMARY FUNCTIONS
# ============================================================================

def create_performance_summary_data(shuffled_df: pd.DataFrame, 
                                   company_list: List[str],
                                   selected_apgs: List[str],
                                   available_apgs: List[str]) -> Dict[str, Any]:
    """
    Create performance summary statistics for display.
    
    Args:
        shuffled_df: Shuffled DataFrame with performance data
        company_list: List of companies in the selected group
        selected_apgs: List of selected APG numbers
        available_apgs: List of all available APG numbers
        
    Returns:
        Dictionary containing summary statistics
    """
    summary_data = {}
    
    if len(company_list) > 0:
        company_columns = [col for col in shuffled_df.columns if col in COMPANIES_RANGE]
        if company_columns:
            company_data = shuffled_df[company_columns[:len(company_list)]].values.flatten()
            company_data = company_data[~pd.isna(company_data)]
            
            if len(company_data) > 0:
                summary_data.update({
                    'average_performance': company_data.mean(),
                    'best_performance': company_data.max(),
                    'worst_performance': company_data.min(),
                    'performance_range': company_data.max() - company_data.min(),
                    'above_average_count': len(company_data[company_data > company_data.mean()]),
                    'total_data_points': len(company_data),
                    'success_rate': (len(company_data[company_data > company_data.mean()]) / len(company_data)) * 100
                })
    
    summary_data.update({
        'total_apgs_analyzed': len(selected_apgs),
        'available_apgs': len(available_apgs),
        'apg_coverage': (len(selected_apgs) / len(available_apgs)) * 100 if available_apgs else 0
    })
    
    return summary_data

def create_cumulative_summary_data(final_filtered_df: pd.DataFrame,
                                  excel_company_names: List[str],
                                  selected_apgs: List[str],
                                  available_apgs: List[str],
                                  cumulative_years: List[str]) -> Dict[str, Any]:
    """
    Create cumulative performance summary statistics.
    
    Args:
        final_filtered_df: Filtered DataFrame with cumulative data
        excel_company_names: List of Excel company names
        selected_apgs: List of selected APG numbers
        available_apgs: List of all available APG numbers
        cumulative_years: List of years for cumulative analysis
        
    Returns:
        Dictionary containing cumulative summary statistics
    """
    summary_data = final_filtered_df[cumulative_years].values.flatten()
    summary_data = summary_data[~pd.isna(summary_data)]
    
    cumulative_summary = {}
    
    if len(summary_data) > 0:
        positive_count = len(summary_data[summary_data > 0])
        cumulative_summary.update({
            'average_performance': summary_data.mean(),
            'best_performance': summary_data.max(),
            'worst_performance': summary_data.min(),
            'performance_range': summary_data.max() - summary_data.min(),
            'positive_performances': positive_count,
            'total_data_points': len(summary_data),
            'success_rate': (positive_count / len(summary_data)) * 100
        })
    
    cumulative_summary.update({
        'total_apgs_analyzed': len(selected_apgs),
        'available_apgs': len(available_apgs),
        'apg_coverage': (len(selected_apgs) / len(available_apgs)) * 100 if available_apgs else 0,
        'companies_analyzed': len(excel_company_names)
    })
    
    return cumulative_summary

# ============================================================================
# CHART STYLING AND CONFIGURATION
# ============================================================================

def apply_custom_styling(fig: go.Figure, custom_config: Dict[str, Any]) -> go.Figure:
    """
    Apply custom styling configuration to a chart.
    
    Args:
        fig: Plotly Figure object
        custom_config: Dictionary containing styling configuration
        
    Returns:
        Styled Plotly Figure object
    """
    # Apply custom colors if provided
    if 'colors' in custom_config:
        # Update trace colors
        for i, trace in enumerate(fig.data):
            if i < len(custom_config['colors']):
                if hasattr(trace, 'marker') and trace.marker:
                    trace.marker.color = custom_config['colors'][i]
                elif hasattr(trace, 'line') and trace.line:
                    trace.line.color = custom_config['colors'][i]
    
    # Apply custom layout settings
    if 'layout' in custom_config:
        fig.update_layout(**custom_config['layout'])
    
    # Apply custom axis settings
    if 'xaxis' in custom_config:
        fig.update_xaxes(**custom_config['xaxis'])
    
    if 'yaxis' in custom_config:
        fig.update_yaxes(**custom_config['yaxis'])
    
    return fig

def get_chart_export_config() -> Dict[str, Any]:
    """
    Get configuration for chart export.
    
    Returns:
        Dictionary containing export configuration
    """
    return {
        'config': {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'dashboard_chart',
                'height': DEFAULT_CHART_HEIGHT,
                'width': 1200,
                'scale': 2
            }
        }
    }

# ============================================================================
# DEBUGGING AND VALIDATION FUNCTIONS
# ============================================================================

def validate_chart_data(row: pd.Series, required_columns: List[str]) -> bool:
    """
    Validate that required data is present for chart creation.
    
    Args:
        row: DataFrame row to validate
        required_columns: List of required column names
        
    Returns:
        True if all required data is present, False otherwise
    """
    for col in required_columns:
        if col not in row or pd.isna(row[col]):
            return False
    return True

def log_chart_creation_info(chart_type: str, apg_no: str, 
                           outlier_count: int = 0, **kwargs) -> None:
    """
    Log information about chart creation for debugging.
    
    Args:
        chart_type: Type of chart being created
        apg_no: APG number being plotted
        outlier_count: Number of outliers detected
        **kwargs: Additional information to log
    """
    info_msg = f"Creating {chart_type} chart for APG {apg_no}"
    
    if outlier_count > 0:
        info_msg += f" (with {outlier_count} outliers)"
    
    if kwargs:
        additional_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        info_msg += f" - {additional_info}"
    
    # Use Streamlit's debug logging if available
    if hasattr(st, 'write'):
        # Only log in debug mode to avoid cluttering the interface
        pass
    else:
        print(info_msg)