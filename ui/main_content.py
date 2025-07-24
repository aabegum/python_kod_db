# ui/main_content.py
"""
Main content area components for the Electric Distribution Dashboard
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from utils.helpers import calculate_data_quality_score, quick_data_summary


class MainContentComponents:
    """Handles main content area UI components"""
    
    def __init__(self):
        pass
    
    def render_header(self):
        """Render application header"""
        st.markdown('''
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h1 style="color: white; margin: 0;">⚡ Electric Distribution Dashboard</h1>
            <p style="color: #e8f4f8; margin: 0; font-size: 1.1em;">
                Advanced Performance Analysis with Outlier Detection
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    def render_data_info(self, df: pd.DataFrame, year: int, company_group: str):
        """Render data information section"""
        st.info(f"📅 Analyzing performance data for year **{year}** - **{company_group}** group")
        st.markdown(f"**Data Summary:** {quick_data_summary(df)}")
    
    def render_performance_summary(self, filtered_data: Dict[str, Any], 
                                 selected_year: int, company_group: str):
        """Render performance summary section"""
        with st.expander(f"📊 {selected_year} Performance Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total APGs", len(filtered_data.get('selected_apgs', [])))
                st.metric("Company Group", company_group)
            
            with col2:
                st.metric("Analysis Year", selected_year)
                if 'filtered_df' in filtered_data:
                    st.metric("Data Points", len(filtered_data['filtered_df']))
            
            with col3:
                if 'filtered_df' in filtered_data:
                    quality_score = calculate_data_quality_score(filtered_data['filtered_df'])
                    st.metric("Data Quality", f"{quality_score['score']:.1f}%")
                    st.metric("Quality Grade", quality_score['grade'])
    
    def render_outlier_summary(self, outlier_results: Dict[str, Any], 
                             options: Dict[str, Any]):
        """Render outlier analysis summary"""
        method_key = outlier_results.get('method_key', 'std')
        results = outlier_results.get('outlier_results', {}).get(method_key, {})
        
        if not results:
            return
        
        with st.expander(f"🔍 Outlier Analysis ({options.get('outlier_method', 'STD')})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Outliers", results.get('outlier_summary', {}).get('total', 0))
            
            with col2:
                st.metric("APGs with Outliers", len(results.get('apgs_with_outliers', [])))
            
            with col3:
                st.metric("Detection Method", options.get('outlier_method', 'Unknown'))
                st.metric("Threshold", f"{options.get('threshold', 'N/A')}")
            
            # Show outlier details if available
            outlier_details = results.get('outlier_details', [])
            if outlier_details:
                st.markdown("**📋 Outlier Details:**")
                outlier_df = pd.DataFrame(outlier_details)
                st.dataframe(outlier_df, use_container_width=True)
    
    def render_apg_selection_info(self, selected_apgs: List[str], 
                                available_apgs: List[str]):
        """Render APG selection information"""
        if not selected_apgs:
            st.warning("⚠️ No APGs selected for analysis")
            return
        
        if len(selected_apgs) != len(available_apgs):
            st.info(f"🎯 Showing {len(selected_apgs)} of {len(available_apgs)} available APGs")
        
        if len(selected_apgs) > 50:
            st.warning(f"⚠️ {len(selected_apgs)} APGs selected. This may affect performance.")
    
    def render_charts_section(self, title: str):
        """Render charts section header"""
        st.subheader(f"📈 {title}")
        st.markdown("---")
    
    def render_data_export_section(self, df: pd.DataFrame, filename: str, 
                                 title: str = "Data Export"):
        """Render data export section"""
        with st.expander(f"📥 {title}", expanded=False):
            # Show data preview
            st.markdown("**📋 Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Export button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Download the complete analysis data"
            )
    
    def render_error_message(self, message: str):
        """Render error message"""
        st.error(f"❌ {message}")
    
    def render_success_message(self, message: str):
        """Render success message"""
        st.success(f"✅ {message}")
    
    def render_warning_message(self, message: str):
        """Render warning message"""
        st.warning(f"⚠️ {message}")
    
    def render_info_message(self, message: str):
        """Render info message"""
        st.info(f"ℹ️ {message}")
    
    def render_loading_message(self, message: str):
        """Render loading message with spinner"""
        return st.spinner(message)
    
    def render_footer(self, version: str):
        """Render application footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⚡ Electric Distribution Dashboard**")
            st.markdown(f"*Version {version}*")
        
        with col2:
            st.markdown("**🔧 Features:**")
            st.markdown("• Advanced Outlier Detection")
            st.markdown("• Multi-Method Comparison")
            st.markdown("• Interactive Visualizations")
        
        with col3:
            st.markdown("**📊 Powered by:**")
            st.markdown("• Streamlit")
            st.markdown("• Plotly")
            st.markdown("• Pandas")


def create_main_content() -> MainContentComponents:
    """Create and return a MainContentComponents instance"""
    return MainContentComponents()