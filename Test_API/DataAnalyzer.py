# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "requests",
#   "openpyxl",
#   "xlrd",
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.output_dir = "analysis_output"
        self.df = None
        self.api_token = "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"
        self.api_url = "https://integrate.api.nvidia.com/v1"
        
    def load_data(self):
        """Load CSV or Excel file with multiple encoding attempts"""
        print("Loading dataset...")
        file_extension = Path(self.file_path).suffix.lower()
        
        try:
            if file_extension in ['.csv']:
                # Try multiple encodings for CSV files
                encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(self.file_path, encoding=encoding)
                        print(f"CSV loaded successfully with {encoding} encoding!")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not read CSV with any supported encoding")
                    
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
                print("Excel file loaded successfully!")
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
                
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def analyze_data(self):
        """Perform comprehensive data analysis"""
        print("Analyzing the data...")
        
        # Basic info
        self.basic_info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict()
        }
        
        # Summary statistics for numerical columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.summary_stats = numeric_df.describe() if not numeric_df.empty else pd.DataFrame()
        
        # Missing values
        self.missing_values = self.df.isnull().sum()
        
        # Correlation matrix for numerical columns
        self.corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
        
        # Detect outliers using IQR method
        self.outliers = self.detect_outliers(numeric_df)
        
        # Data types summary
        self.dtype_summary = self.df.dtypes.value_counts()
        
        print("Data analysis complete.")
        return True
    
    def detect_outliers(self, numeric_df):
        """Detect outliers using IQR method"""
        if numeric_df.empty:
            return pd.Series()
            
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
        return outliers
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use('default')
        visualization_files = []
        
        # 1. Correlation Heatmap
        if not self.corr_matrix.empty:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
            sns.heatmap(self.corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       fmt='.2f', linewidths=0.5, mask=mask, square=True)
            plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            heatmap_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(heatmap_file)
        
        # 2. Missing Values Visualization
        if self.missing_values.sum() > 0:
            plt.figure(figsize=(12, 8))
            missing_data = self.missing_values[self.missing_values > 0].sort_values(ascending=False)
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(missing_data)))
            bars = plt.bar(range(len(missing_data)), missing_data.values, color=colors)
            plt.title('Missing Values by Column', fontsize=16, fontweight='bold')
            plt.xlabel('Columns', fontsize=12)
            plt.ylabel('Number of Missing Values', fontsize=12)
            plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, missing_data.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            missing_file = os.path.join(self.output_dir, 'missing_values.png')
            plt.savefig(missing_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(missing_file)
        
        # 3. Outliers Visualization
        if not self.outliers.empty and self.outliers.sum() > 0:
            plt.figure(figsize=(12, 8))
            outlier_data = self.outliers[self.outliers > 0].sort_values(ascending=False)
            colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(outlier_data)))
            bars = plt.bar(range(len(outlier_data)), outlier_data.values, color=colors)
            plt.title('Outliers Detected by Column (IQR Method)', fontsize=16, fontweight='bold')
            plt.xlabel('Columns', fontsize=12)
            plt.ylabel('Number of Outliers', fontsize=12)
            plt.xticks(range(len(outlier_data)), outlier_data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, outlier_data.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            outliers_file = os.path.join(self.output_dir, 'outliers_detection.png')
            plt.savefig(outliers_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(outliers_file)
        
        # 4. Data Distribution Plots
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Create subplot grid for distributions
            n_cols = min(3, len(numeric_columns))
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns[:9]):  # Limit to 9 plots
                ax = axes[i] if len(numeric_columns) > 1 else axes
                
                # Remove outliers for better visualization
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = self.df[col][(self.df[col] >= Q1 - 1.5*IQR) & (self.df[col] <= Q3 + 1.5*IQR)]
                
                sns.histplot(filtered_data, kde=True, ax=ax, color='skyblue', alpha=0.7)
                ax.set_title(f'Distribution of {col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Data Distributions', fontsize=16, fontweight='bold')
            plt.tight_layout()
            dist_file = os.path.join(self.output_dir, 'data_distributions.png')
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(dist_file)
        
        # 5. Data Types Pie Chart
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.dtype_summary)))
        wedges, texts, autotexts = plt.pie(self.dtype_summary.values, 
                                          labels=self.dtype_summary.index, 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        plt.title('Data Types Distribution', fontsize=16, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        dtypes_file = os.path.join(self.output_dir, 'data_types.png')
        plt.savefig(dtypes_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(dtypes_file)
        
        print(f"Generated {len(visualization_files)} visualizations")
        return visualization_files
    
    def generate_narrative_with_nim(self):
        """Generate narrative using Nvidia NIM API"""
        print("Generating narrative using Nvidia NIM...")
        
        # Prepare context for the LLM
        context = self.prepare_analysis_context()
        
        prompt = f"""
        You are a data analyst writing a comprehensive report. Based on the following dataset analysis, 
        create a detailed, engaging narrative that tells the story of this data. 
        
        Structure your response with:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (3-4 main insights)
        3. Data Quality Assessment
        4. Statistical Insights
        5. Recommendations
        
        Make it professional yet accessible, highlighting the most important patterns and insights.
        
        Dataset Analysis:
        {context}
        """
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }
            
            data = {
                "model": "deepseek-ai/deepseek-r1",
                "messages": [
                    {"role": "system", "content": "You are an expert data analyst who creates insightful, professional reports from data analysis results."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                narrative = response.json()['choices'][0]['message']['content'].strip()
                print("Narrative generated successfully!")
                return narrative
            else:
                print(f"Error generating narrative: {response.status_code} - {response.text}")
                return self.generate_fallback_narrative()
                
        except Exception as e:
            print(f"Error with Nvidia NIM API: {e}")
            return self.generate_fallback_narrative()
    
    def prepare_analysis_context(self):
        """Prepare structured context for LLM"""
        context = f"""
        DATASET OVERVIEW:
        - Shape: {self.basic_info['shape'][0]} rows, {self.basic_info['shape'][1]} columns
        - Columns: {', '.join(self.basic_info['columns'])}
        
        SUMMARY STATISTICS:
        {self.summary_stats.to_string() if not self.summary_stats.empty else "No numeric columns found"}
        
        DATA QUALITY:
        - Missing Values: {self.missing_values.sum()} total missing values
        - Columns with missing data: {', '.join([col for col, count in self.missing_values.items() if count > 0])}
        
        OUTLIERS DETECTED:
        {self.outliers.to_string() if not self.outliers.empty else "No outliers detected"}
        
        CORRELATIONS:
        {self.corr_matrix.to_string() if not self.corr_matrix.empty else "No correlations computed"}
        
        DATA TYPES:
        {self.dtype_summary.to_string()}
        """
        return context
    
    def generate_fallback_narrative(self):
        """Generate a basic narrative if API fails"""
        return f"""
        # Data Analysis Report

        ## Executive Summary
        This dataset contains {self.basic_info['shape'][0]} records with {self.basic_info['shape'][1]} features. 
        The analysis reveals various patterns and data quality aspects that provide insights into the underlying structure.

        ## Key Findings
        - Dataset size: {self.basic_info['shape'][0]} rows × {self.basic_info['shape'][1]} columns
        - Missing values: {self.missing_values.sum()} total across all columns
        - Outliers detected: {self.outliers.sum() if not self.outliers.empty else 0} across numeric columns
        - Data types: {len(self.dtype_summary)} different data types present

        ## Data Quality Assessment
        The dataset shows {'good' if self.missing_values.sum() < self.basic_info['shape'][0] * 0.1 else 'moderate'} data quality 
        with {self.missing_values.sum()}/{self.basic_info['shape'][0]} missing values.

        ## Recommendations
        - Address missing values through imputation or removal
        - Investigate outliers for data integrity
        - Consider feature engineering for better insights
        """
    
    def create_summary_report(self, narrative, visualization_files):
        """Create comprehensive summary report"""
        print("Creating summary report...")
        
        report_file = os.path.join(self.output_dir, 'analysis_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset:** {os.path.basename(self.file_path)}\n\n")
            
            # Add narrative
            f.write("## Analysis Narrative\n\n")
            f.write(narrative)
            f.write("\n\n")
            
            # Add technical details
            f.write("## Technical Summary\n\n")
            f.write(f"- **Dataset Shape:** {self.basic_info['shape'][0]} rows × {self.basic_info['shape'][1]} columns\n")
            f.write(f"- **Total Missing Values:** {self.missing_values.sum()}\n")
            f.write(f"- **Outliers Detected:** {self.outliers.sum() if not self.outliers.empty else 0}\n")
            f.write(f"- **Numeric Columns:** {len(self.df.select_dtypes(include=[np.number]).columns)}\n\n")
            
            # Add visualizations
            f.write("## Visualizations\n\n")
            for viz_file in visualization_files:
                viz_name = os.path.basename(viz_file).replace('.png', '').replace('_', ' ').title()
                f.write(f"### {viz_name}\n")
                f.write(f"![{viz_name}]({os.path.basename(viz_file)})\n\n")
            
            # Add detailed statistics
            if not self.summary_stats.empty:
                f.write("## Detailed Statistics\n\n")
                f.write("```\n")
                f.write(self.summary_stats.to_string())
                f.write("\n```\n\n")
            
            # Add missing values details
            if self.missing_values.sum() > 0:
                f.write("## Missing Values Detail\n\n")
                f.write("| Column | Missing Count | Percentage |\n")
                f.write("|--------|---------------|------------|\n")
                for col, count in self.missing_values[self.missing_values > 0].items():
                    percentage = (count / len(self.df)) * 100
                    f.write(f"| {col} | {count} | {percentage:.2f}% |\n")
                f.write("\n")
        
        return report_file
    
    def display_results(self, narrative, visualization_files, report_file):
        """Display results to user"""
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   • File: {os.path.basename(self.file_path)}")
        print(f"   • Shape: {self.basic_info['shape'][0]} rows × {self.basic_info['shape'][1]} columns")
        print(f"   • Missing Values: {self.missing_values.sum()}")
        print(f"   • Outliers: {self.outliers.sum() if not self.outliers.empty else 0}")
        
        print(f"\n📈 GENERATED VISUALIZATIONS ({len(visualization_files)}):")
        for viz_file in visualization_files:
            print(f"   • {os.path.basename(viz_file)}")
        
        print(f"\n📝 NARRATIVE SUMMARY:")
        print("-" * 50)
        print(narrative)
        print("-" * 50)
        
        print(f"\n📄 OUTPUT FILES:")
        print(f"   • Report: {report_file}")
        print(f"   • Visualizations: {self.output_dir}/")
        
        print(f"\n✅ All files saved in: {os.path.abspath(self.output_dir)}")
        print("="*80)
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.load_data():
            return False
        
        if not self.analyze_data():
            return False
        
        visualization_files = self.create_visualizations()
        narrative = self.generate_narrative_with_nim()
        report_file = self.create_summary_report(narrative, visualization_files)
        
        self.display_results(narrative, visualization_files, report_file)
        return True

def main():
    """Main function to handle command line arguments and run analysis"""
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <file_path>")
        print("Supported formats: CSV (.csv), Excel (.xlsx, .xls)")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    
    # Initialize and run analyzer
    analyzer = DataAnalyzer(file_path)
    
    if analyzer.run_analysis():
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()