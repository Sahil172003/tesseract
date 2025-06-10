from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64
import io
import json
import requests
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/data-visualization", tags=["data-visualization"])

class DataAnalysisRequest(BaseModel):
    visualization_request: str
    chart_type: Optional[str] = "auto"

class DataAnalysisResponse(BaseModel):
    status: str
    message: str
    data_preview: Optional[str] = None
    summary_stats: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None
    narrative: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None

class DataAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.api_token = os.getenv("NVIDIA_NIM_API_KEY")
        
    def load_data(self):
        """Load CSV or Excel file"""
        try:
            file_extension = Path(self.file_path).suffix.lower()
            
            if file_extension in ['.csv']:
                encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(self.file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not read CSV with any supported encoding")
                    
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
                
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def analyze_data(self):
        """Perform comprehensive data analysis"""
        # Basic info
        self.basic_info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.to_dict().items()}
        }
        
        # Summary statistics for numerical columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.summary_stats = numeric_df.describe().to_dict() if not numeric_df.empty else {}
        
        # Missing values
        self.missing_values = self.df.isnull().sum().to_dict()
        
        # Correlation matrix for numerical columns
        self.corr_matrix = numeric_df.corr().to_dict() if not numeric_df.empty else {}
        
        # Detect outliers using IQR method
        self.outliers = self.detect_outliers(numeric_df)
        
        return True
    
    def detect_outliers(self, numeric_df):
        """Detect outliers using IQR method"""
        if numeric_df.empty:
            return {}
            
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
        return outliers.to_dict()
    
    def create_chart_data(self, chart_type: str = "auto"):
        """Create chart data for frontend"""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if chart_type == "auto" or chart_type == "bar":
            # Create bar chart data - use first numeric column or create summary
            if len(numeric_columns) > 0:
                col = numeric_columns[0]
                # Group by categories if possible, otherwise create bins
                if len(self.df[col].unique()) <= 20:
                    chart_data = self.df[col].value_counts().head(10).to_dict()
                    return {
                        "type": "bar",
                        "data": [{"name": str(k), "value": int(v)} for k, v in chart_data.items()]
                    }
                else:
                    # Create histogram bins
                    hist_data = pd.cut(self.df[col], bins=10).value_counts()
                    return {
                        "type": "bar", 
                        "data": [{"name": str(k), "value": int(v)} for k, v in hist_data.items()]
                    }
        
        elif chart_type == "line":
            if len(numeric_columns) > 0:
                col = numeric_columns[0]
                # Use index as x-axis for line chart
                sample_data = self.df[col].head(20)
                return {
                    "type": "line",
                    "data": [{"name": str(i), "value": float(v)} for i, v in enumerate(sample_data)]
                }
        
        elif chart_type == "pie":
            # Use categorical data for pie chart
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                pie_data = self.df[col].value_counts().head(6).to_dict()
                return {
                    "type": "pie",
                    "data": [{"name": str(k), "value": int(v)} for k, v in pie_data.items()]
                }
        
        # Fallback data
        return {
            "type": "bar",
            "data": [
                {"name": "Sample A", "value": 400},
                {"name": "Sample B", "value": 300},
                {"name": "Sample C", "value": 600}
            ]
        }
    
    def create_visualizations(self):
        """Generate visualizations and return as base64 strings"""
        plt.style.use('default')
        visualizations = []
        
        # 1. Correlation Heatmap
        if self.corr_matrix:
            plt.figure(figsize=(10, 8))
            corr_df = pd.DataFrame(self.corr_matrix)
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
                       fmt='.2f', linewidths=0.5, mask=mask, square=True)
            plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            visualizations.append(f"data:image/png;base64,{img_str}")
            plt.close()
        
        # 2. Missing Values Chart
        missing_data = {k: v for k, v in self.missing_values.items() if v > 0}
        if missing_data:
            plt.figure(figsize=(10, 6))
            names = list(missing_data.keys())
            values = list(missing_data.values())
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(names)))
            bars = plt.bar(range(len(names)), values, color=colors)
            plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
            plt.xlabel('Columns')
            plt.ylabel('Number of Missing Values')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            visualizations.append(f"data:image/png;base64,{img_str}")
            plt.close()
        
        # 3. Data Distribution
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns[:4]):
                ax = axes[i]
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = self.df[col][(self.df[col] >= Q1 - 1.5*IQR) & (self.df[col] <= Q3 + 1.5*IQR)]
                
                sns.histplot(filtered_data, kde=True, ax=ax, color='skyblue', alpha=0.7)
                ax.set_title(f'Distribution of {col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_columns), 4):
                axes[i].set_visible(False)
            
            plt.suptitle('Data Distributions', fontsize=16, fontweight='bold')
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            visualizations.append(f"data:image/png;base64,{img_str}")
            plt.close()
        
        return visualizations
    
    def generate_narrative(self, user_request: str = ""):
        """Generate narrative using Nvidia NIM API or fallback"""
        context = self.prepare_analysis_context()
        
        prompt = f"""
        You are a data analyst. Based on the dataset analysis below, create a clear, concise narrative.
        
        User Request: {user_request}
        
        Dataset Analysis:
        {context}
        
        Provide:
        1. Brief summary (2-3 sentences)
        2. Key findings (2-3 points)
        3. Data quality notes
        4. Recommendations
        
        Keep it professional and accessible.
        """
        
        try:
            if self.api_token:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}"
                }
                
                data = {
                    "model": "deepseek-ai/deepseek-r1",
                    "messages": [
                        {"role": "system", "content": "You are an expert data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.7
                }
                
                response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", 
                                       headers=headers, data=json.dumps(data), timeout=30)
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API error: {e}")
        
        return self.generate_fallback_narrative()
    
    def prepare_analysis_context(self):
        """Prepare analysis context"""
        return f"""
        Dataset Shape: {self.basic_info['shape'][0]} rows, {self.basic_info['shape'][1]} columns
        Columns: {', '.join(self.basic_info['columns'])}
        Missing Values: {sum(self.missing_values.values())} total
        Numeric Columns: {len([c for c in self.basic_info['columns'] if c in self.summary_stats])}
        """
    
    def generate_fallback_narrative(self):
        """Generate fallback narrative"""
        return f"""
        ## Dataset Analysis Summary
        
        This dataset contains {self.basic_info['shape'][0]} records with {self.basic_info['shape'][1]} features.
        
        **Key Findings:**
        - Dataset size: {self.basic_info['shape'][0]} rows × {self.basic_info['shape'][1]} columns
        - Missing values: {sum(self.missing_values.values())} total
        - Numeric columns: {len([c for c in self.basic_info['columns'] if c in self.summary_stats])}
        
        **Data Quality:** {'Good' if sum(self.missing_values.values()) < self.basic_info['shape'][0] * 0.1 else 'Needs attention'}
        
        **Recommendations:**
        - Review missing values for data completeness
        - Consider data validation and cleaning
        - Explore relationships between variables
        """

# Global variable to store current analyzer
current_analyzer = None

@router.post("/upload", response_model=DataAnalysisResponse)
async def upload_and_analyze_data(file: UploadFile = File(...)):
    """Upload and analyze data file"""
    global current_analyzer
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Initialize analyzer
        current_analyzer = DataAnalyzer(tmp_file_path)
        
        # Load and analyze data
        if not current_analyzer.load_data():
            raise HTTPException(status_code=400, detail="Failed to load data file")
        
        current_analyzer.analyze_data()
        
        # Generate data preview
        data_preview = current_analyzer.df.head(5).to_string()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return DataAnalysisResponse(
            status="success",
            message="Data uploaded and analyzed successfully",
            data_preview=data_preview,
            summary_stats=current_analyzer.summary_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/visualize", response_model=DataAnalysisResponse)
async def create_visualization(request: DataAnalysisRequest):
    """Create visualization based on user request"""
    global current_analyzer
    
    if current_analyzer is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    try:
        # Generate visualizations
        visualizations = current_analyzer.create_visualizations()
        
        # Create chart data for frontend
        chart_data = current_analyzer.create_chart_data(request.chart_type)
        
        # Generate narrative
        narrative = current_analyzer.generate_narrative(request.visualization_request)
        
        return DataAnalysisResponse(
            status="success",
            message="Visualization created successfully",
            visualizations=visualizations,
            narrative=narrative,
            chart_data=chart_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {"message": "Data visualization endpoint is working"}

@router.post("/clear")
async def clear_data():
    """Clear current data session"""
    global current_analyzer
    current_analyzer = None
    return {"message": "Data cleared successfully"}