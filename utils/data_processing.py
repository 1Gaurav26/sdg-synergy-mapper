"""
Advanced Data Processing Utilities for SDG Synergy Mapper v2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Advanced data validation and cleaning utilities"""
    
    @staticmethod
    def validate_sdg_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of SDG dataset
        
        Returns:
            Dict with validation results and recommendations
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "data_quality_score": 0.0
        }
        
        # Check required columns
        required_cols = ["country", "year"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results["errors"].append(f"Missing required columns: {missing_cols}")
            validation_results["is_valid"] = False
        
        # Check data types
        if "year" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["year"]):
                validation_results["warnings"].append("Year column should be numeric")
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_pct > 30:
            validation_results["warnings"].append(f"High missing data percentage: {missing_pct:.1f}%")
        
        # Check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            if col not in ["year"]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                outlier_count += len(outliers)
        
        if outlier_count > len(df) * 0.1:
            validation_results["warnings"].append(f"High number of outliers detected: {outlier_count}")
        
        # Calculate data quality score
        quality_factors = [
            1.0 if validation_results["is_valid"] else 0.0,
            1.0 - (missing_pct / 100),
            1.0 - min(outlier_count / len(df), 0.5)
        ]
        validation_results["data_quality_score"] = np.mean(quality_factors)
        
        return validation_results
    
    @staticmethod
    def clean_sdg_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess SDG data"""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["year"]:
                # Use median imputation for numeric columns
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical missing values
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna("Unknown")
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        return df_clean

class DataEnricher:
    """Data enrichment utilities for SDG analysis"""
    
    @staticmethod
    def add_region_mapping(df: pd.DataFrame) -> pd.DataFrame:
        """Add regional classifications to countries"""
        region_mapping = {
            "USA": "North America",
            "Canada": "North America",
            "Mexico": "North America",
            "Brazil": "South America",
            "Argentina": "South America",
            "Chile": "South America",
            "Germany": "Europe",
            "France": "Europe",
            "UK": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
            "Japan": "Asia",
            "China": "Asia",
            "India": "Asia",
            "South Korea": "Asia",
            "Australia": "Oceania",
            "New Zealand": "Oceania",
            "Kenya": "Africa",
            "Nigeria": "Africa",
            "South Africa": "Africa",
            "Egypt": "Africa"
        }
        
        df["region"] = df["country"].map(region_mapping).fillna("Other")
        return df
    
    @staticmethod
    def add_development_classification(df: pd.DataFrame) -> pd.DataFrame:
        """Add development level classification based on GDP per capita"""
        if "gdp_per_capita" in df.columns:
            def classify_development(gdp):
                if gdp < 1000:
                    return "Least Developed"
                elif gdp < 4000:
                    return "Developing"
                elif gdp < 12000:
                    return "Emerging"
                else:
                    return "Developed"
            
            df["development_level"] = df["gdp_per_capita"].apply(classify_development)
        
        return df
    
    @staticmethod
    def add_sdg_progress_scores(df: pd.DataFrame, sdg_indicators: Dict) -> pd.DataFrame:
        """Calculate SDG progress scores for each goal"""
        sdg_goals = {}
        
        for indicator, info in sdg_indicators.items():
            sdg_num = info["sdg"]
            if indicator in df.columns:
                if sdg_num not in sdg_goals:
                    sdg_goals[sdg_num] = []
                sdg_goals[sdg_num].append(indicator)
        
        # Calculate average score for each SDG
        for sdg_num, indicators in sdg_goals.items():
            available_indicators = [ind for ind in indicators if ind in df.columns]
            if available_indicators:
                df[f"sdg_{sdg_num}_score"] = df[available_indicators].mean(axis=1)
        
        return df

class APIDataFetcher:
    """Fetch data from external APIs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SDG-Synergy-Mapper-v2/1.0'
        })
    
    def fetch_world_bank_data(self, indicators: List[str], countries: List[str], 
                            start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch data from World Bank API"""
        try:
            base_url = self.config["world_bank"]["base_url"]
            url = f"{base_url}/country/all/indicator/{';'.join(indicators)}"
            params = {
                "date": f"{start_year}:{end_year}",
                "format": "json",
                "per_page": 20000
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if len(data) > 1 and data[1]:
                df = pd.DataFrame(data[1])
                return self._process_world_bank_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return pd.DataFrame()
    
    def _process_world_bank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process World Bank API response"""
        if df.empty:
            return df
        
        # Pivot the data
        df_processed = df.pivot_table(
            index=['country', 'date'],
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df_processed.columns.name = None
        df_processed = df_processed.rename(columns={'date': 'year'})
        
        return df_processed
    
    def fetch_un_stats_data(self, indicators: List[str]) -> pd.DataFrame:
        """Fetch data from UN Statistics API"""
        try:
            base_url = self.config["un_stats"]["base_url"]
            url = f"{base_url}/sdg/data"
            
            params = {
                "indicator": ",".join(indicators),
                "format": "json"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return pd.DataFrame(data.get("data", []))
            
        except Exception as e:
            logger.error(f"Error fetching UN Stats data: {e}")
            return pd.DataFrame()

class DataAggregator:
    """Advanced data aggregation utilities"""
    
    @staticmethod
    def calculate_regional_averages(df: pd.DataFrame, 
                                  group_cols: List[str] = None) -> pd.DataFrame:
        """Calculate regional averages for SDG indicators"""
        if group_cols is None:
            group_cols = ["region", "year"]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ["year"]]
        
        regional_data = df.groupby(group_cols)[numeric_cols].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Flatten column names
        regional_data.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in regional_data.columns.values]
        
        return regional_data
    
    @staticmethod
    def calculate_trend_analysis(df: pd.DataFrame, 
                                indicators: List[str]) -> Dict[str, Any]:
        """Calculate trend analysis for indicators"""
        trends = {}
        
        for indicator in indicators:
            if indicator in df.columns and "year" in df.columns:
                # Calculate linear trend
                years = df["year"].values
                values = df[indicator].values
                
                # Remove NaN values
                mask = ~np.isnan(values)
                if np.sum(mask) > 1:
                    years_clean = years[mask]
                    values_clean = values[mask]
                    
                    # Linear regression
                    coeffs = np.polyfit(years_clean, values_clean, 1)
                    slope = coeffs[0]
                    
                    # Calculate R-squared
                    y_pred = np.polyval(coeffs, years_clean)
                    ss_res = np.sum((values_clean - y_pred) ** 2)
                    ss_tot = np.sum((values_clean - np.mean(values_clean)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    trends[indicator] = {
                        "slope": slope,
                        "r_squared": r_squared,
                        "trend_direction": "increasing" if slope > 0 else "decreasing",
                        "trend_strength": "strong" if abs(r_squared) > 0.7 else 
                                        "moderate" if abs(r_squared) > 0.4 else "weak"
                    }
        
        return trends

class DataExporter:
    """Advanced data export utilities"""
    
    @staticmethod
    def export_to_multiple_formats(data: Dict[str, pd.DataFrame], 
                                 base_filename: str,
                                 formats: List[str] = None) -> Dict[str, str]:
        """Export data to multiple formats"""
        if formats is None:
            formats = ["csv", "excel", "json"]
        
        exported_files = {}
        
        for format_type in formats:
            try:
                if format_type == "csv":
                    for name, df in data.items():
                        filename = f"{base_filename}_{name}.csv"
                        df.to_csv(filename, index=False)
                        exported_files[name] = filename
                
                elif format_type == "excel":
                    filename = f"{base_filename}.xlsx"
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        for name, df in data.items():
                            df.to_excel(writer, sheet_name=name, index=False)
                    exported_files["excel"] = filename
                
                elif format_type == "json":
                    filename = f"{base_filename}.json"
                    json_data = {}
                    for name, df in data.items():
                        json_data[name] = df.to_dict('records')
                    
                    with open(filename, 'w') as f:
                        json.dump(json_data, f, indent=2, default=str)
                    exported_files["json"] = filename
                    
            except Exception as e:
                logger.error(f"Error exporting to {format_type}: {e}")
        
        return exported_files

