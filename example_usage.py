"""
Example Usage Script for SDG Synergy Mapper v2
Demonstrates key features and capabilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.settings import SDG_INDICATORS, CORRELATION_THRESHOLDS
from utils.data_processing import DataValidator, DataEnricher, DataAggregator
from models.ml_models import SDGClusteringModel, SDGPredictionModel, SDGDimensionalityReduction
from utils.visualization import AdvancedVisualizer

def create_sample_data():
    """Create sample SDG data for demonstration"""
    print("📊 Creating sample SDG data...")
    
    countries = ["USA", "China", "India", "Brazil", "Germany", "Japan", "Kenya", "Nigeria"]
    years = list(range(2015, 2024))
    
    data = []
    np.random.seed(42)  # For reproducible results
    
    for country in countries:
        for year in years:
            # Generate realistic SDG data with some correlations
            base_gdp = np.random.normal(30000, 15000) if country in ["USA", "Germany", "Japan"] else np.random.normal(8000, 4000)
            
            row = {
                "country": country,
                "year": year,
                "gdp_per_capita": max(1000, base_gdp + np.random.normal(0, 1000)),
                "life_expectancy": max(50, 75 + np.random.normal(0, 5)),
                "poverty_rate": max(0, 20 + np.random.normal(0, 5)),
                "renewable_energy_pct": max(0, min(100, 30 + np.random.normal(0, 10))),
                "co2_emissions": max(0, 5 + np.random.normal(0, 2)),
                "forest_area_pct": max(0, min(100, 40 + np.random.normal(0, 15))),
                "internet_penetration": max(0, min(100, 60 + np.random.normal(0, 20))),
                "employment_rate": max(0, min(100, 70 + np.random.normal(0, 10))),
                "literacy_rate": max(0, min(100, 85 + np.random.normal(0, 10)))
            }
            
            # Add some correlations
            row["life_expectancy"] = max(50, row["life_expectancy"] + (row["gdp_per_capita"] - 15000) / 1000)
            row["poverty_rate"] = max(0, row["poverty_rate"] - (row["gdp_per_capita"] - 15000) / 2000)
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ Created dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def demonstrate_data_processing(df):
    """Demonstrate data processing capabilities"""
    print("\n🔍 Data Processing Demonstration")
    print("-" * 40)
    
    # Data validation
    validator = DataValidator()
    validation_results = validator.validate_sdg_data(df)
    
    print(f"Data Quality Score: {validation_results['data_quality_score']:.2f}")
    print(f"Missing Data: {validation_results.get('missing_pct', 0):.1f}%")
    
    if validation_results['warnings']:
        print("Warnings:")
        for warning in validation_results['warnings'][:3]:
            print(f"  • {warning}")
    
    # Data enrichment
    enricher = DataEnricher()
    df_enriched = enricher.add_region_mapping(df)
    df_enriched = enricher.add_development_classification(df_enriched)
    df_enriched = enricher.add_sdg_progress_scores(df_enriched, SDG_INDICATORS)
    
    print(f"✅ Data enriched with {len(df_enriched.columns)} columns")
    
    # Data aggregation
    aggregator = DataAggregator()
    regional_data = aggregator.calculate_regional_averages(df_enriched)
    
    print(f"✅ Regional analysis completed for {len(regional_data)} regions")
    
    return df_enriched

def demonstrate_ml_models(df):
    """Demonstrate machine learning capabilities"""
    print("\n🤖 Machine Learning Demonstration")
    print("-" * 40)
    
    # Select indicators for analysis
    indicators = ["gdp_per_capita", "life_expectancy", "poverty_rate", "renewable_energy_pct", "co2_emissions"]
    
    # Clustering analysis
    print("🎯 Running clustering analysis...")
    clustering_model = SDGClusteringModel(n_clusters=3, algorithm="kmeans")
    clustering_results = clustering_model.fit(df, indicators)
    
    if clustering_results["success"]:
        print(f"✅ Clustering completed: {clustering_results['n_clusters_found']} clusters found")
        
        for cluster_id, analysis in clustering_results["cluster_analysis"].items():
            print(f"  Cluster {cluster_id.split('_')[1]}: {analysis['size']} countries")
            print(f"    Countries: {', '.join(analysis['countries'][:3])}...")
    
    # Prediction analysis
    print("\n🔮 Running prediction analysis...")
    prediction_model = SDGPredictionModel(model_type="random_forest")
    prediction_results = prediction_model.fit(df, "life_expectancy", ["gdp_per_capita", "poverty_rate"])
    
    if prediction_results["success"]:
        print(f"✅ Prediction model trained successfully")
        print(f"  R² Score: {prediction_results['performance']['r2']:.3f}")
        print(f"  RMSE: {prediction_results['performance']['rmse']:.3f}")
    
    # Dimensionality reduction
    print("\n📐 Running dimensionality reduction...")
    dim_reduction = SDGDimensionalityReduction(method="pca", n_components=2)
    reduction_results = dim_reduction.fit_transform(df, indicators)
    
    if reduction_results["success"]:
        print(f"✅ Dimensionality reduction completed")
        if reduction_results["explained_variance"]:
            print(f"  Explained variance: {sum(reduction_results['explained_variance']):.1%}")

def demonstrate_visualizations(df):
    """Demonstrate visualization capabilities"""
    print("\n📊 Visualization Demonstration")
    print("-" * 40)
    
    visualizer = AdvancedVisualizer()
    
    # Create radar chart
    print("🎯 Creating radar chart...")
    countries = df["country"].unique()[:3]
    indicators = ["gdp_per_capita", "life_expectancy", "poverty_rate", "renewable_energy_pct"]
    
    try:
        radar_fig = visualizer.create_radar_chart(df, indicators, countries)
        print(f"✅ Radar chart created for {len(countries)} countries")
    except Exception as e:
        print(f"⚠️ Radar chart creation failed: {e}")
    
    # Create correlation heatmap data
    print("🔥 Creating correlation analysis...")
    numeric_data = df[indicators].select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    print(f"✅ Correlation matrix calculated ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]})")
    
    # Show strongest correlations
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append({
                "indicator1": correlation_matrix.columns[i],
                "indicator2": correlation_matrix.columns[j],
                "correlation": corr_value
            })
    
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    print("Top correlations:")
    for corr in correlations[:3]:
        print(f"  • {corr['indicator1']} ↔ {corr['indicator2']}: {corr['correlation']:.3f}")

def demonstrate_api_integration():
    """Demonstrate API integration capabilities"""
    print("\n🔌 API Integration Demonstration")
    print("-" * 40)
    
    try:
        from utils.api_integration import SDGDataAPIClient, DataCache
        
        # Initialize API client
        config = {
            "world_bank": {"base_url": "https://api.worldbank.org/v2", "enabled": True},
            "un_stats": {"base_url": "https://unstats.un.org/sdgapi/v1", "enabled": True},
            "oecd": {"base_url": "https://sdmx.oecd.org/public/rest/data", "enabled": False}
        }
        
        api_client = SDGDataAPIClient(config)
        cache = DataCache()
        
        print("✅ API client initialized")
        print("ℹ️ Note: Actual API calls require internet connection and may take time")
        
        # Example of how to fetch data (commented out to avoid actual API calls)
        # indicators = ["gdp_per_capita", "life_expectancy"]
        # countries = ["USA", "China"]
        # data = api_client.fetch_combined_data(indicators, countries)
        # print(f"✅ Fetched {len(data)} records from APIs")
        
    except ImportError as e:
        print(f"⚠️ API integration not available: {e}")

def main():
    """Main demonstration function"""
    print("🌐 SDG Synergy Mapper v2 - Example Usage")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Demonstrate data processing
    df_enriched = demonstrate_data_processing(df)
    
    # Demonstrate ML models
    demonstrate_ml_models(df_enriched)
    
    # Demonstrate visualizations
    demonstrate_visualizations(df_enriched)
    
    # Demonstrate API integration
    demonstrate_api_integration()
    
    print("\n" + "=" * 50)
    print("🎉 Demonstration completed successfully!")
    print("\nTo run the full application:")
    print("1. Run: streamlit run app/main.py")
    print("2. Open: http://localhost:8501")
    print("3. Load the sample data and explore the features")

if __name__ == "__main__":
    main()

