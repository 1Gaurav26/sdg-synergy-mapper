"""
SDG Synergy Mapper v2 - Advanced Analysis Platform
Enhanced with ML, Real-time Data, and Advanced Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SDG_INDICATORS, CORRELATION_THRESHOLDS, ML_MODELS, CHART_THEMES
from utils.data_processing import DataValidator, DataEnricher, APIDataFetcher, DataAggregator, DataExporter
from models.ml_models import SDGClusteringModel, SDGPredictionModel, SDGDimensionalityReduction, SDGAnomalyDetector, ModelManager
from utils.visualization import AdvancedVisualizer, ExportVisualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SDG Synergy Mapper v2",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class SDGSynergyMapperV2:
    """Main application class for SDG Synergy Mapper v2"""
    
    def __init__(self):
        self.data_validator = DataValidator()
        self.data_enricher = DataEnricher()
        self.data_aggregator = DataAggregator()
        self.data_exporter = DataExporter()
        self.visualizer = AdvancedVisualizer()
        self.model_manager = ModelManager()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'ml_models' not in st.session_state:
            st.session_state.ml_models = {}
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🌐 SDG Synergy Mapper v2</h1>', unsafe_allow_html=True)
        st.markdown("**Advanced Analytics Platform for Sustainable Development Goals**")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", "🔍 Analysis", "🤖 ML Models", "🌍 Geospatial", 
            "📈 Trends", "📤 Export"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_analysis()
        
        with tab3:
            self.render_ml_models()
        
        with tab4:
            self.render_geospatial()
        
        with tab5:
            self.render_trends()
        
        with tab6:
            self.render_export()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Control Panel")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "📁 Data Source",
            ["Upload CSV", "Sample Data", "API Integration", "Database"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload your SDG dataset"
            )
            if uploaded_file is not None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Data loaded successfully!")
        
        elif data_source == "Sample Data":
            if st.sidebar.button("Load Sample Data"):
                sample_path = Path(__file__).parent.parent / "data" / "sdg_sample_data.csv"
                if sample_path.exists():
                    st.session_state.data = pd.read_csv(sample_path)
                    st.sidebar.success("✅ Sample data loaded!")
                else:
                    st.sidebar.error("Sample data not found")
        
        # Data validation
        if st.session_state.data is not None:
            st.sidebar.subheader("🔍 Data Quality")
            validation_results = self.data_validator.validate_sdg_data(st.session_state.data)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Quality Score", f"{validation_results['data_quality_score']:.2f}")
            with col2:
                st.metric("Missing Data", f"{validation_results.get('missing_pct', 0):.1f}%")
            
            if validation_results['warnings']:
                st.sidebar.warning("⚠️ Data Quality Issues")
                for warning in validation_results['warnings'][:3]:
                    st.sidebar.text(f"• {warning}")
        
        # Analysis parameters
        st.sidebar.subheader("⚙️ Analysis Settings")
        
        # Country selection
        if st.session_state.data is not None and "country" in st.session_state.data.columns:
            countries = st.sidebar.multiselect(
                "🌍 Select Countries",
                options=sorted(st.session_state.data["country"].unique()),
                default=sorted(st.session_state.data["country"].unique())[:5]
            )
            st.session_state.selected_countries = countries
        
        # SDG categories
        categories = st.sidebar.multiselect(
            "📋 SDG Categories",
            options=["Social", "Environment", "Economy", "Governance"],
            default=["Social", "Environment", "Economy"]
        )
        st.session_state.selected_categories = categories
        
        # Correlation threshold
        threshold = st.sidebar.slider(
            "🔗 Correlation Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        st.session_state.correlation_threshold = threshold
        
        # Year range
        if st.session_state.data is not None and "year" in st.session_state.data.columns:
            years = sorted(st.session_state.data["year"].unique())
            year_range = st.sidebar.slider(
                "📅 Year Range",
                min_value=int(years[0]),
                max_value=int(years[-1]),
                value=(int(years[0]), int(years[-1]))
            )
            st.session_state.year_range = year_range
    
    def render_dashboard(self):
        """Render main dashboard"""
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.header("📊 SDG Analysis Dashboard")
        
        # Data enrichment
        df_enriched = self.data_enricher.add_region_mapping(st.session_state.data)
        df_enriched = self.data_enricher.add_development_classification(df_enriched)
        df_enriched = self.data_enricher.add_sdg_progress_scores(df_enriched, SDG_INDICATORS)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Countries Analyzed",
                len(df_enriched["country"].unique()),
                delta=None
            )
        
        with col2:
            st.metric(
                "Indicators Tracked",
                len([col for col in df_enriched.columns if col in SDG_INDICATORS]),
                delta=None
            )
        
        with col3:
            if "year" in df_enriched.columns:
                year_range = f"{int(df_enriched['year'].min())}-{int(df_enriched['year'].max())}"
                st.metric("Time Period", year_range)
        
        with col4:
            st.metric(
                "Data Points",
                len(df_enriched),
                delta=None
            )
        
        # Main visualizations
        st.subheader("🎯 SDG Performance Overview")
        
        # SDG scores radar chart
        if any(f"sdg_{i}_score" in df_enriched.columns for i in range(1, 18)):
            sdg_score_cols = [col for col in df_enriched.columns if col.startswith("sdg_") and col.endswith("_score")]
            if sdg_score_cols:
                countries = st.session_state.get("selected_countries", df_enriched["country"].unique()[:5])
                
                fig_radar = self.visualizer.create_radar_chart(
                    df_enriched, 
                    sdg_score_cols, 
                    countries
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # Regional comparison
        st.subheader("🌍 Regional Comparison")
        
        if "region" in df_enriched.columns:
            regional_data = self.data_aggregator.calculate_regional_averages(df_enriched)
            
            # Select indicator for regional comparison
            numeric_cols = [col for col in df_enriched.columns 
                          if col in SDG_INDICATORS and col not in ["year", "country"]]
            
            if numeric_cols:
                selected_indicator = st.selectbox("Select Indicator", numeric_cols)
                
                if selected_indicator in regional_data.columns:
                    fig_regional = px.box(
                        df_enriched,
                        x="region",
                        y=selected_indicator,
                        title=f"{selected_indicator} by Region",
                        color="region"
                    )
                    st.plotly_chart(fig_regional, use_container_width=True)
        
        # Development level analysis
        st.subheader("🏗️ Development Level Analysis")
        
        if "development_level" in df_enriched.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Development level distribution
                dev_counts = df_enriched["development_level"].value_counts()
                fig_pie = px.pie(
                    values=dev_counts.values,
                    names=dev_counts.index,
                    title="Development Level Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Development level trends
                if "year" in df_enriched.columns:
                    dev_trends = df_enriched.groupby(["year", "development_level"]).size().reset_index(name="count")
                    fig_trend = px.line(
                        dev_trends,
                        x="year",
                        y="count",
                        color="development_level",
                        title="Development Level Trends"
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
    
    def render_analysis(self):
        """Render correlation analysis"""
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.header("🔍 Advanced Correlation Analysis")
        
        # Filter data
        df_filtered = self._filter_data()
        
        # Select indicators for analysis
        available_indicators = [col for col in df_filtered.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicators = st.multiselect(
            "Select Indicators for Analysis",
            options=available_indicators,
            default=available_indicators[:10],
            format_func=lambda x: f"{x} — {SDG_INDICATORS[x]['sdg']}"
        )
        
        if not selected_indicators:
            st.warning("Please select at least 2 indicators")
            return
        
        # Calculate correlations
        numeric_data = df_filtered[selected_indicators].select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="SDG Indicators Correlation Matrix"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Network visualization
        st.subheader("🕸️ SDG Synergy Network")
        
        threshold = st.session_state.get("correlation_threshold", 0.5)
        network = self.visualizer.create_network_visualization(correlation_matrix, threshold)
        
        # Render network
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            network.save_graph(tmp.name)
            tmp_path = tmp.name
        
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        
        st.components.v1.html(html_code, height=600)
        
        try:
            os.remove(tmp_path)
        except:
            pass
        
        # Correlation insights
        st.subheader("💡 Correlation Insights")
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    correlations.append({
                        "indicator1": correlation_matrix.columns[i],
                        "indicator2": correlation_matrix.columns[j],
                        "correlation": corr_value,
                        "strength": "Strong" if abs(corr_value) >= 0.7 else "Moderate"
                    })
        
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        for corr in correlations[:10]:
            direction = "positive" if corr["correlation"] > 0 else "negative"
            st.write(f"• **{corr['indicator1']}** and **{corr['indicator2']}** have a {corr['strength'].lower()} {direction} correlation ({corr['correlation']:.3f})")
    
    def render_ml_models(self):
        """Render machine learning models section"""
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.header("🤖 Machine Learning Analysis")
        
        df_filtered = self._filter_data()
        
        # Model selection
        model_type = st.selectbox(
            "Select Analysis Type",
            ["Clustering", "Prediction", "Dimensionality Reduction", "Anomaly Detection"]
        )
        
        if model_type == "Clustering":
            self._render_clustering_analysis(df_filtered)
        elif model_type == "Prediction":
            self._render_prediction_analysis(df_filtered)
        elif model_type == "Dimensionality Reduction":
            self._render_dimensionality_reduction(df_filtered)
        elif model_type == "Anomaly Detection":
            self._render_anomaly_detection(df_filtered)
    
    def _render_clustering_analysis(self, df):
        """Render clustering analysis"""
        st.subheader("🎯 Country Clustering Analysis")
        
        # Select indicators for clustering
        available_indicators = [col for col in df.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicators = st.multiselect(
            "Select Indicators for Clustering",
            options=available_indicators,
            default=available_indicators[:8] if len(available_indicators) >= 8 else available_indicators
        )
        
        if len(selected_indicators) < 2:
            st.warning("Please select at least 2 indicators")
            return
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        with col2:
            algorithm = st.selectbox("Clustering Algorithm", ["kmeans", "hdbscan"])
        
        if st.button("Run Clustering Analysis"):
            with st.spinner("Running clustering analysis..."):
                clustering_model = SDGClusteringModel(n_clusters=n_clusters, algorithm=algorithm)
                results = clustering_model.fit(df, selected_indicators)
                
                if results["success"]:
                    st.success("✅ Clustering analysis completed!")
                    
                    # Display cluster analysis
                    st.subheader("📊 Cluster Analysis Results")
                    
                    for cluster_id, analysis in results["cluster_analysis"].items():
                        with st.expander(f"Cluster {cluster_id.split('_')[1]}"):
                            st.write(f"**Size:** {analysis['size']} countries")
                            st.write(f"**Countries:** {', '.join(analysis['countries'])}")
                            st.write(f"**Characteristics:** {analysis['characteristics']}")
                            
                            # Show average values
                            avg_values = analysis['avg_values']
                            for indicator, value in avg_values.items():
                                st.write(f"• {indicator}: {value:.2f}")
                    
                    # Visualize clusters
                    st.subheader("📈 Cluster Visualization")
                    
                    # Create 2D projection for visualization
                    dim_reduction = SDGDimensionalityReduction(method="pca", n_components=2)
                    projection_results = dim_reduction.fit_transform(df, selected_indicators)
                    
                    if projection_results["success"]:
                        projection_df = projection_results["transformed_data"]
                        projection_df["cluster"] = results["cluster_labels"]
                        
                        fig_clusters = px.scatter(
                            projection_df,
                            x="PCA_1",
                            y="PCA_2",
                            color="cluster",
                            hover_data=["country"],
                            title="Country Clusters (2D Projection)"
                        )
                        st.plotly_chart(fig_clusters, use_container_width=True)
                
                else:
                    st.error(f"❌ Clustering failed: {results['error']}")
    
    def _render_prediction_analysis(self, df):
        """Render prediction analysis"""
        st.subheader("🔮 SDG Indicator Prediction")
        
        # Select target and feature indicators
        available_indicators = [col for col in df.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        col1, col2 = st.columns(2)
        with col1:
            target_indicator = st.selectbox("Target Indicator", available_indicators)
        with col2:
            feature_options = [col for col in available_indicators if col != target_indicator]
            feature_indicators = st.multiselect(
                "Feature Indicators",
                options=feature_options,
                default=feature_options[:5] if len(feature_options) >= 5 else feature_options
            )
        
        if not feature_indicators:
            st.warning("Please select feature indicators")
            return
        
        model_type = st.selectbox("Prediction Model", ["random_forest", "linear_regression"])
        
        if st.button("Train Prediction Model"):
            with st.spinner("Training prediction model..."):
                prediction_model = SDGPredictionModel(model_type=model_type)
                results = prediction_model.fit(df, target_indicator, feature_indicators)
                
                if results["success"]:
                    st.success("✅ Prediction model trained successfully!")
                    
                    # Display model performance
                    st.subheader("📊 Model Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{results['performance']['r2']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{results['performance']['rmse']:.3f}")
                    with col3:
                        st.metric("MSE", f"{results['performance']['mse']:.3f}")
                    
                    # Feature importance
                    if results['feature_importance']:
                        st.subheader("🎯 Feature Importance")
                        
                        importance_df = pd.DataFrame(
                            list(results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                else:
                    st.error(f"❌ Prediction failed: {results['error']}")
    
    def _render_dimensionality_reduction(self, df):
        """Render dimensionality reduction analysis"""
        st.subheader("📐 Dimensionality Reduction")
        
        # Select indicators
        available_indicators = [col for col in df.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicators = st.multiselect(
            "Select Indicators",
            options=available_indicators,
            default=available_indicators[:10] if len(available_indicators) >= 10 else available_indicators
        )
        
        if len(selected_indicators) < 3:
            st.warning("Please select at least 3 indicators")
            return
        
        method = st.selectbox("Reduction Method", ["pca", "umap", "tsne"])
        
        if st.button("Run Dimensionality Reduction"):
            with st.spinner("Running dimensionality reduction..."):
                dim_reduction = SDGDimensionalityReduction(method=method, n_components=2)
                results = dim_reduction.fit_transform(df, selected_indicators)
                
                if results["success"]:
                    st.success("✅ Dimensionality reduction completed!")
                    
                    # Display explained variance for PCA
                    if method == "pca" and results["explained_variance"]:
                        st.subheader("📊 Explained Variance")
                        explained_var = results["explained_variance"]
                        st.write(f"Component 1: {explained_var[0]:.1%}")
                        st.write(f"Component 2: {explained_var[1]:.1%}")
                        st.write(f"Total: {sum(explained_var):.1%}")
                    
                    # Visualize results
                    st.subheader("📈 2D Projection")
                    
                    projection_df = results["transformed_data"]
                    
                    fig_projection = px.scatter(
                        projection_df,
                        x=f"{method.upper()}_1",
                        y=f"{method.upper()}_2",
                        hover_data=["country", "year"],
                        title=f"{method.upper()} Projection of SDG Indicators"
                    )
                    st.plotly_chart(fig_projection, use_container_width=True)
                
                else:
                    st.error(f"❌ Dimensionality reduction failed: {results['error']}")
    
    def _render_anomaly_detection(self, df):
        """Render anomaly detection analysis"""
        st.subheader("🚨 Anomaly Detection")
        
        # Select indicators
        available_indicators = [col for col in df.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicators = st.multiselect(
            "Select Indicators for Anomaly Detection",
            options=available_indicators,
            default=available_indicators[:8] if len(available_indicators) >= 8 else available_indicators
        )
        
        if len(selected_indicators) < 2:
            st.warning("Please select at least 2 indicators")
            return
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                anomaly_detector = SDGAnomalyDetector()
                results = anomaly_detector.detect_anomalies(df, selected_indicators)
                
                if results["success"]:
                    st.success("✅ Anomaly detection completed!")
                    
                    # Display anomaly summary
                    st.subheader("📊 Anomaly Summary")
                    
                    analysis = results["anomaly_analysis"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Anomalies", analysis["total_anomalies"])
                    with col2:
                        st.metric("Anomaly Rate", f"{analysis['anomaly_percentage']:.1f}%")
                    with col3:
                        st.metric("Affected Countries", len(analysis["anomaly_countries"]))
                    
                    # Show anomalous countries
                    if analysis["anomaly_countries"]:
                        st.subheader("🌍 Anomalous Countries")
                        st.write(", ".join(analysis["anomaly_countries"]))
                    
                    # Show most anomalous indicators
                    st.subheader("🎯 Most Anomalous Indicators")
                    for indicator, score in list(analysis["most_anomalous_indicators"].items())[:5]:
                        st.write(f"• {indicator}: {score:.3f}")
                    
                    # Visualize anomalies
                    st.subheader("📈 Anomaly Visualization")
                    
                    anomaly_df = results["anomaly_data"]
                    
                    # Create 2D projection
                    dim_reduction = SDGDimensionalityReduction(method="pca", n_components=2)
                    projection_results = dim_reduction.fit_transform(df, selected_indicators)
                    
                    if projection_results["success"]:
                        projection_df = projection_results["transformed_data"]
                        projection_df["is_anomaly"] = anomaly_df["is_anomaly"]
                        
                        fig_anomalies = px.scatter(
                            projection_df,
                            x="PCA_1",
                            y="PCA_2",
                            color="is_anomaly",
                            hover_data=["country"],
                            title="Anomaly Detection Results"
                        )
                        st.plotly_chart(fig_anomalies, use_container_width=True)
                
                else:
                    st.error(f"❌ Anomaly detection failed: {results['error']}")
    
    def render_geospatial(self):
        """Render geospatial analysis"""
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.header("🌍 Geospatial Analysis")
        
        df_filtered = self._filter_data()
        
        # Select indicator for mapping
        available_indicators = [col for col in df_filtered.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicator = st.selectbox("Select Indicator for Mapping", available_indicators)
        
        if "year" in df_filtered.columns:
            years = sorted(df_filtered["year"].unique())
            selected_year = st.selectbox("Select Year", years, index=len(years)-1)
            df_year = df_filtered[df_filtered["year"] == selected_year]
        else:
            df_year = df_filtered
        
        # Create geospatial map
        st.subheader("🗺️ Interactive Map")
        
        try:
            map_obj = self.visualizer.create_geospatial_map(df_year, selected_indicator)
            
            # Display map
            map_html = map_obj._repr_html_()
            st.components.v1.html(map_html, height=600)
            
        except Exception as e:
            st.error(f"Error creating map: {e}")
            st.info("Map visualization requires country coordinates. Using sample coordinates.")
    
    def render_trends(self):
        """Render trend analysis"""
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.header("📈 Trend Analysis")
        
        df_filtered = self._filter_data()
        
        if "year" not in df_filtered.columns:
            st.warning("No year column found for trend analysis")
            return
        
        # Select indicators for trend analysis
        available_indicators = [col for col in df_filtered.columns 
                              if col in SDG_INDICATORS and col not in ["year", "country"]]
        
        selected_indicators = st.multiselect(
            "Select Indicators for Trend Analysis",
            options=available_indicators,
            default=available_indicators[:5] if len(available_indicators) >= 5 else available_indicators
        )
        
        if not selected_indicators:
            st.warning("Please select indicators")
            return
        
        # Calculate trends
        trends = self.data_aggregator.calculate_trend_analysis(df_filtered, selected_indicators)
        
        # Display trend summary
        st.subheader("📊 Trend Summary")
        
        trend_data = []
        for indicator, trend_info in trends.items():
            trend_data.append({
                "Indicator": indicator,
                "Direction": trend_info["trend_direction"],
                "Strength": trend_info["trend_strength"],
                "Slope": trend_info["slope"],
                "R²": trend_info["r_squared"]
            })
        
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True)
        
        # Visualize trends
        st.subheader("📈 Trend Visualization")
        
        # Create animated timeline
        fig_animated = self.visualizer.create_animated_timeline(
            df_filtered,
            selected_indicators[0],
            selected_indicators[1] if len(selected_indicators) > 1 else selected_indicators[0],
            animation_frame="year"
        )
        st.plotly_chart(fig_animated, use_container_width=True)
        
        # Individual trend plots
        for indicator in selected_indicators[:3]:  # Limit to 3 for performance
            st.subheader(f"📊 {indicator} Trend")
            
            # Calculate yearly averages
            yearly_avg = df_filtered.groupby("year")[indicator].mean().reset_index()
            
            fig_trend = px.line(
                yearly_avg,
                x="year",
                y=indicator,
                title=f"{indicator} Over Time",
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def render_export(self):
        """Render export section"""
        st.header("📤 Export & Reports")
        
        if st.session_state.data is None:
            st.info("👆 Please load data from the sidebar to begin analysis")
            return
        
        st.subheader("📊 Data Export")
        
        df_filtered = self._filter_data()
        
        # Export options
        export_formats = st.multiselect(
            "Select Export Formats",
            options=["CSV", "Excel", "JSON", "HTML"],
            default=["CSV"]
        )
        
        if st.button("Export Data"):
            export_data = {
                "filtered_data": df_filtered,
                "summary_stats": df_filtered.describe()
            }
            
            exported_files = self.data_exporter.export_to_multiple_formats(
                export_data,
                "sdg_analysis_export",
                [fmt.lower() for fmt in export_formats]
            )
            
            st.success("✅ Export completed!")
            
            for format_type, filename in exported_files.items():
                st.write(f"📁 {format_type.upper()}: {filename}")
        
        st.subheader("📈 Visualization Export")
        
        # Generate comprehensive report
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Generating comprehensive report..."):
                # Create dashboard
                dashboard_data = {
                    "sdg_scores": df_filtered,
                    "correlations": df_filtered.select_dtypes(include=[np.number]).corr(),
                    "trends": df_filtered,
                    "regional": df_filtered,
                    "development": df_filtered,
                    "network": df_filtered
                }
                
                dashboard_fig = self.visualizer.create_interactive_dashboard(dashboard_data)
                
                # Export dashboard
                html_path = ExportVisualization.export_to_html(dashboard_fig, "sdg_dashboard")
                png_path = ExportVisualization.export_to_png(dashboard_fig, "sdg_dashboard")
                
                st.success("✅ Comprehensive report generated!")
                st.write(f"📊 Dashboard HTML: {html_path}")
                st.write(f"🖼️ Dashboard PNG: {png_path}")
    
    def _filter_data(self):
        """Filter data based on sidebar selections"""
        df = st.session_state.data.copy()
        
        # Filter by countries
        if hasattr(st.session_state, 'selected_countries') and st.session_state.selected_countries:
            df = df[df["country"].isin(st.session_state.selected_countries)]
        
        # Filter by year range
        if hasattr(st.session_state, 'year_range') and "year" in df.columns:
            year_start, year_end = st.session_state.year_range
            df = df[(df["year"] >= year_start) & (df["year"] <= year_end)]
        
        return df

# Main application
if __name__ == "__main__":
    app = SDGSynergyMapperV2()
    app.run()
