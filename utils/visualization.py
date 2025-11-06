"""
Advanced Visualization Utilities for SDG Synergy Mapper v2
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import folium
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import tempfile
import os

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """Advanced visualization capabilities for SDG analysis"""
    
    def __init__(self, theme: str = "plotly"):
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
        
    def create_interactive_dashboard(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create comprehensive interactive dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("SDG Progress Overview", "Correlation Heatmap", 
                          "Trend Analysis", "Regional Comparison",
                          "Development Level Distribution", "SDG Synergy Network"),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "box"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Add various visualizations
        self._add_progress_overview(fig, data.get("sdg_scores", pd.DataFrame()))
        self._add_correlation_heatmap(fig, data.get("correlations", pd.DataFrame()))
        self._add_trend_analysis(fig, data.get("trends", pd.DataFrame()))
        self._add_regional_comparison(fig, data.get("regional", pd.DataFrame()))
        self._add_development_distribution(fig, data.get("development", pd.DataFrame()))
        self._add_synergy_network(fig, data.get("network", pd.DataFrame()))
        
        fig.update_layout(
            height=1200,
            title_text="SDG Synergy Analysis Dashboard",
            showlegend=False,
            template=self.theme
        )
        
        return fig
    
    def create_geospatial_map(self, df: pd.DataFrame, indicator: str, 
                             year: int = None) -> folium.Map:
        """Create geospatial visualization of SDG indicators"""
        # Country coordinates mapping
        country_coords = {
            "USA": [39.8283, -98.5795],
            "China": [35.8617, 104.1954],
            "India": [20.5937, 78.9629],
            "Brazil": [-14.2350, -51.9253],
            "Germany": [51.1657, 10.4515],
            "Japan": [36.2048, 138.2529],
            "Kenya": [-0.0236, 37.9062],
            "Nigeria": [9.0820, 8.6753],
            "South Africa": [-30.5595, 22.9375],
            "Australia": [-25.2744, 133.7751],
            "Canada": [56.1304, -106.3468],
            "France": [46.2276, 2.2137],
            "UK": [55.3781, -3.4360],
            "Italy": [41.8719, 12.5674],
            "Spain": [40.4637, -3.7492]
        }
        
        # Create base map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Filter data by year if specified
        if year and "year" in df.columns:
            df_filtered = df[df["year"] == year]
        else:
            df_filtered = df
        
        # Add markers for each country
        for _, row in df_filtered.iterrows():
            country = row["country"]
            if country in country_coords and indicator in row:
                value = row[indicator]
                if pd.notna(value):
                    # Color based on value
                    color = self._get_color_for_value(value, indicator)
                    
                    folium.CircleMarker(
                        location=country_coords[country],
                        radius=10,
                        popup=f"{country}: {indicator} = {value:.2f}",
                        color=color,
                        fill=True,
                        fillOpacity=0.7
                    ).add_to(m)
        
        return m
    
    def create_sankey_diagram(self, df: pd.DataFrame, 
                            source_col: str, target_col: str, 
                            value_col: str) -> go.Figure:
        """Create Sankey diagram for SDG flow analysis"""
        # Prepare data for Sankey
        sankey_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
        
        # Create unique labels
        all_labels = list(set(sankey_data[source_col].unique()) | 
                         set(sankey_data[target_col].unique()))
        label_dict = {label: i for i, label in enumerate(all_labels)}
        
        # Map to indices
        sankey_data["source_idx"] = sankey_data[source_col].map(label_dict)
        sankey_data["target_idx"] = sankey_data[target_col].map(label_dict)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color="blue"
            ),
            link=dict(
                source=sankey_data["source_idx"],
                target=sankey_data["target_idx"],
                value=sankey_data[value_col]
            )
        )])
        
        fig.update_layout(
            title_text="SDG Indicator Flow Analysis",
            font_size=10,
            template=self.theme
        )
        
        return fig
    
    def create_radar_chart(self, df: pd.DataFrame, indicators: List[str], 
                          countries: List[str] = None) -> go.Figure:
        """Create radar chart for multi-dimensional SDG comparison"""
        if countries is None:
            countries = df["country"].unique()[:5]  # Limit to 5 countries
        
        fig = go.Figure()
        
        for country in countries:
            country_data = df[df["country"] == country]
            if len(country_data) > 0:
                values = []
                for indicator in indicators:
                    if indicator in country_data.columns:
                        values.append(country_data[indicator].mean())
                    else:
                        values.append(0)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=indicators,
                    fill='toself',
                    name=country
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="SDG Performance Radar Chart",
            template=self.theme
        )
        
        return fig
    
    def create_waterfall_chart(self, df: pd.DataFrame, 
                              indicator: str, year_col: str = "year") -> go.Figure:
        """Create waterfall chart showing SDG progress over time"""
        # Sort by year
        df_sorted = df.sort_values(year_col)
        
        # Calculate changes
        values = df_sorted[indicator].values
        changes = np.diff(values)
        
        # Create waterfall data
        x_labels = [f"{int(year)}" for year in df_sorted[year_col].values[1:]]
        
        fig = go.Figure(go.Waterfall(
            name="SDG Progress",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(changes) - 1),
            x=x_labels,
            y=[values[0]] + list(changes),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"SDG {indicator} Progress Over Time",
            showlegend=False,
            template=self.theme
        )
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame, 
                                  indicators: List[str], 
                                  color_by: str = None) -> go.Figure:
        """Create parallel coordinates plot for multi-dimensional analysis"""
        # Prepare data
        plot_data = df[indicators + ([color_by] if color_by else [])].copy()
        
        # Handle missing values
        for col in indicators:
            plot_data[col] = plot_data[col].fillna(plot_data[col].mean())
        
        fig = px.parallel_coordinates(
            plot_data,
            dimensions=indicators,
            color=color_by if color_by else None,
            title="SDG Indicators Parallel Coordinates",
            template=self.theme
        )
        
        return fig
    
    def create_3d_scatter(self, df: pd.DataFrame, 
                         x_col: str, y_col: str, z_col: str,
                         color_col: str = None, size_col: str = None) -> go.Figure:
        """Create 3D scatter plot for SDG analysis"""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            size=size_col,
            hover_data=["country", "year"] if "country" in df.columns else ["year"],
            title=f"3D Analysis: {x_col} vs {y_col} vs {z_col}",
            template=self.theme
        )
        
        return fig
    
    def create_animated_timeline(self, df: pd.DataFrame, 
                               x_col: str, y_col: str,
                               animation_frame: str = "year",
                               color_col: str = None) -> go.Figure:
        """Create animated timeline visualization"""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            animation_frame=animation_frame,
            color=color_col,
            hover_data=["country"] if "country" in df.columns else [],
            title=f"Animated Timeline: {x_col} vs {y_col}",
            template=self.theme
        )
        
        # Update animation settings
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}}],
                     "label": "Play",
                     "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                     "label": "Pause",
                     "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        return fig
    
    def create_network_visualization(self, correlation_matrix: pd.DataFrame, 
                                   threshold: float = 0.5) -> Network:
        """Create advanced network visualization"""
        G = nx.Graph()
        
        # Add nodes
        for col in correlation_matrix.columns:
            G.add_node(col, label=col, size=20)
        
        # Add edges based on correlation
        for i in correlation_matrix.columns:
            for j in correlation_matrix.columns:
                if i != j:
                    corr_value = correlation_matrix.loc[i, j]
                    if abs(corr_value) >= threshold:
                        G.add_edge(
                            i, j,
                            weight=abs(corr_value),
                            color="green" if corr_value > 0 else "red",
                            title=f"Correlation: {corr_value:.3f}"
                        )
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -20000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04
            },
            "maxVelocity": 50,
            "minVelocity": 0.1
          },
          "nodes": {
            "font": {"size": 16},
            "borderWidth": 2,
            "shadow": true
          },
          "edges": {
            "width": 2,
            "shadow": true
          }
        }
        """)
        
        return net
    
    def _get_color_for_value(self, value: float, indicator: str) -> str:
        """Get color based on indicator value"""
        # Define color scales for different indicators
        if "gdp_per_capita" in indicator.lower():
            if value > 20000:
                return "green"
            elif value > 10000:
                return "yellow"
            else:
                return "red"
        elif "life_expectancy" in indicator.lower():
            if value > 75:
                return "green"
            elif value > 65:
                return "yellow"
            else:
                return "red"
        elif "poverty" in indicator.lower():
            if value < 10:
                return "green"
            elif value < 25:
                return "yellow"
            else:
                return "red"
        else:
            # Default color scale
            if value > 75:
                return "green"
            elif value > 50:
                return "yellow"
            else:
                return "red"
    
    def _add_progress_overview(self, fig: go.Figure, data: pd.DataFrame):
        """Add SDG progress overview to dashboard"""
        if not data.empty and "sdg_score" in data.columns:
            fig.add_trace(
                go.Bar(x=data["sdg"], y=data["sdg_score"], name="SDG Score"),
                row=1, col=1
            )
    
    def _add_correlation_heatmap(self, fig: go.Figure, data: pd.DataFrame):
        """Add correlation heatmap to dashboard"""
        if not data.empty:
            fig.add_trace(
                go.Heatmap(z=data.values, x=data.columns, y=data.index),
                row=1, col=2
            )
    
    def _add_trend_analysis(self, fig: go.Figure, data: pd.DataFrame):
        """Add trend analysis to dashboard"""
        if not data.empty and "year" in data.columns:
            for col in data.columns:
                if col != "year":
                    fig.add_trace(
                        go.Scatter(x=data["year"], y=data[col], name=col),
                        row=2, col=1
                    )
    
    def _add_regional_comparison(self, fig: go.Figure, data: pd.DataFrame):
        """Add regional comparison to dashboard"""
        if not data.empty and "region" in data.columns:
            fig.add_trace(
                go.Box(y=data["value"], x=data["region"], name="Regional Comparison"),
                row=2, col=2
            )
    
    def _add_development_distribution(self, fig: go.Figure, data: pd.DataFrame):
        """Add development level distribution to dashboard"""
        if not data.empty and "development_level" in data.columns:
            counts = data["development_level"].value_counts()
            fig.add_trace(
                go.Pie(labels=counts.index, values=counts.values),
                row=3, col=1
            )
    
    def _add_synergy_network(self, fig: go.Figure, data: pd.DataFrame):
        """Add synergy network to dashboard"""
        if not data.empty:
            fig.add_trace(
                go.Scatter(x=data["x"], y=data["y"], mode="markers+lines"),
                row=3, col=2
            )

class ExportVisualization:
    """Export visualization utilities"""
    
    @staticmethod
    def export_to_html(fig: go.Figure, filename: str) -> str:
        """Export Plotly figure to HTML"""
        html_path = f"{filename}.html"
        fig.write_html(html_path)
        return html_path
    
    @staticmethod
    def export_to_png(fig: go.Figure, filename: str, width: int = 1200, 
                     height: int = 800) -> str:
        """Export Plotly figure to PNG"""
        png_path = f"{filename}.png"
        fig.write_image(png_path, width=width, height=height)
        return png_path
    
    @staticmethod
    def export_network_to_html(network: Network, filename: str) -> str:
        """Export PyVis network to HTML"""
        html_path = f"{filename}.html"
        network.save_graph(html_path)
        return html_path
    
    @staticmethod
    def export_map_to_html(map_obj: folium.Map, filename: str) -> str:
        """Export Folium map to HTML"""
        html_path = f"{filename}.html"
        map_obj.save(html_path)
        return html_path

