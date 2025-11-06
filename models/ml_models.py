"""
Machine Learning Models for SDG Synergy Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import umap
import hdbscan
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class SDGClusteringModel:
    """Advanced clustering for SDG data analysis"""
    
    def __init__(self, n_clusters: int = 5, algorithm: str = "kmeans"):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Fit clustering model to SDG data"""
        try:
            # Prepare data
            X = df[indicators].fillna(df[indicators].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Choose clustering algorithm
            if self.algorithm == "kmeans":
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            elif self.algorithm == "dbscan":
                self.model = DBSCAN(eps=0.5, min_samples=5)
            elif self.algorithm == "hdbscan":
                self.model = hdbscan.HDBSCAN(min_cluster_size=5)
            
            # Fit model
            self.cluster_labels = self.model.fit_predict(X_scaled)
            self.cluster_centers = self.model.cluster_centers_ if hasattr(self.model, 'cluster_centers_') else None
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, indicators, self.cluster_labels)
            
            return {
                "success": True,
                "cluster_labels": self.cluster_labels,
                "cluster_centers": self.cluster_centers,
                "cluster_analysis": cluster_analysis,
                "n_clusters_found": len(np.unique(self.cluster_labels))
            }
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_clusters(self, df: pd.DataFrame, indicators: List[str], 
                         labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "countries": cluster_data["country"].unique().tolist() if "country" in cluster_data.columns else [],
                "avg_values": cluster_data[indicators].mean().to_dict(),
                "characteristics": self._describe_cluster(cluster_data, indicators)
            }
        
        return analysis
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, indicators: List[str]) -> str:
        """Generate human-readable cluster description"""
        descriptions = []
        
        for indicator in indicators:
            if indicator in cluster_data.columns:
                mean_val = cluster_data[indicator].mean()
                std_val = cluster_data[indicator].std()
                
                if indicator == "gdp_per_capita":
                    if mean_val > 20000:
                        descriptions.append("High-income")
                    elif mean_val > 10000:
                        descriptions.append("Upper-middle income")
                    elif mean_val > 4000:
                        descriptions.append("Lower-middle income")
                    else:
                        descriptions.append("Low-income")
                
                elif indicator == "life_expectancy":
                    if mean_val > 75:
                        descriptions.append("High life expectancy")
                    elif mean_val > 65:
                        descriptions.append("Moderate life expectancy")
                    else:
                        descriptions.append("Low life expectancy")
                
                elif indicator == "renewable_energy_pct":
                    if mean_val > 50:
                        descriptions.append("High renewable energy")
                    elif mean_val > 20:
                        descriptions.append("Moderate renewable energy")
                    else:
                        descriptions.append("Low renewable energy")
        
        return ", ".join(set(descriptions))

class SDGPredictionModel:
    """Predictive models for SDG indicators"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_performance = None
        
    def fit(self, df: pd.DataFrame, target_indicator: str, 
            feature_indicators: List[str]) -> Dict[str, Any]:
        """Fit prediction model"""
        try:
            # Prepare data
            X = df[feature_indicators].fillna(df[feature_indicators].mean())
            y = df[target_indicator].fillna(df[target_indicator].mean())
            
            # Remove rows with missing target
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return {"success": False, "error": "Insufficient data for training"}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Choose model
            if self.model_type == "random_forest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == "linear_regression":
                self.model = LinearRegression()
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.model_performance = {
                "mse": mse,
                "r2": r2,
                "rmse": np.sqrt(mse)
            }
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(feature_indicators, self.model.feature_importances_))
            
            return {
                "success": True,
                "performance": self.model_performance,
                "feature_importance": self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in prediction model: {e}")
            return {"success": False, "error": str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class SDGDimensionalityReduction:
    """Dimensionality reduction for SDG data visualization"""
    
    def __init__(self, method: str = "pca", n_components: int = 2):
        self.method = method
        self.n_components = n_components
        self.model = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Fit and transform data"""
        try:
            # Prepare data
            X = df[indicators].fillna(df[indicators].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Choose method
            if self.method == "pca":
                self.model = PCA(n_components=self.n_components)
            elif self.method == "tsne":
                self.model = TSNE(n_components=self.n_components, random_state=42)
            elif self.method == "umap":
                self.model = umap.UMAP(n_components=self.n_components, random_state=42)
            
            # Fit and transform
            X_reduced = self.model.fit_transform(X_scaled)
            
            # Create result DataFrame
            result_df = df[["country", "year"]].copy() if "country" in df.columns else df.copy()
            for i in range(self.n_components):
                result_df[f"{self.method.upper()}_{i+1}"] = X_reduced[:, i]
            
            # Calculate explained variance for PCA
            explained_variance = None
            if self.method == "pca" and hasattr(self.model, 'explained_variance_ratio_'):
                explained_variance = self.model.explained_variance_ratio_.tolist()
            
            return {
                "success": True,
                "transformed_data": result_df,
                "explained_variance": explained_variance,
                "components": X_reduced
            }
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            return {"success": False, "error": str(e)}

class SDGAnomalyDetector:
    """Anomaly detection for SDG data"""
    
    def __init__(self, method: str = "isolation_forest"):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Detect anomalies in SDG data"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare data
            X = df[indicators].fillna(df[indicators].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit anomaly detection model
            self.model = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = self.model.fit_predict(X_scaled)
            
            # Get anomaly scores
            anomaly_scores = self.model.decision_function(X_scaled)
            
            # Create results
            result_df = df.copy()
            result_df["anomaly_label"] = anomaly_labels
            result_df["anomaly_score"] = anomaly_scores
            result_df["is_anomaly"] = anomaly_labels == -1
            
            # Analyze anomalies
            anomalies = result_df[result_df["is_anomaly"]]
            anomaly_analysis = {
                "total_anomalies": len(anomalies),
                "anomaly_percentage": len(anomalies) / len(df) * 100,
                "anomaly_countries": anomalies["country"].unique().tolist() if "country" in anomalies.columns else [],
                "most_anomalous_indicators": self._identify_anomalous_indicators(anomalies, indicators)
            }
            
            return {
                "success": True,
                "anomaly_data": result_df,
                "anomaly_analysis": anomaly_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"success": False, "error": str(e)}
    
    def _identify_anomalous_indicators(self, anomalies: pd.DataFrame, 
                                      indicators: List[str]) -> Dict[str, float]:
        """Identify which indicators contribute most to anomalies"""
        indicator_anomaly_scores = {}
        
        for indicator in indicators:
            if indicator in anomalies.columns:
                # Calculate how much this indicator deviates from normal
                score = anomalies[indicator].std()
                indicator_anomaly_scores[indicator] = score
        
        # Sort by anomaly score
        return dict(sorted(indicator_anomaly_scores.items(), 
                          key=lambda x: x[1], reverse=True))

class ModelManager:
    """Manage and persist ML models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> str:
        """Save model to disk"""
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump({
            "model": model,
            "metadata": metadata or {},
            "timestamp": pd.Timestamp.now().isoformat()
        }, model_path)
        
        return str(model_path)
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load model from disk"""
        model_path = self.models_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        return joblib.load(model_path)
    
    def list_models(self) -> List[str]:
        """List available models"""
        return [f.stem for f in self.models_dir.glob("*.joblib")]

