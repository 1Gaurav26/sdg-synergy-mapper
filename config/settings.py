# SDG Synergy Mapper v2 Configuration

# Application Settings
APP_NAME = "SDG Synergy Mapper v2"
APP_VERSION = "2.0.0"
DEBUG = True

# Data Sources
DATA_SOURCES = {
    "world_bank": {
        "base_url": "https://api.worldbank.org/v2",
        "enabled": True
    },
    "un_stats": {
        "base_url": "https://unstats.un.org/sdgapi/v1",
        "enabled": True
    },
    "oecd": {
        "base_url": "https://sdmx.oecd.org/public/rest/data",
        "enabled": False
    }
}

# SDG Indicator Mapping (Extended)
SDG_INDICATORS = {
    # SDG 1: No Poverty
    "poverty_rate": {"sdg": 1, "category": "Social", "priority": "high"},
    "income_inequality": {"sdg": 1, "category": "Social", "priority": "high"},
    "social_protection": {"sdg": 1, "category": "Social", "priority": "medium"},
    
    # SDG 2: Zero Hunger
    "malnutrition_rate": {"sdg": 2, "category": "Social", "priority": "high"},
    "food_security_index": {"sdg": 2, "category": "Social", "priority": "high"},
    "agricultural_productivity": {"sdg": 2, "category": "Economy", "priority": "medium"},
    
    # SDG 3: Good Health & Well-being
    "life_expectancy": {"sdg": 3, "category": "Social", "priority": "high"},
    "infant_mortality": {"sdg": 3, "category": "Social", "priority": "high"},
    "healthcare_access": {"sdg": 3, "category": "Social", "priority": "high"},
    "mental_health": {"sdg": 3, "category": "Social", "priority": "medium"},
    
    # SDG 4: Quality Education
    "education_index": {"sdg": 4, "category": "Social", "priority": "high"},
    "literacy_rate": {"sdg": 4, "category": "Social", "priority": "high"},
    "digital_skills": {"sdg": 4, "category": "Economy", "priority": "medium"},
    
    # SDG 5: Gender Equality
    "female_labor_participation": {"sdg": 5, "category": "Social", "priority": "high"},
    "gender_parity_index": {"sdg": 5, "category": "Social", "priority": "high"},
    "gender_wage_gap": {"sdg": 5, "category": "Social", "priority": "medium"},
    
    # SDG 6: Clean Water & Sanitation
    "access_to_clean_water": {"sdg": 6, "category": "Environment", "priority": "high"},
    "sanitation_coverage": {"sdg": 6, "category": "Environment", "priority": "high"},
    "water_quality_index": {"sdg": 6, "category": "Environment", "priority": "medium"},
    
    # SDG 7: Affordable & Clean Energy
    "renewable_energy_pct": {"sdg": 7, "category": "Environment", "priority": "high"},
    "electricity_access": {"sdg": 7, "category": "Economy", "priority": "high"},
    "energy_efficiency": {"sdg": 7, "category": "Environment", "priority": "medium"},
    
    # SDG 8: Decent Work & Economic Growth
    "gdp_per_capita": {"sdg": 8, "category": "Economy", "priority": "high"},
    "employment_rate": {"sdg": 8, "category": "Economy", "priority": "high"},
    "informal_economy": {"sdg": 8, "category": "Economy", "priority": "medium"},
    
    # SDG 9: Industry, Innovation & Infrastructure
    "internet_penetration": {"sdg": 9, "category": "Economy", "priority": "high"},
    "research_spending": {"sdg": 9, "category": "Economy", "priority": "high"},
    "infrastructure_quality": {"sdg": 9, "category": "Economy", "priority": "medium"},
    
    # SDG 10: Reduced Inequalities
    "gini_index": {"sdg": 10, "category": "Social", "priority": "high"},
    "income_quintile_ratio": {"sdg": 10, "category": "Social", "priority": "medium"},
    
    # SDG 11: Sustainable Cities
    "urban_population_pct": {"sdg": 11, "category": "Environment", "priority": "high"},
    "air_quality_index": {"sdg": 11, "category": "Environment", "priority": "high"},
    "public_transport": {"sdg": 11, "category": "Environment", "priority": "medium"},
    
    # SDG 12: Responsible Consumption
    "waste_recycling_rate": {"sdg": 12, "category": "Environment", "priority": "high"},
    "material_footprint": {"sdg": 12, "category": "Environment", "priority": "high"},
    "circular_economy": {"sdg": 12, "category": "Environment", "priority": "medium"},
    
    # SDG 13: Climate Action
    "co2_emissions": {"sdg": 13, "category": "Environment", "priority": "high"},
    "climate_risk_index": {"sdg": 13, "category": "Environment", "priority": "high"},
    "carbon_intensity": {"sdg": 13, "category": "Environment", "priority": "medium"},
    
    # SDG 14: Life Below Water
    "marine_protected_areas": {"sdg": 14, "category": "Environment", "priority": "high"},
    "fish_stock_health": {"sdg": 14, "category": "Environment", "priority": "high"},
    "ocean_acidification": {"sdg": 14, "category": "Environment", "priority": "medium"},
    
    # SDG 15: Life on Land
    "forest_area_pct": {"sdg": 15, "category": "Environment", "priority": "high"},
    "biodiversity_index": {"sdg": 15, "category": "Environment", "priority": "high"},
    "land_degradation": {"sdg": 15, "category": "Environment", "priority": "medium"},
    
    # SDG 16: Peace, Justice & Institutions
    "corruption_index": {"sdg": 16, "category": "Governance", "priority": "high"},
    "rule_of_law_index": {"sdg": 16, "category": "Governance", "priority": "high"},
    "political_stability": {"sdg": 16, "category": "Governance", "priority": "medium"},
    
    # SDG 17: Partnerships for the Goals
    "foreign_aid_received": {"sdg": 17, "category": "Governance", "priority": "high"},
    "international_partnerships": {"sdg": 17, "category": "Governance", "priority": "high"},
    "technology_transfer": {"sdg": 17, "category": "Governance", "priority": "medium"}
}

# Analysis Settings
CORRELATION_THRESHOLDS = {
    "weak": 0.3,
    "moderate": 0.5,
    "strong": 0.7,
    "very_strong": 0.9
}

ML_MODELS = {
    "clustering": ["kmeans", "hdbscan", "gmm"],
    "prediction": ["linear_regression", "random_forest", "xgboost"],
    "dimensionality": ["pca", "umap", "tsne"]
}

# Visualization Settings
CHART_THEMES = {
    "default": "plotly",
    "dark": "plotly_dark",
    "white": "plotly_white",
    "minimal": "simple_white"
}

# Export Settings
EXPORT_FORMATS = ["pdf", "html", "excel", "csv", "json", "png", "svg"]
MAX_EXPORT_SIZE = 100  # MB

# Cache Settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000  # MB

