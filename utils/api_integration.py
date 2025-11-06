"""
API Integration Module for SDG Synergy Mapper v2
Real-time data fetching from various SDG data sources
"""

import requests
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SDGDataAPIClient:
    """Client for fetching SDG data from various APIs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SDG-Synergy-Mapper-v2/1.0',
            'Accept': 'application/json'
        })
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_world_bank_data(self, indicators: List[str], countries: List[str] = None,
                            start_year: int = 2015, end_year: int = 2023) -> pd.DataFrame:
        """Fetch data from World Bank API"""
        try:
            base_url = self.config.get("world_bank", {}).get("base_url", "https://api.worldbank.org/v2")
            
            # Map SDG indicators to World Bank indicators
            wb_indicators = self._map_sdg_to_world_bank(indicators)
            
            if not wb_indicators:
                logger.warning("No World Bank indicators found for selected SDG indicators")
                return pd.DataFrame()
            
            # Fetch data for each indicator
            all_data = []
            
            for indicator in wb_indicators:
                url = f"{base_url}/country/all/indicator/{indicator}"
                params = {
                    "date": f"{start_year}:{end_year}",
                    "format": "json",
                    "per_page": 20000
                }
                
                if countries:
                    country_codes = self._map_countries_to_wb_codes(countries)
                    if country_codes:
                        params["country"] = ";".join(country_codes)
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if len(data) > 1 and data[1]:
                    df_indicator = pd.DataFrame(data[1])
                    df_indicator['indicator_id'] = indicator
                    all_data.append(df_indicator)
                
                # Rate limiting
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return self._process_world_bank_data(combined_df)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return pd.DataFrame()
    
    def fetch_un_stats_data(self, indicators: List[str], countries: List[str] = None) -> pd.DataFrame:
        """Fetch data from UN Statistics API"""
        try:
            base_url = self.config.get("un_stats", {}).get("base_url", "https://unstats.un.org/sdgapi/v1")
            
            # Map SDG indicators to UN Stats indicators
            un_indicators = self._map_sdg_to_un_stats(indicators)
            
            if not un_indicators:
                logger.warning("No UN Stats indicators found for selected SDG indicators")
                return pd.DataFrame()
            
            all_data = []
            
            for indicator in un_indicators:
                url = f"{base_url}/sdg/data"
                params = {
                    "indicator": indicator,
                    "format": "json"
                }
                
                if countries:
                    country_codes = self._map_countries_to_un_codes(countries)
                    if country_codes:
                        params["country"] = ",".join(country_codes)
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data.get("data"):
                    df_indicator = pd.DataFrame(data["data"])
                    df_indicator['indicator_id'] = indicator
                    all_data.append(df_indicator)
                
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return self._process_un_stats_data(combined_df)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching UN Stats data: {e}")
            return pd.DataFrame()
    
    def fetch_oecd_data(self, indicators: List[str], countries: List[str] = None) -> pd.DataFrame:
        """Fetch data from OECD API"""
        try:
            base_url = self.config.get("oecd", {}).get("base_url", "https://sdmx.oecd.org/public/rest/data")
            
            # Map SDG indicators to OECD indicators
            oecd_indicators = self._map_sdg_to_oecd(indicators)
            
            if not oecd_indicators:
                logger.warning("No OECD indicators found for selected SDG indicators")
                return pd.DataFrame()
            
            all_data = []
            
            for indicator in oecd_indicators:
                url = f"{base_url}/OECD.SDD.SDG,DSD_SDG@DF_SDG,1.0/{indicator}"
                params = {
                    "format": "jsondata"
                }
                
                if countries:
                    country_codes = self._map_countries_to_oecd_codes(countries)
                    if country_codes:
                        params["country"] = "+".join(country_codes)
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data.get("data"):
                    df_indicator = pd.DataFrame(data["data"])
                    df_indicator['indicator_id'] = indicator
                    all_data.append(df_indicator)
                
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return self._process_oecd_data(combined_df)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching OECD data: {e}")
            return pd.DataFrame()
    
    def fetch_combined_data(self, indicators: List[str], countries: List[str] = None,
                          start_year: int = 2015, end_year: int = 2023) -> pd.DataFrame:
        """Fetch data from multiple sources and combine"""
        all_data = []
        
        # Try World Bank
        wb_data = self.fetch_world_bank_data(indicators, countries, start_year, end_year)
        if not wb_data.empty:
            wb_data['source'] = 'World Bank'
            all_data.append(wb_data)
        
        # Try UN Stats
        un_data = self.fetch_un_stats_data(indicators, countries)
        if not un_data.empty:
            un_data['source'] = 'UN Stats'
            all_data.append(un_data)
        
        # Try OECD (if enabled)
        if self.config.get("oecd", {}).get("enabled", False):
            oecd_data = self.fetch_oecd_data(indicators, countries)
            if not oecd_data.empty:
                oecd_data['source'] = 'OECD'
                all_data.append(oecd_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return self._deduplicate_data(combined_df)
        
        return pd.DataFrame()
    
    def _map_sdg_to_world_bank(self, sdg_indicators: List[str]) -> List[str]:
        """Map SDG indicators to World Bank indicator codes"""
        mapping = {
            "gdp_per_capita": "NY.GDP.PCAP.CD",
            "life_expectancy": "SP.DYN.LE00.IN",
            "infant_mortality": "SP.DYN.IMRT.IN",
            "literacy_rate": "SE.ADT.LITR.ZS",
            "renewable_energy_pct": "EG.FEC.RNEW.ZS",
            "electricity_access": "EG.ELC.ACCS.ZS",
            "forest_area_pct": "AG.LND.FRST.ZS",
            "co2_emissions": "EN.ATM.CO2E.PC",
            "urban_population_pct": "SP.URB.TOTL.IN.ZS",
            "internet_penetration": "IT.NET.USER.ZS",
            "employment_rate": "SL.EMP.TOTL.SP.ZS",
            "poverty_rate": "SI.POV.DDAY",
            "gini_index": "SI.POV.GINI"
        }
        
        return [mapping.get(indicator) for indicator in sdg_indicators if mapping.get(indicator)]
    
    def _map_sdg_to_un_stats(self, sdg_indicators: List[str]) -> List[str]:
        """Map SDG indicators to UN Stats indicator codes"""
        mapping = {
            "poverty_rate": "1.1.1",
            "malnutrition_rate": "2.1.1",
            "life_expectancy": "3.1.1",
            "infant_mortality": "3.2.1",
            "literacy_rate": "4.6.1",
            "gender_parity_index": "5.1.1",
            "access_to_clean_water": "6.1.1",
            "renewable_energy_pct": "7.2.1",
            "gdp_per_capita": "8.1.1",
            "internet_penetration": "9.c.1",
            "gini_index": "10.1.1",
            "urban_population_pct": "11.1.1",
            "co2_emissions": "13.2.1",
            "marine_protected_areas": "14.5.1",
            "forest_area_pct": "15.1.1",
            "corruption_index": "16.5.1"
        }
        
        return [mapping.get(indicator) for indicator in sdg_indicators if mapping.get(indicator)]
    
    def _map_sdg_to_oecd(self, sdg_indicators: List[str]) -> List[str]:
        """Map SDG indicators to OECD indicator codes"""
        mapping = {
            "gdp_per_capita": "GDP_PER_CAPITA",
            "life_expectancy": "LIFE_EXPECTANCY",
            "employment_rate": "EMPLOYMENT_RATE",
            "renewable_energy_pct": "RENEWABLE_ENERGY",
            "co2_emissions": "CO2_EMISSIONS",
            "forest_area_pct": "FOREST_AREA"
        }
        
        return [mapping.get(indicator) for indicator in sdg_indicators if mapping.get(indicator)]
    
    def _map_countries_to_wb_codes(self, countries: List[str]) -> List[str]:
        """Map country names to World Bank country codes"""
        mapping = {
            "United States": "US",
            "USA": "US",
            "China": "CN",
            "India": "IN",
            "Brazil": "BR",
            "Germany": "DE",
            "Japan": "JP",
            "Kenya": "KE",
            "Nigeria": "NG",
            "South Africa": "ZA",
            "Australia": "AU",
            "Canada": "CA",
            "France": "FR",
            "United Kingdom": "GB",
            "UK": "GB",
            "Italy": "IT",
            "Spain": "ES"
        }
        
        return [mapping.get(country) for country in countries if mapping.get(country)]
    
    def _map_countries_to_un_codes(self, countries: List[str]) -> List[str]:
        """Map country names to UN country codes"""
        mapping = {
            "United States": "840",
            "USA": "840",
            "China": "156",
            "India": "356",
            "Brazil": "76",
            "Germany": "276",
            "Japan": "392",
            "Kenya": "404",
            "Nigeria": "566",
            "South Africa": "710",
            "Australia": "36",
            "Canada": "124",
            "France": "250",
            "United Kingdom": "826",
            "UK": "826",
            "Italy": "380",
            "Spain": "724"
        }
        
        return [mapping.get(country) for country in countries if mapping.get(country)]
    
    def _map_countries_to_oecd_codes(self, countries: List[str]) -> List[str]:
        """Map country names to OECD country codes"""
        mapping = {
            "United States": "USA",
            "USA": "USA",
            "Germany": "DEU",
            "Japan": "JPN",
            "Australia": "AUS",
            "Canada": "CAN",
            "France": "FRA",
            "United Kingdom": "GBR",
            "UK": "GBR",
            "Italy": "ITA",
            "Spain": "ESP"
        }
        
        return [mapping.get(country) for country in countries if mapping.get(country)]
    
    def _process_world_bank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process World Bank API response"""
        if df.empty:
            return df
        
        # Rename columns
        df = df.rename(columns={
            'country': 'country',
            'date': 'year',
            'value': 'value',
            'indicator': 'indicator_name'
        })
        
        # Pivot the data
        df_pivoted = df.pivot_table(
            index=['country', 'year'],
            columns='indicator_id',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df_pivoted.columns.name = None
        
        return df_pivoted
    
    def _process_un_stats_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process UN Stats API response"""
        if df.empty:
            return df
        
        # Rename columns
        df = df.rename(columns={
            'country': 'country',
            'year': 'year',
            'value': 'value'
        })
        
        # Pivot the data
        df_pivoted = df.pivot_table(
            index=['country', 'year'],
            columns='indicator_id',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df_pivoted.columns.name = None
        
        return df_pivoted
    
    def _process_oecd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process OECD API response"""
        if df.empty:
            return df
        
        # Rename columns
        df = df.rename(columns={
            'country': 'country',
            'year': 'year',
            'value': 'value'
        })
        
        # Pivot the data
        df_pivoted = df.pivot_table(
            index=['country', 'year'],
            columns='indicator_id',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df_pivoted.columns.name = None
        
        return df_pivoted
    
    def _deduplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate data from combined sources"""
        if df.empty:
            return df
        
        # Sort by source priority (World Bank > UN Stats > OECD)
        source_priority = {'World Bank': 1, 'UN Stats': 2, 'OECD': 3}
        df['source_priority'] = df['source'].map(source_priority)
        
        # Keep the first occurrence (highest priority source)
        df_deduplicated = df.sort_values('source_priority').drop_duplicates(
            subset=['country', 'year'], keep='first'
        )
        
        df_deduplicated = df_deduplicated.drop('source_priority', axis=1)
        
        return df_deduplicated

class DataCache:
    """Caching system for API data"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data if it exists and is not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        cache_time = cache_file.stat().st_mtime
        if time.time() - cache_time > self.cache_ttl:
            cache_file.unlink()  # Remove expired cache
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            return None
    
    def cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data to disk"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data.to_dict('records'), f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
    
    def generate_cache_key(self, indicators: List[str], countries: List[str] = None,
                          start_year: int = None, end_year: int = None) -> str:
        """Generate cache key for data request"""
        key_parts = [
            "sdg_data",
            "_".join(sorted(indicators)),
            "_".join(sorted(countries)) if countries else "all_countries",
            f"{start_year}-{end_year}" if start_year and end_year else "all_years"
        ]
        return "_".join(key_parts)

class RealTimeDataUpdater:
    """Real-time data updating system"""
    
    def __init__(self, api_client: SDGDataAPIClient, cache: DataCache):
        self.api_client = api_client
        self.cache = cache
        self.update_interval = 3600  # 1 hour
    
    def update_data(self, indicators: List[str], countries: List[str] = None) -> pd.DataFrame:
        """Update data from APIs with caching"""
        cache_key = self.cache.generate_cache_key(indicators, countries)
        
        # Try to get cached data first
        cached_data = self.cache.get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {cache_key}")
            return cached_data
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {cache_key}")
        fresh_data = self.api_client.fetch_combined_data(indicators, countries)
        
        if not fresh_data.empty:
            # Cache the fresh data
            self.cache.cache_data(cache_key, fresh_data)
        
        return fresh_data
    
    def schedule_updates(self, indicators: List[str], countries: List[str] = None):
        """Schedule periodic data updates"""
        import threading
        import time
        
        def update_worker():
            while True:
                try:
                    self.update_data(indicators, countries)
                    logger.info("Data update completed")
                except Exception as e:
                    logger.error(f"Error in scheduled update: {e}")
                
                time.sleep(self.update_interval)
        
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()
        logger.info("Scheduled data updates started")

