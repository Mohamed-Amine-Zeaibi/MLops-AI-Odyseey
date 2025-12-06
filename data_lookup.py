# data_lookup.py
import pandas as pd
import numpy as np
import random
from pathlib import Path

class SmartDataLookup:
    """Cherche dans dataset OU génère aléatoirement"""
    
    def __init__(self, data_path="dataset.csv"):
        self.df = None
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ Dataset chargé: {len(self.df)} lignes")
        except Exception as e:
            print(f"⚠️  Dataset non trouvé ({e}), utilisation de valeurs aléatoires")
            self.df = pd.DataFrame()
        
        self.cache = {}
        self.rng = np.random.default_rng(42)  # Pour reproductibilité
        
    def _generate_random_values(self, energy_type, energy_subtype):
        """Génère des valeurs aléatoires réalistes"""
        # Tailles typiques par type d'énergie (en kW)
        typical_sizes = {
            'Solar': (100, 5000),
            'Wind': (1000, 10000),
            'Hydro': (500, 5000),
            'Biomass': (500, 3000),
            'Geothermal': (1000, 5000)
        }
        
        # Localisations aléatoires dans des zones réalistes
        regions = {
            'Solar': [(15, 45), (-10, 40)],  # Zones ensoleillées
            'Wind': [(30, 60), (-30, 60)],   # Zones venteuses
            'Hydro': [(0, 60), (-30, 60)],   # Zones avec rivières
            'Biomass': [(0, 60), (-30, 60)],
            'Geothermal': [(0, 60), (-30, 60)]
        }
        
        # Valeurs par défaut basées sur le type
        default_type = energy_type if energy_type in typical_sizes else 'Solar'
        
        # Taille d'installation aléatoire
        min_size, max_size = typical_sizes.get(default_type, (100, 5000))
        installation_size = self.rng.uniform(min_size, max_size)
        
        # Localisation aléatoire
        lat_range, lon_range = regions.get(default_type, [(0, 60), (-30, 60)])
        latitude = self.rng.uniform(lat_range[0], lat_range[1])
        longitude = self.rng.uniform(lon_range[0], lon_range[1])
        
        # Âge des panneaux (en mois)
        panel_age = self.rng.integers(6, 120)  # 6 mois à 10 ans
        
        # Production historique (kWh) - basée sur la taille
        base_production = installation_size * self.rng.uniform(100, 300)
        historical = {}
        for i in range(1, 13):
            # Variation saisonnière + bruit
            seasonal_factor = 1 + 0.3 * np.sin((i-1) * np.pi / 6)  # CORRECTION: Ajout du #
            noise = self.rng.uniform(0.8, 1.2)  # Bruit aléatoire
            historical[f'hist_month_{i}'] = base_production * seasonal_factor * noise
        
        return {
            'installation_size_kw': float(installation_size),
            'location_latitude': float(latitude),
            'location_longitude': float(longitude),
            'panel_age_months': int(panel_age),
            **historical
        }
    
    def get_complete_data(self, energy_type, energy_subtype, month, 
                         investment_per_share_eur, total_shares):
        """Retourne données complètes (lookup OU aléatoire)"""
        cache_key = f"{energy_type}_{energy_subtype}"
        
        if cache_key in self.cache:
            base_data = self.cache[cache_key].copy()
        else:
            if self.df is not None and len(self.df) > 0:
                # Essaie de trouver dans le dataset
                mask = (self.df['energy_type'] == energy_type) & \
                       (self.df['energy_subtype'] == energy_subtype)
                
                filtered = self.df[mask]
                
                if len(filtered) >= 3:  # Au moins 3 échantillons pour être fiable
                    # Moyennes réelles
                    base_data = {
                        'installation_size_kw': float(filtered['installation_size_kw'].mean()),
                        'location_latitude': float(filtered['location_latitude'].mean()),
                        'location_longitude': float(filtered['location_longitude'].mean()),
                        'panel_age_months': float(filtered['panel_age_months'].mean()),
                    }
                    
                    # Historiques
                    for i in range(1, 13):
                        col_name = f'hist_month_{i}'
                        if col_name in filtered.columns:
                            base_data[col_name] = float(filtered[col_name].mean())
                        else:
                            base_data[f'hist_month_{i}'] = 0
                    
                    print(f"📊 Données réelles trouvées: {energy_type}/{energy_subtype} ({len(filtered)} échantillons)")
                else:
                    # Pas assez de données → génération aléatoire
                    print(f"🎲 Génération aléatoire pour: {energy_type}/{energy_subtype}")
                    base_data = self._generate_random_values(energy_type, energy_subtype)
            else:
                # Pas de dataset → génération aléatoire
                print(f"🎲 Génération aléatoire (pas de dataset): {energy_type}/{energy_subtype}")
                base_data = self._generate_random_values(energy_type, energy_subtype)
            
            self.cache[cache_key] = base_data.copy()
        
        # Ajoute les données utilisateur
        base_data.update({
            'energy_type': energy_type,
            'energy_subtype': energy_subtype,
            'month': month,
            'investment_per_share_eur': investment_per_share_eur,
            'total_shares': total_shares
        })
        
        return base_data

# Instance globale
smart_lookup = SmartDataLookup()