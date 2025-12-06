#!/usr/bin/env python3
"""
Pipeline MLOps final fonctionnel pour Windows
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration simplifiée"""
    
    BASE_DIR = Path(__file__).parent
    
    # Chemins
    RAW_DATA = BASE_DIR / "dataset.csv"  # À la racine
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    MODEL_PATH = MODELS_DIR / "best_model.pkl"
    ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
    CONFIG_PATH = MODELS_DIR / "model_config.json"
    
    # Colonnes
    CATEGORICAL_COLS = ['energy_type', 'energy_subtype']
    TARGET_COL = 'kwh_per_share_per_month'
    
    # Paramètres
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    def __init__(self):
        """Création des répertoires"""
        for directory in [self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOGS_DIR / 'pipeline.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE SIMPLIFIÉ
# ============================================================================
def run_training_pipeline(data_path: Optional[str] = None):
    """Pipeline d'entraînement simplifié et fonctionnel"""
    config = Config()
    
    config.logger.info("="*60)
    config.logger.info("PIPELINE MLOps - ENTRAINEMENT")
    config.logger.info("="*60)
    
    try:
        # 1. Chargement et prétraitement
        config.logger.info("Chargement des donnees...")
        
        # Cherche le fichier
        if data_path:
            path = Path(data_path)
        elif config.RAW_DATA.exists():
            path = config.RAW_DATA
        else:
            raise FileNotFoundError(f"Fichier dataset.csv non trouve. Placez-le dans: {config.RAW_DATA}")
        
        df = pd.read_csv(path)
        config.logger.info(f"Donnees chargees: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Prétraitement
        if 'project_id' in df.columns:
            df = df.drop(columns=['project_id'])
        
        if 'historical_production_kwh' in df.columns:
            import ast
            df['historical_production_kwh'] = df['historical_production_kwh'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            hist_df = pd.DataFrame(
                df['historical_production_kwh'].to_list(),
                columns=[f'hist_month_{i+1}' for i in range(12)]
            )
            df = pd.concat([df.drop(columns=['historical_production_kwh']), hist_df], axis=1)
        
        # 2. Préparation des données
        config.logger.info("Preparation des donnees...")
        
        # Séparation
        X = df.drop(columns=[config.TARGET_COL])
        y = df[config.TARGET_COL]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Encodage pour XGBoost/LightGBM
        X_train_le = X_train.copy()
        X_test_le = X_test.copy()
        encoders = {}
        
        for col in config.CATEGORICAL_COLS:
            le = LabelEncoder()
            X_train_le[f'{col}_encoded'] = le.fit_transform(X_train_le[col])
            X_test_le[f'{col}_encoded'] = le.transform(X_test_le[col])
            encoders[col] = le
        
        # 3. Entraînement des modèles
        config.logger.info("Entrainement des modeles...")
        
        # Configuration des modèles
        models = {
            'catboost': CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                random_seed=config.RANDOM_STATE,
                verbose=0
            ),
            'xgboost': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=config.RANDOM_STATE
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=config.RANDOM_STATE,
                verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            config.logger.info(f"  Entrainement {name}...")
            
            if name == 'catboost':
                # CatBoost avec catégories brutes
                model.fit(X_train, y_train, cat_features=config.CATEGORICAL_COLS, verbose=0)
                y_pred = model.predict(X_test)
            else:
                # XGBoost/LightGBM avec features encodées
                encoded_cols = [f'{col}_encoded' for col in config.CATEGORICAL_COLS]
                other_cols = [c for c in X_train.columns if c not in config.CATEGORICAL_COLS]
                X_train_encoded = X_train_le[encoded_cols + other_cols]
                X_test_encoded = X_test_le[encoded_cols + other_cols]
                
                model.fit(X_train_encoded, y_train)
                y_pred = model.predict(X_test_encoded)
            
            # Calcul des métriques
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            config.logger.info(f"    R²: {r2:.4f}, RMSE: {rmse:.2f}")
        
        # 4. Sélection du meilleur modèle
        best_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
        best_model = results[best_name]['model']
        best_r2 = results[best_name]['r2']
        
        config.logger.info(f"Meilleur modele: {best_name} (R²: {best_r2:.4f})")
        
        # 5. Sauvegarde
        config.logger.info("Sauvegarde du modele...")
        
        with open(config.MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        with open(config.ENCODERS_PATH, 'wb') as f:
            pickle.dump(encoders, f)
        
        config_data = {
            'model_name': best_name,
            'r2_score': best_r2,
            'rmse': results[best_name]['rmse'],
            'mae': results[best_name]['mae'],
            'training_date': datetime.now().isoformat(),
            'features': list(encoders.keys())
        }
        
        with open(config.CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        config.logger.info("Pipeline termine avec succes!")
        config.logger.info(f"Modele sauvegarde: {config.MODEL_PATH}")
        
        return {
            'success': True,
            'best_model': best_name,
            'r2_score': best_r2,
            'results': {k: {'r2': v['r2'], 'rmse': v['rmse']} for k, v in results.items()}
        }
        
    except Exception as e:
        config.logger.error(f"Erreur dans le pipeline: {e}")
        return {'success': False, 'error': str(e)}

def make_prediction(input_data: Dict) -> Dict:
    """Fonction pour faire des prédictions"""
    config = Config()
    
    try:
        # Charger le modèle et les encodeurs
        with open(config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(config.ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        
        # Prétraitement
        df = pd.DataFrame([input_data])
        
        # Renommage des colonnes historiques
        for i in range(1, 13):
            old_name = f'historical_production_kwh_{i}'
            new_name = f'hist_month_{i}'
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Encodage
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    df[col] = 0  # Valeur par défaut
        
        # Colonnes attendues
        expected_cols = config.CATEGORICAL_COLS + [
            'installation_size_kw', 'location_latitude', 'location_longitude',
            'month', 'investment_per_share_eur', 'total_shares', 'panel_age_months'
        ] + [f'hist_month_{i}' for i in range(1, 13)]
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Prédiction
        prediction = model.predict(df)[0]
        
        return {
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat(),
            'units': 'kWh per share per month',
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }

# ============================================================================
# EXÉCUTION
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline MLOps de prediction d\'énergie')
    parser.add_argument('--mode', choices=['train', 'predict', 'test'], 
                       default='train', help='Mode d\'execution')
    parser.add_argument('--data', help='Chemin vers les donnees d\'entrainement')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        result = run_training_pipeline(args.data)
        
        if result['success']:
            print("\nEntrainement reussi!")
            print(f"Meilleur modele: {result['best_model']}")
            print(f"R² sur test: {result['r2_score']:.4f}")
            print("\nResultats complets:")
            for name, metrics in result['results'].items():
                print(f"  {name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}")
        else:
            print(f"\nEchec de l'entrainement: {result.get('error', 'Unknown error')}")
    
    elif args.mode == 'predict':
        # Exemple de prédiction
        example_data = {
            'energy_type': 'Solar',
            'energy_subtype': 'Photovoltaic',
            'installation_size_kw': 500.0,
            'location_latitude': 48.8566,
            'location_longitude': 2.3522,
            'month': 7,
            'investment_per_share_eur': 100.0,
            'total_shares': 1000,
            'panel_age_months': 12
        }
        
        for i in range(1, 13):
            example_data[f'historical_production_kwh_{i}'] = 400 + i*10
        
        result = make_prediction(example_data)
        
        print("\nPREDICTION:")
        print(f"Donnees: Projet solaire 500kW, Paris, Juillet")
        print(f"Resultat: {result['prediction']:.2f} kWh/share/mois")
        print(f"Statut: {result['status']}")
    
    elif args.mode == 'test':
        print("Test du pipeline...")
        
        # Vérifie si un modèle existe déjà
        config = Config()
        if config.MODEL_PATH.exists():
            print("Modele trouve. Test de prediction...")
            test_data = {
                'energy_type': 'Wind',
                'energy_subtype': 'Onshore',
                'installation_size_kw': 1000.0,
                'location_latitude': 45.0,
                'location_longitude': 2.0,
                'month': 6,
                'investment_per_share_eur': 150.0,
                'total_shares': 5000,
                'panel_age_months': 24
            }
            
            for i in range(1, 13):
                test_data[f'historical_production_kwh_{i}'] = 800 + i*20
            
            result = make_prediction(test_data)
            print(f"Test de prediction: {result}")
        else:
            print("Aucun modele trouve. Lancez d'abord --mode train")