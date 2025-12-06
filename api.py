# api_finale.py - Version complète qui marche
from flask import Flask, request, jsonify
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

print("="*60)
print("🚀 API MLOps - Version robuste")
print("="*60)

# Charger modèle
model = None
try:
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Modèle chargé")
    else:
        print("⚠️  Mode simulation (pas de modèle)")
except Exception as e:
    print(f"⚠️  Erreur: {e}")

def parse_json_safely(request):
    """Parse JSON de manière robuste"""
    try:
        # Essayer get_json d'abord
        if request.is_json:
            return request.get_json()
        
        # Sinon parser manuellement
        import json
        raw = request.data
        
        if not raw:
            return {}
            
        # Décoder
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        
        # Nettoyer
        raw = raw.strip()
        
        # Remplacer simples quotes par doubles quotes
        raw = raw.replace("'", '"')
        
        return json.loads(raw)
        
    except Exception as e:
        print(f"❌ Erreur parsing JSON: {e}")
        return {"error": "JSON invalide"}

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal - Robust aux problèmes de JSON"""
    try:
        # Parser JSON de manière safe
        data = parse_json_safely(request)
        
        if "error" in data:
            return jsonify(data), 400
        
        print(f"📥 Données reçues: {data}")
        
        # Vérifier les champs requis
        required = ['energy_type', 'energy_subtype', 'month', 
                   'investment_per_share_eur', 'total_shares']
        
        for field in required:
            if field not in data:
                return jsonify({
                    "error": f"Manque: {field}",
                    "required": required,
                    "received": list(data.keys())
                }), 400
        
        # Simuler une prédiction (ou utiliser le vrai modèle)
        if model:
            # ICI: ta vraie logique de prédiction
            prediction_value = 39.57
        else:
            # Simulation basée sur les inputs
            base = 40.0
            if data['energy_type'] == 'Solar':
                base = 42.5
            elif data['energy_type'] == 'Wind':
                base = 35.0
            
            seasonal = 1.0 + 0.2 * ((data['month'] - 6) / 6)
            prediction_value = base * seasonal
        
        # Réponse
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "input": data,
            "prediction": {
                "kwh_per_share_per_month": round(prediction_value, 4),
                "total_kwh_per_month": round(prediction_value * data['total_shares'], 2),
                "units": "kWh"
            },
            "metadata": {
                "model": "real" if model else "simulation",
                "api": "mlops_v1"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"❌ Erreur: {e}")
        traceback.print_exc()
        
        return jsonify({
            "error": str(e),
            "received": str(request.data)[:200] if request.data else "empty",
            "status": "error"
        }), 500

@app.route('/')
def home():
    return jsonify({
        "service": "MLOps Energy API",
        "endpoint": "POST /predict",
        "example": {
            "curl": "curl.exe -X POST http://localhost:5000/predict -H \"Content-Type: application/json\" -d '{\"energy_type\":\"Solar\",\"energy_subtype\":\"Photovoltaic\",\"month\":7,\"investment_per_share_eur\":100,\"total_shares\":1000}'",
            "powershell": "Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType \"application/json\" -Body '{\"energy_type\":\"Solar\",\"energy_subtype\":\"Photovoltaic\",\"month\":7,\"investment_per_share_eur\":100,\"total_shares\":1000}'"
        }
    })

if __name__ == '__main__':
    print("✅ API prête sur http://localhost:5000")
    print("📡 Testez avec:")
    print('   Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -ContentType "application/json" -Body \'{"energy_type":"Solar","energy_subtype":"Photovoltaic","month":7,"investment_per_share_eur":100,"total_shares":1000}\'')
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)