from flask import Flask, request, jsonify
from flask_cors import CORS
from energy_pipeline import make_prediction
import os

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "energy-prediction-ml"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = make_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    print(f"✅ API ML prête : http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)