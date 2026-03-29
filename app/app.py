import os
import threading
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

ML_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
ML_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ML_METADATA_PATH = os.path.join(MODELS_DIR, 'ml_metadata.pkl')

NN_MODEL_PATH = os.path.join(MODELS_DIR, 'nn_model.keras')
NN_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_nn.pkl')
NN_METADATA_PATH = os.path.join(MODELS_DIR, 'nn_metadata.pkl')

ml_lock = threading.Lock()
nn_lock = threading.Lock()
ml_warmup_started = False
nn_warmup_started = False

# ===== Load Models =====
def _LoadML():
    try:
        model    = joblib.load(ML_MODEL_PATH)
        scaler   = joblib.load(ML_SCALER_PATH)
        metadata = joblib.load(ML_METADATA_PATH)
        return model, scaler, metadata
    except Exception as e:
        print(f'[WARN] ML model not loaded: {e}')
        return None, None, None

def _LoadNN():
    try:
        import tensorflow as tf
        model    = tf.keras.models.load_model(NN_MODEL_PATH)
        scaler   = joblib.load(NN_SCALER_PATH)
        metadata = joblib.load(NN_METADATA_PATH)
        return model, scaler, metadata
    except Exception as e:
        print(f'[WARN] NN model not loaded: {e}')
        return None, None, None

def _LoadMetadata(metadata_path):
    try:
        return joblib.load(metadata_path)
    except Exception as e:
        print(f'[WARN] Metadata not loaded: {e}')
        return None

def _FilesExist(*file_paths):
    return all(os.path.exists(file_path) for file_path in file_paths)

ensemble_model = None
ml_scaler = None
ml_meta = _LoadMetadata(ML_METADATA_PATH)

nn_model = None
nn_scaler = None
nn_meta = _LoadMetadata(NN_METADATA_PATH)

def _EnsureMLLoaded():
    global ensemble_model, ml_scaler, ml_meta

    if ensemble_model is not None and ml_scaler is not None and ml_meta is not None:
        return True

    with ml_lock:
        if ensemble_model is not None and ml_scaler is not None and ml_meta is not None:
            return True

        ensemble_model, ml_scaler, loaded_meta = _LoadML()
        if loaded_meta is not None:
            ml_meta = loaded_meta

    return ensemble_model is not None and ml_scaler is not None and ml_meta is not None

def _EnsureNNLoaded():
    global nn_model, nn_scaler, nn_meta

    if nn_model is not None and nn_scaler is not None and nn_meta is not None:
        return True

    with nn_lock:
        if nn_model is not None and nn_scaler is not None and nn_meta is not None:
            return True

        nn_model, nn_scaler, loaded_meta = _LoadNN()
        if loaded_meta is not None:
            nn_meta = loaded_meta

    return nn_model is not None and nn_scaler is not None and nn_meta is not None

def _StartMLWarmup():
    global ml_warmup_started

    if ml_warmup_started or not _FilesExist(ML_MODEL_PATH, ML_SCALER_PATH, ML_METADATA_PATH):
        return

    ml_warmup_started = True
    threading.Thread(target=_EnsureMLLoaded, daemon=True).start()

def _StartNNWarmup():
    global nn_warmup_started

    if nn_warmup_started or not _FilesExist(NN_MODEL_PATH, NN_SCALER_PATH, NN_METADATA_PATH):
        return

    nn_warmup_started = True
    threading.Thread(target=_EnsureNNLoaded, daemon=True).start()

# ===== Pages =====
@app.route('/')
def Index():
    _StartMLWarmup()
    return render_template('ml-explanation.html')

@app.route('/favicon.ico')
def Favicon():
    return ('', 204)

@app.route('/ml')
def MlExplanation():
    _StartMLWarmup()
    return render_template('ml-explanation.html')

@app.route('/nn')
def NnExplanation():
    _StartNNWarmup()
    return render_template('nn-explanation.html')

@app.route('/demo/ml')
def MlDemo():
    _StartMLWarmup()
    feature_cols = ml_meta['feature_cols'] if ml_meta else []
    return render_template('ml-demo.html', features=feature_cols,
                           model_ready=_FilesExist(ML_MODEL_PATH, ML_SCALER_PATH, ML_METADATA_PATH))

@app.route('/demo/nn')
def NnDemo():
    _StartNNWarmup()
    feature_cols = nn_meta['feature_cols'] if nn_meta else []
    return render_template('nn-demo.html', features=feature_cols,
                           model_ready=_FilesExist(NN_MODEL_PATH, NN_SCALER_PATH, NN_METADATA_PATH))

# ===== Prediction APIs =====
@app.route('/predict/ml', methods=['POST'])
def PredictMl():
    if not _EnsureMLLoaded():
        return jsonify({'error': 'โมเดลยังไม่พร้อม กรุณารัน notebook ก่อน'}), 503
    try:
        request_body = request.get_json(force=True)
        if not isinstance(request_body, dict):
            return jsonify({'error': 'Invalid JSON body'}), 400

        feature_cols = ml_meta['feature_cols']
        classes      = ml_meta['classes']

        # Build feature vector — sanitize each value
        feature_values = []
        for col in feature_cols:
            raw_value = request_body.get(col, 0)
            try:
                feature_values.append(float(raw_value))
            except (TypeError, ValueError):
                feature_values.append(0.0)

        feature_matrix = np.array(feature_values, dtype=float).reshape(1, -1)
        scaled_features = ml_scaler.transform(feature_matrix)

        raw_prediction = ensemble_model.predict(scaled_features)[0]
        probabilities  = ensemble_model.predict_proba(scaled_features)[0].tolist()
        if isinstance(raw_prediction, (int, np.integer)):
            predicted_index = int(raw_prediction)
            prediction      = classes[predicted_index]
        else:
            prediction      = str(raw_prediction)
            predicted_index = classes.index(prediction) if prediction in classes else 0

        # Feature importance from RF base estimator
        feature_importance = {}
        rf_estimator = ensemble_model.named_estimators_.get('rf')
        if rf_estimator is not None and hasattr(rf_estimator, 'feature_importances_'):
            feature_importance = dict(zip(
                feature_cols,
                [round(float(score), 4) for score in rf_estimator.feature_importances_]
            ))

        return jsonify({
            'prediction':         prediction,
            'probabilities':      dict(zip(classes, [round(prob, 4) for prob in probabilities])),
            'feature_importance': feature_importance
        })
    except Exception as e:
        app.logger.exception('ML prediction error')
        return jsonify({'error': 'เกิดข้อผิดพลาดในการทำนาย กรุณาลองใหม่อีกครั้ง'}), 400


@app.route('/predict/nn', methods=['POST'])
def PredictNn():
    if not _EnsureNNLoaded():
        return jsonify({'error': 'โมเดลยังไม่พร้อม กรุณารัน notebook ก่อน'}), 503
    try:
        request_body = request.get_json(force=True)
        if not isinstance(request_body, dict):
            return jsonify({'error': 'Invalid JSON body'}), 400

        feature_cols = nn_meta['feature_cols']
        classes      = nn_meta['classes']

        feature_values = []
        for col in feature_cols:
            raw_value = request_body.get(col, 0)
            try:
                feature_values.append(float(raw_value))
            except (TypeError, ValueError):
                feature_values.append(0.0)

        feature_matrix  = np.array(feature_values, dtype=float).reshape(1, -1)
        scaled_features = nn_scaler.transform(feature_matrix)

        probabilities   = nn_model.predict(scaled_features, verbose=0)[0].tolist()
        predicted_index = int(np.argmax(probabilities))

        result_label_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

        return jsonify({
            'prediction':       classes[predicted_index],
            'prediction_label': result_label_map.get(classes[predicted_index], classes[predicted_index]),
            'confidence':       round(float(max(probabilities)), 4),
            'probabilities':    {
                result_label_map.get(class_key, class_key): round(float(prob), 4)
                for class_key, prob in zip(classes, probabilities)
            }
        })
    except Exception as e:
        app.logger.exception('NN prediction error')
        return jsonify({'error': 'เกิดข้อผิดพลาดในการทำนาย กรุณาลองใหม่อีกครั้ง'}), 400


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', '5000'))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
