import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# ===== Load Models =====
def _LoadML():
    try:
        model    = joblib.load(os.path.join(MODELS_DIR, 'ensemble_model.pkl'))
        scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'ml_metadata.pkl'))
        return model, scaler, metadata
    except Exception as e:
        print(f'[WARN] ML model not loaded: {e}')
        return None, None, None

def _LoadNN():
    try:
        import tensorflow as tf
        model    = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'nn_model.keras'))
        scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler_nn.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'nn_metadata.pkl'))
        return model, scaler, metadata
    except Exception as e:
        print(f'[WARN] NN model not loaded: {e}')
        return None, None, None

ensemble_model, ml_scaler, ml_meta = _LoadML()
nn_model,       nn_scaler, nn_meta = _LoadNN()

# ===== Pages =====
@app.route('/')
def Index():
    return render_template('ml-explanation.html')

@app.route('/ml')
def MlExplanation():
    return render_template('ml-explanation.html')

@app.route('/nn')
def NnExplanation():
    return render_template('nn-explanation.html')

@app.route('/demo/ml')
def MlDemo():
    feature_cols = ml_meta['feature_cols'] if ml_meta else []
    return render_template('ml-demo.html', features=feature_cols,
                           model_ready=(ensemble_model is not None))

@app.route('/demo/nn')
def NnDemo():
    feature_cols = nn_meta['feature_cols'] if nn_meta else []
    return render_template('nn-demo.html', features=feature_cols,
                           model_ready=(nn_model is not None))

# ===== Prediction APIs =====
@app.route('/predict/ml', methods=['POST'])
def PredictMl():
    if ensemble_model is None:
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
    if nn_model is None:
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

        probabilities   = nn_model.predict(scaled_features)[0].tolist()
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
