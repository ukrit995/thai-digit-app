import os
import io
import base64
import json
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

MODEL_PATH = 'models/current_model.pkl'
MODEL_INFO_PATH = 'models/model_info.json'
DATASET_DIR = 'dataset'
ALLOWED_EXTENSIONS = {'pkl'}

# Thai digit labels for classes 71-75
LABELS = {0: '๗๑', 1: '๗๒', 2: '๗๓', 3: '๗๔', 4: '๗๕'}
LABEL_NAMES = {0: '71', 1: '72', 2: '73', 3: '74', 4: '75'}

# Global model variable
current_model = None
current_model_name = "ยังไม่มีโมเดล"


def load_model():
    global current_model, current_model_name
    if os.path.exists(MODEL_PATH):
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            current_model = pickle.load(f)
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r', encoding='utf-8') as f:
                info = json.load(f)
                current_model_name = info.get('name', 'model.pkl')
        print(f"[INFO] Loaded model: {current_model_name}")
    else:
        print("[INFO] No model found at startup.")


def preprocess_image(image_data_url):
    """Preprocess canvas image for prediction."""
    # Decode base64
    header, encoded = image_data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')

    # White background
    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert('L')  # Grayscale

    # Invert: make digit white on black (like MNIST)
    img = Image.fromarray(255 - np.array(img))

    # Crop to bounding box of digit
    arr = np.array(img)
    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1], cmax + pad)
        img = img.crop((cmin, rmin, cmax, rmax))

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


@app.route('/')
def index():
    return render_template('user.html')


@app.route('/admin')
def admin():
    model_info = {
        'name': current_model_name,
        'loaded': current_model is not None
    }
    return render_template('admin.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    if current_model is None:
        return jsonify({'error': 'ยังไม่มีโมเดล กรุณาอัปโหลดโมเดลก่อน'}), 400

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'ไม่พบข้อมูลภาพ'}), 400

    try:
        arr = preprocess_image(data['image'])
        flat = arr.flatten().reshape(1, -1)

        prediction = int(current_model.predict(flat)[0])
        
        # Get probabilities/confidence
        if hasattr(current_model, 'predict_proba'):
            proba = current_model.predict_proba(flat)[0]
            confidence = float(proba[prediction]) * 100
            all_proba = {LABELS[i]: float(proba[i]) * 100 for i in range(len(LABELS))}
        elif hasattr(current_model, 'decision_function'):
            df = current_model.decision_function(flat)[0]
            exp_df = np.exp(df - np.max(df))
            proba = exp_df / exp_df.sum()
            confidence = float(proba[prediction]) * 100
            all_proba = {LABELS[i]: float(proba[i]) * 100 for i in range(len(LABELS))}
        else:
            confidence = 100.0
            all_proba = {LABELS[i]: (100.0 if i == prediction else 0.0) for i in range(len(LABELS))}

        thai_label = LABELS.get(prediction, '?')
        
        return jsonify({
            'prediction': thai_label,
            'prediction_num': LABEL_NAMES.get(prediction, '?'),
            'confidence': round(confidence, 2),
            'all_probabilities': all_proba
        })

    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500


@app.route('/upload-model', methods=['POST'])
def upload_model():
    global current_model, current_model_name

    if 'model' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์โมเดล'}), 400

    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    if not file.filename.endswith('.pkl'):
        return jsonify({'error': 'รองรับเฉพาะไฟล์ .pkl เท่านั้น'}), 400

    try:
        os.makedirs('models', exist_ok=True)
        file.save(MODEL_PATH)

        import pickle
        with open(MODEL_PATH, 'rb') as f:
            current_model = pickle.load(f)

        current_model_name = secure_filename(file.filename)

        # Save model info
        with open(MODEL_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'name': current_model_name,
                'uploaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': f'อัปโหลดและโหลดโมเดล "{current_model_name}" สำเร็จ',
            'model_name': current_model_name
        })

    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}'}), 500


@app.route('/model-status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': current_model is not None,
        'name': current_model_name
    })


@app.route('/save-sample', methods=['POST'])
def save_sample():
    """Extra endpoint for data collection."""
    data = request.get_json()
    if not data or 'image' not in data or 'label' not in data:
        return jsonify({'error': 'ต้องการ image และ label'}), 400

    label = data['label']
    try:
        arr = preprocess_image(data['image'])
        img = Image.fromarray((arr * 255).astype(np.uint8))

        save_dir = os.path.join(DATASET_DIR, str(label))
        os.makedirs(save_dir, exist_ok=True)

        existing = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
        filename = f"{label}_{existing+1:03d}.png"
        img.save(os.path.join(save_dir, filename))

        return jsonify({'success': True, 'filename': filename, 'count': existing + 1})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset-stats', methods=['GET'])
def dataset_stats():
    stats = {}
    thai_labels = {'71': '๗๑', '72': '๗๒', '73': '๗๓', '74': '๗๔', '75': '๗๕'}
    for label in ['71', '72', '73', '74', '75']:
        d = os.path.join(DATASET_DIR, label)
        if os.path.exists(d):
            count = len([f for f in os.listdir(d) if f.endswith('.png')])
        else:
            count = 0
        stats[thai_labels[label]] = count
    return jsonify(stats)


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    load_model()
    app.run(debug=True, port=5000)
