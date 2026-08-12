# -*- coding: utf-8 -*-
# OVANCO v2 - Ovarian Cancer Detection Platform
# Model: EfficientNetB3 | Dataset: MMOTU | Accuracy: 76.68% | AUC: 0.905
# Visualization: GradCAM

import os
import uuid
import logging
import numpy as np
import cv2
import tensorflow as tf
from flask import (Flask, request, redirect, url_for,
                   render_template, send_from_directory,
                   session, flash)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ── App config ──────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ovanco-v2-dev-key-2026')

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ── Model config ────────────────────────────────────────────────
IMG_SIZE = (300, 300)  # EfficientNetB3 native size

CLASS_LABELS = {
    0: 'Normal',
    1: 'Benign',
    2: 'Malignant'
}

CLASS_DESCRIPTIONS = {
    'Normal':    'No signs of ovarian abnormality detected. Routine follow-up as advised by your clinician.',
    'Benign':    'Findings consistent with a non-cancerous ovarian condition. Clinical monitoring is recommended.',
    'Malignant': 'Findings consistent with a malignant ovarian condition. Immediate specialist referral is required.'
}

RISK_LEVEL = {
    'Normal':    'low',
    'Benign':    'moderate',
    'Malignant': 'high'
}

# ── Demo users ──────────────────────────────────────────────────
USERS = {
    'doctor@ovanco.com':  generate_password_hash('doctor123'),
    'patient@ovanco.com': generate_password_hash('patient123'),
}

# ── Model loading ───────────────────────────────────────────────
classification_model = None

def load_models():
    global classification_model
    # Try v2 model first, fall back to v1
    for model_name in ['ovarian_cancer_model_v2.h5',
                       'ovarian_cancer_model.h5']:
        path = os.path.join('models', model_name)
        if os.path.exists(path):
            classification_model = load_model(path)
            print('[OK] Model loaded: ' + model_name)
            print('[OK] Input shape: ' + str(classification_model.input_shape))
            return
    print('[WARN] No model found in models/ folder')
    print('       Place ovarian_cancer_model_v2.h5 in models/')

# ── Helpers ──────────────────────────────────────────────────────

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_image(img_path):
    """Load and preprocess image for EfficientNet."""
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def compute_gradcam(model, img_array, class_idx):
    """
    Gradient-based saliency map.
    Works with Keras 3.x by watching input tensor directly.
    """
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor, training=False)
        loss        = predictions[:, class_idx]
    grads    = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy().squeeze()
    saliency = ((saliency - saliency.min()) /
                (saliency.max() - saliency.min() + 1e-8))
    return saliency


def generate_heatmap_overlay(img_path, heatmap, save_path, alpha=0.45):
    """
    Overlay GradCAM heatmap on original image.
    Uses JET colormap — blue=low attention, red=high attention.
    """
    img     = load_img(img_path, target_size=IMG_SIZE)
    img_arr = np.array(img)
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    clahe           = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    heatmap_eq      = clahe.apply(heatmap_uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_eq, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(img_bgr, 1-alpha, heatmap_colored, alpha, 0)
    cv2.imwrite(save_path, blended)
    return save_path


def run_analysis(img_path):
    """Full pipeline: classification + GradCAM heatmap."""
    result = {
        'error':             None,
        'predicted_class':   None,
        'confidence':        None,
        'description':       None,
        'risk_level':        None,
        'gradcam_filename':  None,
        'all_probs':         None,
        'low_confidence':    False,
        'models_loaded':     classification_model is not None,
    }

    if not result['models_loaded']:
        result['error'] = ('Model not loaded. '
                           'Place ovarian_cancer_model_v2.h5 in models/ folder.')
        return result

    try:
        # Classification
        img_tensor = preprocess_image(img_path)
        preds      = classification_model.predict(img_tensor, verbose=0)
        pred_idx   = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])

        predicted_class           = CLASS_LABELS.get(pred_idx, 'Unknown')
        result['predicted_class'] = predicted_class
        result['confidence']      = round(confidence * 100, 2)
        result['description']     = CLASS_DESCRIPTIONS.get(predicted_class, '')
        result['risk_level']      = RISK_LEVEL.get(predicted_class, 'unknown')
        result['low_confidence']  = confidence < 0.60
        result['all_probs']       = {
            CLASS_LABELS[i]: round(float(preds[0][i]) * 100, 1)
            for i in range(len(CLASS_LABELS))
        }

        # GradCAM heatmap
        heatmap    = compute_gradcam(classification_model, img_tensor, pred_idx)
        gcam_name  = 'gradcam_' + uuid.uuid4().hex + '.png'
        gcam_path  = os.path.join(app.config['UPLOAD_FOLDER'], gcam_name)
        generate_heatmap_overlay(img_path, heatmap, gcam_path)
        result['gradcam_filename'] = gcam_name

    except Exception as e:
        result['error'] = 'Analysis failed: ' + str(e)

    return result


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('upload'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if email in USERS and check_password_hash(USERS[email], password):
            session['user']      = email
            session['user_name'] = email.split('@')[0].title()
            return redirect(url_for('upload'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        flash('Please log in to upload a scan.', 'info')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file in request.', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Please upload a JPG or PNG image only.', 'error')
            return redirect(request.url)

        # Validate it is actually an image
        try:
            from PIL import Image as PILImage
            file.seek(0)
            PILImage.open(file).verify()
            file.seek(0)
        except Exception:
            flash('Invalid image file. Please upload a valid JPG or PNG.', 'error')
            return redirect(request.url)

        ext   = file.filename.rsplit('.', 1)[1].lower()
        fname = uuid.uuid4().hex + '.' + ext
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)
        return redirect(url_for('results', filename=fname))

    return render_template('upload.html')


@app.route('/results/<filename>')
def results(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    safe  = secure_filename(filename)
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], safe)

    if not os.path.exists(fpath):
        flash('Image not found. Please upload again.', 'error')
        return redirect(url_for('upload'))

    result = run_analysis(fpath)
    return render_template('results.html', filename=safe, result=result)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/about')
def about():
    return render_template('about.html')


# ── Error handlers ────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum 16 MB.', 'error')
    return redirect(url_for('upload'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    print('Starting OVANCO v2...')
    load_models()
    print('Running on http://localhost:5000')
    app.run(debug=True, port=5000, use_reloader=False)
