import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def compute_ita_features(image_path):
    img_bgr = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    skin_mask = img_lab[:, :, 0] > 30

    l_vals = img_lab[:, :, 0][skin_mask]
    b_vals = img_lab[:, :, 2][skin_mask]

    l_mean = np.mean(l_vals)
    b_mean = np.mean(b_vals)

    ita = np.arctan((l_mean - 50) / (b_mean + 1e-6)) * (180 / np.pi)

    # ITA → Fitzpatrick
    if ita > 55:
        fitz = 'I (Very Light)'
    elif ita > 41:
        fitz = 'II (Light)'
    elif ita > 28:
        fitz = 'III (Intermediate)'
    elif ita > 10:
        fitz = 'IV (Tan)'
    elif ita > -30:
        fitz = 'V (Brown)'
    else:
        fitz = 'VI (Dark)'

    # ITA → Monk
    monk = int(np.round(1 + 9 * (90 - np.clip(ita, -60, 90)) / (90 - (-60))))

    return round(ita, 2), fitz, monk

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            ita, fitz, monk = compute_ita_features(filepath)
            return render_template('index.html', filename=file.filename, ita=ita, fitz=fitz, monk=monk)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

if __name__ == '__main__':
    app.run(debug=True)
