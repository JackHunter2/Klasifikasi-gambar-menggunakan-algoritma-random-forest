from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model & definisi kelas
model = pickle.load(open('model/rf_model.pkl', 'rb'))
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
category_map = {
    'cardboard': 'Organik',
    'glass': 'Anorganik',
    'metal': 'Anorganik',
    'paper': 'Organik',
    'plastic': 'Anorganik',
    'trash': 'B3'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

            pred_idx = model.predict(img)[0]
            class_label = classes[pred_idx]
            label = category_map.get(class_label, "Tidak diketahui")

            return render_template('result.html', image=file.filename, label=label, class_label=class_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
