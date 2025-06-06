import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Kelas dan path data
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
base_path = os.path.join("static", "uploads", "garbage_classification")

def load_images(folder, label):
    data = []
    print(f"🔍 Memuat gambar dari: {folder}")
    if not os.path.exists(folder):
        print(f"⚠️ Folder tidak ditemukan: {folder}")
        return data
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (64, 64)).flatten()
            data.append((img, label))
    return data

dataset = []
for idx, cls in enumerate(classes):
    folder_path = os.path.join(base_path, cls)
    dataset.extend(load_images(folder_path, idx))

if not dataset:
    print("❌ Dataset kosong. Pastikan gambar ada di setiap folder kelas.")
    exit()

X = np.array([item[0] for item in dataset])
y = np.array([item[1] for item in dataset])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Simpan model
os.makedirs("model", exist_ok=True)
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluasi
y_pred = model.predict(X_test)
print("\n✅ Model berhasil disimpan ke model/rf_model.pkl")
print("🎯 Akurasi:", accuracy_score(y_test, y_pred))
print("\n📊 Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Visualisasi
os.makedirs("static/images", exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("static/images/confusion_matrix.png")
plt.close()

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [f'Fitur {i}' for i in indices])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig("static/images/feature_importance.png")
plt.close()
