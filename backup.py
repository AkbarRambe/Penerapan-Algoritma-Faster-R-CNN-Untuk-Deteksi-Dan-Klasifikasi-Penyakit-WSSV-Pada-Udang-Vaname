import os
import io
import torch
import torchvision
import uuid
import zipfile
import json
from flask import Flask, request, render_template, url_for, redirect, send_file, session, jsonify
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from werkzeug.utils import secure_filename

# --- 1. Inisialisasi Aplikasi Flask & Konfigurasi ---
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this'  # Ganti dengan kunci yang aman
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'graphs'), exist_ok=True)

# Dictionary untuk menyimpan hasil sementara
temp_results = {}
# Dictionary untuk menyimpan statistik deteksi
detection_stats = {}

# --- 2. Konfigurasi Model ---
DISPLAY_CLASSES = ['wssv', 'healthy']
MODEL_PATH = os.path.join("model", "shrimp_detector_model_final.pth")
CONFIDENCE_THRESHOLD = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3. Fungsi untuk Memuat Model ---
def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 4. Fungsi untuk Menghitung Confusion Matrix ---
def calculate_confusion_matrix(results):
    """Menghitung confusion matrix berdasarkan hasil deteksi"""
    confusion_matrix = {
        'true_positive_wssv': 0,    # Prediksi WSSV, sebenarnya WSSV
        'false_positive_wssv': 0,   # Prediksi WSSV, sebenarnya Healthy
        'true_negative_wssv': 0,    # Prediksi Healthy, sebenarnya Healthy
        'false_negative_wssv': 0,   # Prediksi Healthy, sebenarnya WSSV
        'true_positive_healthy': 0, # Prediksi Healthy, sebenarnya Healthy
        'false_positive_healthy': 0,# Prediksi Healthy, sebenarnya WSSV
        'true_negative_healthy': 0, # Prediksi WSSV, sebenarnya WSSV
        'false_negative_healthy': 0 # Prediksi WSSV, sebenarnya Healthy
    }
    
    # Untuk contoh ini, kita akan membuat asumsi klasifikasi gambar berdasarkan deteksi dominan
    # Dalam praktik nyata, Anda perlu ground truth labels
    
    for result in results:
        wssv_detections = sum(1 for d in result['detections'] if d['label'] == 'wssv')
        healthy_detections = sum(1 for d in result['detections'] if d['label'] == 'healthy')
        
        # Klasifikasi gambar berdasarkan deteksi yang dominan
        if wssv_detections > healthy_detections:
            predicted_class = 'wssv'
        elif healthy_detections > wssv_detections:
            predicted_class = 'healthy'
        else:
            # Jika sama atau tidak ada deteksi, gunakan kepercayaan tertinggi
            if result['detections']:
                max_confidence_detection = max(result['detections'], key=lambda x: x['confidence'])
                predicted_class = max_confidence_detection['label']
            else:
                predicted_class = 'healthy'  # Default jika tidak ada deteksi
        
        # Untuk demo, kita akan menggunakan heuristic sederhana untuk "ground truth"
        # Dalam implementasi nyata, ini harus berasal dari label yang sudah diketahui
        # Asumsi: jika ada deteksi WSSV dengan confidence > 0.7, maka ground truth = WSSV
        high_confidence_wssv = any(d['label'] == 'wssv' and d['confidence'] > 0.7 for d in result['detections'])
        ground_truth = 'wssv' if high_confidence_wssv else 'healthy'
        
        # Update confusion matrix
        if predicted_class == 'wssv' and ground_truth == 'wssv':
            confusion_matrix['true_positive_wssv'] += 1
        elif predicted_class == 'wssv' and ground_truth == 'healthy':
            confusion_matrix['false_positive_wssv'] += 1
        elif predicted_class == 'healthy' and ground_truth == 'healthy':
            confusion_matrix['true_negative_wssv'] += 1
        elif predicted_class == 'healthy' and ground_truth == 'wssv':
            confusion_matrix['false_negative_wssv'] += 1
    
    return confusion_matrix

# --- 5. Fungsi untuk Menghitung Statistik Deteksi ---
def calculate_detection_stats(results):
    """Menghitung statistik dari hasil deteksi"""
    total_images = len(results)
    wssv_detected = 0
    healthy_detected = 0
    total_detections = 0
    confidence_scores = []
    
    for result in results:
        has_wssv = False
        has_healthy = False
        
        for detection in result['detections']:
            total_detections += 1
            confidence_scores.append(detection['confidence'])
            
            if detection['label'] == 'wssv':
                has_wssv = True
            elif detection['label'] == 'healthy':
                has_healthy = True
        
        if has_wssv:
            wssv_detected += 1
        if has_healthy:
            healthy_detected += 1
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # Tambahkan confusion matrix ke statistik
    confusion_matrix = calculate_confusion_matrix(results)
    
    # Hitung metrik evaluasi
    tp_wssv = confusion_matrix['true_positive_wssv']
    fp_wssv = confusion_matrix['false_positive_wssv']
    tn_wssv = confusion_matrix['true_negative_wssv']
    fn_wssv = confusion_matrix['false_negative_wssv']
    
    precision_wssv = tp_wssv / (tp_wssv + fp_wssv) if (tp_wssv + fp_wssv) > 0 else 0
    recall_wssv = tp_wssv / (tp_wssv + fn_wssv) if (tp_wssv + fn_wssv) > 0 else 0
    f1_wssv = 2 * (precision_wssv * recall_wssv) / (precision_wssv + recall_wssv) if (precision_wssv + recall_wssv) > 0 else 0
    accuracy = (tp_wssv + tn_wssv) / total_images if total_images > 0 else 0
    
    return {
        'total_images': total_images,
        'wssv_detected': wssv_detected,
        'healthy_detected': healthy_detected,
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'wssv_percentage': (wssv_detected / total_images * 100) if total_images > 0 else 0,
        'healthy_percentage': (healthy_detected / total_images * 100) if total_images > 0 else 0,
        'confusion_matrix': confusion_matrix,
        'precision_wssv': precision_wssv,
        'recall_wssv': recall_wssv,
        'f1_score_wssv': f1_wssv,
        'accuracy': accuracy
    }

# --- 6. Memuat Model (Hanya Sekali Saat Server Dinyalakan) ---
print("Memuat model PyTorch, harap tunggu...")
num_classes_internal = len(DISPLAY_CLASSES) + 1
model = get_detection_model(num_classes_internal)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model berhasil dimuat dan siap menerima request.")
except FileNotFoundError:
    print(f"❌ KESALAHAN: File model tidak ditemukan di {MODEL_PATH}")
except Exception as e:
    print(f"❌ Terjadi error saat memuat model: {e}")

# --- 7. Definisi Rute Aplikasi ---

@app.route("/")
def halaman_utama():
    return render_template("Halaman_Utama.html")

@app.route("/grafik")
def halaman_grafik():
    # Ambil statistik dari session saat ini
    session_id = session.get('current_session_id')
    stats = detection_stats.get(session_id, {})
    
    return render_template("Halaman_Grafik.html", stats=stats)

@app.route("/hasil")
def halaman_hasil():
    # Cek apakah ada session_id yang tersimpan
    session_id = session.get('current_session_id')
    
    if session_id and session_id in temp_results:
        # Jika ada data hasil yang tersimpan, tampilkan
        results = temp_results[session_id]
        return render_template("Halaman_Hasil.html", results=results, session_id=session_id)
    else:
        # Jika tidak ada data, redirect ke halaman utama
        return redirect(url_for('halaman_utama'))

@app.route("/api/detection_stats")
def api_detection_stats():
    """API endpoint untuk mendapatkan statistik deteksi dalam format JSON"""
    session_id = session.get('current_session_id')
    stats = detection_stats.get(session_id, {})
    return jsonify(stats)

@app.route("/deteksi", methods=["POST"])
def deteksi():
    uploaded_files = request.files.getlist("files[]")
    
    if not uploaded_files or uploaded_files[0].filename == '':
        return redirect(url_for('halaman_utama'))

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir)

    all_results = []
    summary_content = ""

    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            original_img_path_on_disk = os.path.join(session_dir, filename)
            file.save(original_img_path_on_disk)

            img = Image.open(original_img_path_on_disk).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
            
            with torch.no_grad():
                model.eval()
                prediction = model([img_tensor])[0]

            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            font = ImageFont.load_default()
            
            detection_details = []
            summary_content += f"--- Hasil untuk: {filename} ---\n"
            
            for i in range(len(prediction['boxes'])):
                score = prediction['scores'][i].item()
                if score > CONFIDENCE_THRESHOLD:
                    box = prediction['boxes'][i].cpu().numpy()
                    label_idx = prediction['labels'][i].item()
                    
                    if label_idx > 0 and (label_idx - 1) < len(DISPLAY_CLASSES):
                        label_name = DISPLAY_CLASSES[label_idx - 1]
                        confidence_percent = score * 100
                        detection_details.append({"label": label_name, "confidence": score})
                        
                        summary_content += f"Objek: {label_name.upper()}, Kepercayaan: {confidence_percent:.1f}%\n"

                        color = 'red' if label_name == 'wssv' else 'lime'
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=4)
                        text = f"{label_name}: {score:.2f}"
                        text_y = box[1] - 15 if box[1] > 15 else box[1] + 5
                        draw.text((box[0] + 5, text_y), text, fill=color, font=font)
            
            if not detection_details:
                summary_content += "Tidak ada objek terdeteksi.\n"
            
            summary_content += "\n"

            result_filename = "result_" + filename
            result_img_path_on_disk = os.path.join(session_dir, result_filename)
            img_draw.save(result_img_path_on_disk)

            path_for_template_original = f'uploads/{session_id}/{filename}'
            path_for_template_result = f'uploads/{session_id}/{result_filename}'
            
            all_results.append({
                "original_image_url": path_for_template_original,
                "result_image_url": path_for_template_result,
                "detections": detection_details,
                "original_filename_only": filename
            })
    
    summary_path = os.path.join(session_dir, 'ringkasan_deteksi.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    # Simpan hasil ke dictionary sementara
    temp_results[session_id] = all_results
    
    # Hitung dan simpan statistik deteksi
    stats = calculate_detection_stats(all_results)
    detection_stats[session_id] = stats
    
    # Simpan session_id ke Flask session
    session['current_session_id'] = session_id

    return render_template("Halaman_Hasil.html", results=all_results, session_id=session_id)

@app.route('/download/<session_id>')
def download_hasil(session_id):
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        return redirect(url_for('halaman_utama'))
        
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(session_dir):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f'hasil_deteksi_{session_id}.zip',
        mimetype='application/zip'
    )

# Route tambahan untuk membersihkan session (opsional)
@app.route('/clear_session')
def clear_session():
    session_id = session.get('current_session_id')
    if session_id:
        # Bersihkan data dari dictionary
        temp_results.pop(session_id, None)
        detection_stats.pop(session_id, None)
    
    session.pop('current_session_id', None)
    return redirect(url_for('halaman_utama'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)