# 🔬 Phytoplankton Object Detection with YOLO26

Proyek deteksi objek fitoplankton (*phytoplankton*) berbasis deep learning menggunakan arsitektur mutakhir **YOLO26 (YOLOv26)** dari Ultralytics. Arsitektur proyek dirancang secara **modular**, siap di-push ke **GitHub**, mendukung **pengujian cepat lokal (mini-dataset)**, dan menyediakan **Kaggle Notebook Runner** untuk pelatihan penuh (*full training*) dengan akselerasi GPU.

---

## 📊 Karakteristik & Analisis Dataset

Berdasarkan analisis dataset pada folder `dataset/`:
* **Format Asli**: Pascal VOC (`Annotations/` berupa XML, `JPEGImages/` berupa file JPG, dan split di `ImageSets/Main/`).
* **Total Gambar**: 11.089 gambar mikroskopik beresolusi tinggi (contoh: 1360x1024, 2736x1824, 5440x3648).
* **Total Anotasi XML**: 10.908 file XML.
* **Total Bounding Boxes**: 13.817 bounding boxes.
* **Jumlah Kelas**: **24 Kelas Fitoplankton Unik**
* **Distribusi Split Bawaan**:
  - `train.txt`: 2.726 gambar
  - `val.txt`: 2.728 gambar
  - `test.txt`: 5.457 gambar

### Daftar 24 Kelas Fitoplankton (ID 0 - 23)
| ID | Nama Kelas | ID | Nama Kelas | ID | Nama Kelas |
|---|---|---|---|---|---|
| `0` | *bacteriastrum* | `8` | *corethron* | `16` | *helicotheca* |
| `1` | *biddulphia* | `9` | *coscinodiscus* | `17` | *navicula* |
| `2` | *ceratium carriense* | `10` | *coscinodiscus flank* | `18` | *pleurosigma pelagicum* |
| `3` | *ceratium furca* | `11` | *detonula pumila* | `19` | *protoperidinium* |
| `4` | *ceratium fusus* | `12` | *dinophysis caudata* | `20` | *rhizosolenia* |
| `5` | *ceratium trichoceros* | `13` | *ditylum* | `21` | *skeletonema* |
| `6` | *ceratium tripos* | `14` | *eucampia* | `22` | *thalassionema frauenfeldii* |
| `7` | *chaetoceros* | `15` | *guinardia flaccida* | `23` | *thalassionema nitzschioides* |

---

## ⚡ Mengapa YOLO26 untuk Deteksi Fitoplankton?
1. **End-to-End NMS-Free Inference**: Mengurangi latensi dan menghilangkan ketergantungan pada threshold Non-Maximum Suppression klasik.
2. **STAL (Small-Target-Aware Label Assignment)**: Sangat optimal untuk sel dan koloni fitoplankton yang berukuran kecil atau berbentuk rantai memanjang (*Chaetoceros*, *Skeletonema*, *Thalassionema*).
3. **MuSGD Optimizer**: Stabilitas konvergensi training yang lebih tinggi dibandingkan optimizer konvensional.
4. **Efisiensi Edge & GPU**: Inferensi CPU hingga 43% lebih cepat dan throughput tinggi pada GPU NVIDIA (T4/P100 di Kaggle).

---

## 📁 Struktur Direktori Modular

```
phytoplankton_detection/
├── .gitignore                      # Mengabaikan dataset besar & runs agar aman di-push ke GitHub
├── README.md                       # Dokumentasi lengkap proyek
├── requirements.txt                # Dependensi Python
├── prepare_data.py                 # CLI konversi VOC -> YOLO (mendukung full & mini)
├── train.py                        # CLI training model YOLO26
├── val.py                          # CLI evaluasi metrik (mAP@50, mAP@50-95)
├── predict.py                      # CLI inferensi & visualisasi deteksi
├── run_local_test.py               # 1-Click fast runner lokal (mini sample + 1 epoch)
├── configs/
│   ├── classes.yaml                # Definisi 24 kelas fitoplankton
│   ├── train_kaggle.yaml           # Konfigurasi full training GPU di Kaggle
│   └── train_local.yaml            # Konfigurasi fast test training lokal
├── src/
│   ├── data/
│   │   ├── voc_to_yolo.py          # Konverter koordinat VOC ke YOLO normalisasi
│   │   └── validator.py            # Validasi integritas label dan distribusi dataset
│   ├── models/
│   │   ├── trainer.py              # Wrapper Ultralytics YOLO26 training
│   │   ├── evaluator.py            # Wrapper evaluasi split val & test
│   │   └── predictor.py            # Wrapper inferensi & visualisasi bounding box
│   └── utils/
│       ├── helpers.py              # Helper YAML, logging, dan deteksi device
│       └── visualizer.py           # Render anotasi & bounding box berwarna
├── notebooks/
│   ├── kaggle_training_runner.ipynb # Hybrid Jupyter Notebook (Kaggle GPU & Local Debug)
│   └── kaggle_runner.py            # Hybrid Python runner script (Kaggle & Local)
├── scripts/
│   ├── filter_dataset.py       # Filter dataset VOC per kelas ke sub_dataset/
│   ├── package_for_kaggle.py   # Paketkan sub_dataset ke file ZIP siap upload Kaggle
│   └── setup_git.bat           # Setup Git otomatis & push ke GitHub
└── tests/
    └── test_pipeline.py            # Unit test pipeline otomatis
```

---

## 🚀 Panduan 1: Pengujian Cepat di Komputer Lokal (Local Test)

Fitur ini dirancang agar Anda dapat menguji seluruh pipeline hanya dalam hitungan detik tanpa memakan memori besar atau membutuhkan GPU bertenaga.

### 1. Buat Virtual Environment & Install Dependensi
```bash
# Buat virtual environment (opsional namun disarankan)
py -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install requirements
pip install -r requirements.txt
```

### 2. Jalankan 1-Click Fast Local Test
```bash
python run_local_test.py
```
Skrip ini akan secara otomatis:
1. Mengambil subset kecil (~30 gambar) dari `dataset/` ke `data/yolo_mini/`.
2. Memvalidasi format bounding box dan label YOLO.
3. Menjalankan 1 epoch training dengan `yolo26n.pt` di CPU untuk memastikan kode berjalan lancar.
4. Menjalankan 1 sampel prediksi untuk memastikan pipeline inferensi bekerja.

---

## 📤 Panduan 2: Push ke GitHub

Folder dataset mentah (`dataset/`), dataset hasil konversi (`data/`), dan bobot training (`runs/`, `*.pt`) sudah otomatis diatur dalam `.gitignore` sehingga **tidak akan ikut ter-upload** ke GitHub. Hanya kode, konfigurasi, skrip, dan notebook yang akan di-push.

```bash
# 1. Inisialisasi Git
git init

# 2. Tambahkan semua file yang tidak diabaikan gitignore
git add .

# 3. Commit pertama
git commit -m "feat: initial commit of modular YOLO26 phytoplankton detection pipeline"

# 4. Ganti nama branch utama menjadi main
git branch -M main

# 5. Tambahkan remote repository GitHub Anda (ganti URL di bawah)
git remote add origin https://github.com/USERNAME/phytoplankton_detection.git

# 6. Push ke GitHub
git push -u origin main
```

---

## ☁️ Panduan 3: Menjalankan Full Training di Kaggle

Pelatihan penuh 11.000+ gambar paling optimal dijalankan di GPU Kaggle (NVIDIA T4 x2 atau P100 gratis).

### Langkah 1: Unggah Dataset ke Kaggle
1. File dataset 7 kelas siap upload sudah dibuat di: `sub_dataset_kaggle.zip` (667 MB).
   *(Atau Anda bisa membuat ulang kapan saja dengan menjalankan `py scripts/package_for_kaggle.py`)*
2. Buka [Kaggle Datasets](https://www.kaggle.com/datasets) -> Klik **New Dataset**.
3. Beri nama dataset (misal: `phytoplankton-7classes-dataset`).
4. Drag & drop file `sub_dataset_kaggle.zip` lalu klik **Create**.

### Langkah 2: Buat Notebook Baru di Kaggle
1. Buka [Kaggle Code](https://www.kaggle.com/code) -> Klik **New Notebook**.
2. Pada panel kanan (**Notebook Settings**):
   - **Accelerator**: Pilih **GPU T4 x2** atau **GPU P100**.
   - **Internet**: Pastikan **Internet ON**.
3. Klik **+ Add Input** di kanan atas -> Pilih dataset `phytoplankton-7classes-dataset` yang Anda unggah.

### Langkah 3: Gunakan Notebook Runner
Anda dapat mengimpor file [kaggle_training_runner.ipynb](file:///d:/phytoplankton_detection/notebooks/kaggle_training_runner.ipynb) ke Kaggle (*File -> Upload Notebook*), atau membuat cell dengan urutan berikut:

```python
# 1. Install Ultralytics
!pip install -q --upgrade ultralytics

# 2. Clone repository GitHub Anda
!git clone https://github.com/USERNAME/phytoplankton_detection.git /kaggle/working/phytoplankton_detection
%cd /kaggle/working/phytoplankton_detection

# 3. Konversi Dataset dari /kaggle/input ke format YOLO
!python prepare_data.py --raw_dir /kaggle/input/phytoplankton-voc-dataset/dataset --output_dir data/yolo --mode full

# 4. Jalankan Full Training YOLO26 di GPU
!python train.py --config configs/train_kaggle.yaml --model yolo26s.pt --epochs 100 --batch 16 --imgsz 800 --device 0

# 5. Evaluasi pada Test Set (5.457 gambar test)
!python val.py --weights runs/train/yolo26_phytoplankton_full/weights/best.pt --data data/yolo/data.yaml --split test

# 6. Jalankan Prediksi Sampel
!python predict.py --weights runs/train/yolo26_phytoplankton_full/weights/best.pt --source data/yolo/images/test/ --conf 0.3

# 7. Zip Hasil Pelatihan untuk Di-download
!zip -r /kaggle/working/yolo26_phytoplankton_results.zip runs/train/yolo26_phytoplankton_full/weights/best.pt runs/train/yolo26_phytoplankton_full/results.png runs/train/yolo26_phytoplankton_full/confusion_matrix.png
```

---

## 🛠️ Dokumentasi CLI Commands

### 1. `prepare_data.py` (Dataset Preparation)
```bash
# Konversi penuh semua gambar
python prepare_data.py --raw_dir dataset --output_dir data/yolo --mode full

# Konversi mini untuk pengujian cepat
python prepare_data.py --raw_dir dataset --output_dir data/yolo_mini --mode mini --mini_samples 50
```

### 2. `train.py` (Model Training)
```bash
# Full training dengan konfigurasi Kaggle
python train.py --config configs/train_kaggle.yaml --model yolo26s.pt --epochs 100 --batch 16

# Custom override via CLI
python train.py --model yolo26m.pt --epochs 50 --batch 8 --imgsz 1024 --device 0
```

### 3. `val.py` (Evaluation)
```bash
python val.py --weights runs/train/yolo26_phytoplankton_full/weights/best.pt --data data/yolo/data.yaml --split test
```

### 4. `predict.py` (Inference)
```bash
# Prediksi satu gambar
python predict.py --weights path/to/best.pt --source sample.jpg --conf 0.3

# Prediksi seluruh folder
python predict.py --weights path/to/best.pt --source data/yolo/images/test/ --conf 0.25 --save_dir runs/my_predictions
```

### 5. `scripts/filter_dataset.py` (Filter Dataset per Kelas)
Memfilter dataset VOC berdasarkan jenis fitoplankton tertentu ke folder `sub_dataset/`:

```bash
# Mode interaktif (menu pilihan nomor kelas)
python scripts/filter_dataset.py

# Pilih kelas tertentu langsung via CLI
python scripts/filter_dataset.py --classes "chaetoceros" "coscinodiscus" "eucampia"

# Pilih semua kelas yang mengandung kata kunci
python scripts/filter_dataset.py --search "ceratium"

# Konversi sub_dataset hasil filter ke YOLO dan latih
python prepare_data.py --raw_dir sub_dataset --output_dir data/yolo_sub --classes_config sub_dataset/classes.yaml
python train.py --data data/yolo_sub/data.yaml --epochs 20 --batch 4 --imgsz 416 --device cpu
```

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah lisensi MIT.
