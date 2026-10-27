# 🎓 Panduan Presentasi & Diskusi dengan Promotor: Menjelaskan YOLO untuk Riset Biologi

Dokumen ini adalah pendamping untuk file slide LaTeX Beamer: [`presentasi_yolo_fitoplankton.tex`](file:///d:/phytoplankton_detection/docs/presentasi_yolo_fitoplankton.tex).

---

## 🚀 1. Cara Menjalankan di Overleaf

1. Buka [Overleaf](https://www.overleaf.com/) dan buat **New Project** $\to$ **Blank Project** (misal beri nama `Presentasi_YOLO_Fitoplankton`).
2. Hapus isi file bawaan `main.tex`, lalu salin seluruh isi dari file [`presentasi_yolo_fitoplankton.tex`](file:///d:/phytoplankton_detection/docs/presentasi_yolo_fitoplankton.tex) ke dalam `main.tex` di Overleaf.
3. Klik tombol hijau **Recompile**.
4. Slide Anda akan langsung ter-render dengan sempurna tanpa memerlukan instalasi paket tambahan karena seluruh pustaka yang digunakan (`tikz`, `tcolorbox`, `lmodern`, `booktabs`) merupakan paket standar baku TeX Live di Overleaf.

---

## 🎙️ 2. Cara Mengaktifkan Catatan Presenter (*Speaker Notes*)

Di baris 8–12 pada file LaTeX, terdapat opsi konfigurasi catatan:

```latex
% Pilihan Tampilan:
% 1. Hanya slide (Default saat presentasi resmi di depan proyektor):
\setbeameroption{hide notes}

% 2. Mode Gladi Bersih (Layar terbagi 2: Slide di Kiri, Contekan Catatan di Kanan):
% \setbeameroption{show notes on second screen=right}

% 3. Hanya lembar catatan (bisa di-print untuk dipegang di tangan saat berbicara):
% \setbeameroption{show only notes}
```

> **Tips Praktis:** Saat Anda latihan sendiri atau gladi bersih di rumah, aktifkan opsi ke-2 (`show notes on second screen=right`). Anda akan melihat slide sekaligus contekan kata-kata yang harus Anda ucapkan di sebelah kanannya!

---

## 🧠 3. Tabel Konversi: Jargon Komputer $\longrightarrow$ Analogi Biologi Murni

Gunakan analogi ini jika promotor merasa bingung dengan istilah-istilah kecerdasan buatan:

| Istilah TI / Komputer | Analogi dalam Dunia Biologi / Mikroskopi | Penjelasan Ramah Promotor |
|---|---|---|
| **YOLO (*You Only Look Once*)** | Taksonom senior yang melayangkan pandangan sekilas (*single glance*) | Berbeda dengan asisten baru yang menggeser mikroskop millimeter demi millimeter, YOLO langsung memetakan semua sel dalam satu sapuan pandang. |
| **Object Detection** | Sensus cacah sel individu di kamar hitung | Bukan cuma menebak *"foto ini ada fitoplankton atau tidak"*, tapi menggambar kotak di setiap sel, menyebut spesiesnya, dan menghitung total kelimpahan (sel/L). |
| **Bounding Box** | Lingkaran atau kotak penanda di kaca preparat mikroskop | Koordinat koordinat spasial $[x, y, \text{lebar}, \text{tinggi}]$ yang mengisolasi satu individu sel dari matriks air di sekitarnya. |
| **Backbone (Neural Network)** | Reseptor retina & korteks visual otak | Lapisan saraf pendeteksi ciri morfologis: kurva katup silika (*valve*), celah *raphe*, duri halus (*setae* *Chaetoceros*), atau tanduk *Ceratium*. |
| **Neck / Feature Pyramid** | Pengatur perbesaran mikroskop ($10\times$ vs $40\times$ vs $100\times$) | Menggabungkan resolusi tinggi untuk mendeteksi sel mikro ($10\,\mu\text{m}$) sekaligus menangkap koloni rantai raksasa ($>300\,\mu\text{m}$) agar tidak terpotong. |
| **Training (Pelatihan)** | Bimbingan mahasiswa baru menggunakan buku atlas identifikasi | Model diberi ribuan foto contoh yang sudah diberi label benar (*ground truth*), lalu model berlatih mengenali pola sampai jarang berbuat salah. |
| **Loss Function** | Koreksi dan coretan merah dari dosen taksonom | Ukuran seberapa besar tebakan model meleset (salah posisi kotak atau salah menduga spesies). Nilai ini dipaksa sekecil mungkin selama latihan. |
| **Epoch** | Satu putaran penuh membaca seluruh koleksi preparat di laboratorium | 1 Epoch = Model telah memeriksa seluruh 11.089 foto dalam dataset satu kali putaran. |
| **Precision (Presisi)** | Kemurnian Taksonomi (*Taxonomic Purity*) | Jika model bilang ada 100 *Dinophysis*, berapa persen yang memang benar *Dinophysis* asli dan bukan kotoran sedimen? Mencegah *False Alarm*. |
| **Recall (Sensitivitas)** | Kelengkapan Sensus (*Census Completeness*) | Dari 100 sel *Chaetoceros* yang riil ada di preparat, berapa banyak yang berhasil ditemukan tanpa terlewat? Mencegah estimasi kelimpahan terlalu rendah (*underestimation*). |
| **mAP (*mean Average Precision*)** | Indeks Mutu Komposit Akurasi Global | Rata-rata terbobot keakuratan model pada seluruh 24 spesies fitoplankton yang diuji. |

---

## 🎯 4. Antisipasi 5 Pertanyaan Jebakan Promotor Biologi & Jawaban Elegan

### ❓ Pertanyaan 1: *"Apakah AI ini bisa menggantikan peran ahli taksonomi fitoplankton?"*
* **Jawaban Anda:**  
  *"Sama sekali tidak, Prof. AI ini tidak memiliki pemahaman biologis otonom. Kualitas model ini 100% bergantung pada kurasi taksonomi yang kita ajarkan di awal (Ground Truth). Model ini fungsinya persis seperti mikroskop atau spektrofotometer berkecepatan tinggi: ia adalah asisten digital yang membebaskan mata kita dari kelelahan sensus manual ribuan jam, sehingga waktu kita sebagai peneliti biologi bisa dialokasikan untuk memikirkan analisis ekologi, interaksi trofik, dan dinamika populasi."*

---

### ❓ Pertanyaan 2: *"Bagaimana kalau air lautnya keruh, banyak kotoran sedimen, partikel detritus, atau gelembung udara?"*
* **Jawaban Anda:**  
  *"Dalam framework YOLO, partikel non-target tersebut dikelompokkan sebagai kelas 'Background'. Model dilatih mengenali keteraturan pola biologis (seperti simetri silika atau struktur setae berulang). Serpihan detritus yang acak dan tidak beraturan akan menghasilkan skor keyakinan (*confidence score*) yang sangat rendah ($<0.4$), sehingga secara otomatis difilter dan tidak akan dihitung sebagai sel fitoplankton."*

---

### ❓ Pertanyaan 3: *"Fitoplankton kan sering membentuk rantai panjang (seperti Chaetoceros atau Skeletonema) atau selnya bertumpuk (overlapping). Apakah model tidak bingung?"*
* **Jawaban Anda:**  
  *"Itulah alasan utama kita memilih YOLO versi mutakhir yang memiliki modul STAL (*Small-Target-Aware Label Assignment*). Model ini dirancang khusus untuk membedakan batas-batas individual sel yang berdekatan atau bersilangan. Setiap sel dalam rantai atau tumpukan akan diprediksi sebagai bounding box independen berdasarkan pusat massanya masing-masing."*

---

### ❓ Pertanyaan 4: *"Apakah AI ini tidak berhalusinasi atau mengarang spesies baru yang tidak ada?"*
* **Jawaban Anda:**  
  *"Penting untuk membedakan antara AI Generatif (seperti ChatGPT yang menyusun kalimat dan bisa berhalusinasi) dengan AI Deteksi Objek Diskriminatif seperti YOLO. YOLO bekerja dengan ruang hipotesis tertutup (*closed-set classification*): ia hanya menghitung probabilitas matematis dari 24 taksa yang telah kita tetapkan dalam kamus taksonomi. Jika ada benda aneh yang tidak mirip dengan ke-24 taksa tersebut, model akan mengabaikannya sebagai latar belakang, bukan mengarang nama spesies baru."*

---

### ❓ Pertanyaan 5: *"Mengapa topik ini layak menjadi disertasi Doktor Biologi dan bukan Skripsi Teknik Informatika?"*
* **Jawaban Anda:**  
  *"Dalam disertasi ini, fokus kebaruan ilmiah (*scientific novelty*) bukan pada penciptaan algoritma matematika baru, melainkan pada pemecahan masalah fundamental biologi kelautan: yaitu mengatasi 'bottleneck' kuantifikasi mikroskopi yang selama setengah abad membatasi resolusi spasio-temporal studi ekologi plankton. Kontribusi biologi kita mencakup: pembentukan basis data morfologi 24 taksa fitoplankton tropis, validasi korelasi sensus otomatis terhadap metode standar Sedgwick-Rafter, serta analisis dinamika suksesi komunitas fitoplankton terhadap fluktuasi oseanografi dalam skala sampel masif yang belum pernah bisa dilakukan sebelumnya."*

---

## 📊 5. Data Konkret Proyek Kita yang Siap Disebutkan

Saat promotor menanyakan skala data penelitian Anda:
* **Jumlah Taksa**: 24 Taksa Fitoplankton unik (termasuk Diatom sentrik, Diatom pennat, dan Dinoflagellata penting seperti *Ceratium* dan *Dinophysis*).
* **Total Citra Mikroskop**: 11.089 citra mikroskopik resolusi tinggi.
* **Total Anotasi Kotak (*Bounding Boxes*)**: 13.817 kotak sel terverifikasi.
* **Distribusi Split Uji**: 
  * Training: 2.726 gambar
  * Validasi: 2.728 gambar
  * Blind Testing (Uji Buta): 5.457 gambar
