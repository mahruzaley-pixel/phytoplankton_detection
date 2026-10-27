"""
Universal Phytoplankton YOLO26 Pipeline Runner (Kaggle & Local Hybrid)
=======================================================================
Script ini dapat dijalankan baik di Kaggle GPU maupun di komputer lokal (CPU/GPU):
  1. Di Kaggle (Full Training 100 Epochs):
     python notebooks/kaggle_runner.py --repo_url https://github.com/USERNAME/phytoplankton_detection.git

  2. Di Lokal (Local Debugging / Fast Test):
     python notebooks/kaggle_runner.py --debug
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run_cmd(cmd: str):
    """Menjalankan perintah shell dengan penanganan error."""
    print(f"\n[EXEC]: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR] Perintah gagal dengan kode {res.returncode}: {cmd}")
        sys.exit(res.returncode)


def detect_environment():
    """Mendeteksi apakah script berjalan di Kaggle atau di komputer lokal."""
    is_kaggle = Path("/kaggle/working").exists() or Path("/kaggle/input").exists()
    return "kaggle" if is_kaggle else "local"


def detect_dataset_dir(env: str) -> Path:
    """Mendeteksi path dataset yang tersedia (Kaggle input vs local sub_dataset/dataset)."""
    if env == "kaggle":
        possible_annots = glob.glob("/kaggle/input/**/Annotations", recursive=True)
        if possible_annots:
            detected = Path(possible_annots[0]).parent
            print(f"[DATASET] Terdeteksi Kaggle Dataset di: {detected}")
            return detected
        print("[WARNING] Tidak ditemukan dataset di /kaggle/input, mencari di folder lokal...")

    # Cek folder lokal
    for candidate in [Path("sub_dataset"), Path("dataset"), Path("../sub_dataset"), Path("../dataset")]:
        if candidate.exists() and (candidate / "Annotations").exists():
            print(f"[DATASET] Menggunakan dataset lokal: {candidate.resolve()}")
            return candidate

    print("[ERROR] Dataset tidak ditemukan! Pastikan dataset terhubung di Kaggle atau ada folder 'sub_dataset/' secara lokal.")
    sys.exit(1)


def find_classes_config(dataset_dir: Path) -> Path:
    """Mencari file classes.yaml yang sesuai (milik sub_dataset atau configs default)."""
    # Prioritaskan classes.yaml yang ada di folder dataset (hasil filtering 7 kelas)
    if (dataset_dir / "classes.yaml").exists():
        return dataset_dir / "classes.yaml"
    if Path("configs/classes.yaml").exists():
        return Path("configs/classes.yaml")
    return Path("configs/classes.yaml")


def package_results(output_zip: Path, result_dir: Path, extra_files: list):
    """Mengompresi hasil pelatihan ke ZIP secara cross-platform menggunakan Python zipfile."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[EXPORT] Mengompresi hasil eksperimen ke: {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        if result_dir.exists():
            for root, _, files in os.walk(result_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(result_dir.parent)
                    zipf.write(full_p, arcname=str(rel_p))
        for extra in extra_files:
            extra_p = Path(extra)
            if extra_p.exists():
                zipf.write(extra_p, arcname=extra_p.name)
    print(f"[EXPORT] Selesai! Ukuran file: {output_zip.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Universal YOLO26 Phytoplankton Pipeline Runner.")
    parser.add_argument("--repo_url", type=str, default="", help="GitHub repository URL to clone (khusus di Kaggle).")
    parser.add_argument("--debug", action="store_true", help="Mode debug lokal (sedikit sampel & epoch singkat).")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch training (default: 100).")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Varian model YOLO26 (default: yolo26s.pt).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size training (default: 16).")
    parser.add_argument("--imgsz", type=int, default=800, help="Resolusi gambar (default: 800).")
    parser.add_argument("--device", type=str, default="auto", help="Device target ('auto', 'cpu', '0').")
    args = parser.parse_args()

    env = detect_environment()
    print("=" * 65)
    print(f"  PHYTAPLANKTON YOLO26 RUNNER | Environment: {env.upper()}")
    print("=" * 65)

    # Langkah 1: Kloning repositori GitHub jika berjalan di Kaggle
    if env == "kaggle" and args.repo_url:
        working_dir = Path("/kaggle/working")
        repo_name = args.repo_url.split("/")[-1].replace(".git", "")
        project_dir = working_dir / repo_name
        if not project_dir.exists():
            print(f"[GIT] Mengkloning {args.repo_url} ke {project_dir}...")
            run_cmd(f"git clone {args.repo_url} {project_dir}")
        else:
            print(f"[GIT] Repositori sudah ada. Memperbarui via git pull...")
            run_cmd(f"cd {project_dir} && git pull")
        os.chdir(project_dir)
        print(f"[CWD] Berpindah direktori kerja ke: {os.getcwd()}")
    else:
        print(f"[GIT] Menjalankan kode pada direktori saat ini: {Path.cwd()}")

    # Langkah 2: Setup konfigurasi sesuai mode
    if args.debug:
        print("\n[MODE] === DEBUG MODE AKTIF (Local Fast Verification) ===")
        epochs = 2
        batch = 4
        imgsz = 416
        prep_mode = "mini"
        mini_samples = 30
        device = "cpu" if args.device == "auto" else args.device
        model_name = "yolo26n.pt"
    else:
        print("\n[MODE] === FULL TRAINING MODE ===")
        epochs = args.epochs
        batch = args.batch
        imgsz = args.imgsz
        prep_mode = "full"
        mini_samples = 40
        model_name = args.model
        if args.device == "auto":
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = args.device

    print(f"  Model  : {model_name}")
    print(f"  Epochs : {epochs}")
    print(f"  Batch  : {batch}")
    print(f"  Imgsz  : {imgsz}")
    print(f"  Device : {device}")

    # Langkah 3: Deteksi dataset & konfigurasi kelas
    dataset_dir = detect_dataset_dir(env)
    classes_config = find_classes_config(dataset_dir)
    print(f"[CONFIG] Menggunakan konfigurasi kelas: {classes_config}")

    # Langkah 4: Konversi dataset ke format YOLO
    yolo_output_dir = Path("data/yolo_debug" if args.debug else "data/yolo")
    print(f"\n[STEP 1/4] Mengonversi dataset ke format YOLO...")
    prep_cmd = (
        f"{sys.executable} prepare_data.py "
        f"--raw_dir \"{dataset_dir}\" "
        f"--output_dir \"{yolo_output_dir}\" "
        f"--mode {prep_mode} "
        f"--mini_samples {mini_samples} "
        f"--classes_config \"{classes_config}\""
    )
    run_cmd(prep_cmd)

    yolo_data_yaml = yolo_output_dir / "data.yaml"

    # Langkah 5: Pelatihan Model (Training)
    print(f"\n[STEP 2/4] Melatih Model YOLO26 ({model_name})...")
    exp_name = "yolo26_debug" if args.debug else "yolo26_phytoplankton_full"
    train_cmd = (
        f"{sys.executable} train.py "
        f"--data \"{yolo_data_yaml}\" "
        f"--model {model_name} "
        f"--epochs {epochs} "
        f"--batch {batch} "
        f"--imgsz {imgsz} "
        f"--device {device} "
        f"--name {exp_name}"
    )
    run_cmd(train_cmd)

    # Langkah 6: Evaluasi Metrik pada Test Split
    candidates = list(Path("runs").rglob(f"*{exp_name}*/weights/best.pt"))
    if not candidates:
        candidates = list(Path("runs").rglob("best.pt"))
    best_weights = candidates[0] if candidates else Path(f"runs/train/{exp_name}/weights/best.pt")

    if best_weights.exists():
        print(f"\n[STEP 3/4] Evaluasi Metrik pada Test Set ({best_weights})...")
        val_cmd = (
            f"{sys.executable} val.py "
            f"--weights \"{best_weights}\" "
            f"--data \"{yolo_data_yaml}\" "
            f"--split test "
            f"--imgsz {imgsz} "
            f"--device {device}"
        )
        run_cmd(val_cmd)

        # Langkah 7: Prediksi Sampel
        print(f"\n[STEP 4/4] Inferensi Sample Test Images...")
        test_images_dir = yolo_output_dir / "images" / "test"
        predict_cmd = (
            f"{sys.executable} predict.py "
            f"--weights \"{best_weights}\" "
            f"--source \"{test_images_dir}\" "
            f"--conf 0.15 "
            f"--imgsz {imgsz} "
            f"--device {device} "
            f"--save_dir \"runs/predict_{exp_name}\""
        )
        run_cmd(predict_cmd)

    # Langkah 8: Export Hasil ke ZIP
    if env == "kaggle":
        export_zip = Path("/kaggle/working/yolo26_phytoplankton_results.zip")
    else:
        export_zip = Path("runs/yolo26_results.zip")

    exp_dirs = [p for p in Path("runs").rglob(exp_name) if p.is_dir() and (p / "weights").exists()]
    actual_exp_dir = exp_dirs[0] if exp_dirs else Path(f"runs/train/{exp_name}")

    package_results(
        output_zip=export_zip,
        result_dir=actual_exp_dir,
        extra_files=[str(yolo_data_yaml), str(classes_config)],
    )

    print("\n" + "=" * 65)
    print("  [SUCCESS] Seluruh pipeline berhasil diselesaikan!")
    print(f"  File hasil: {export_zip.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
