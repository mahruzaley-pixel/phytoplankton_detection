"""
Script untuk Mengompresi dan Menyiapkan Sub-Dataset ke Format Upload Kaggle
==========================================================================
Mengompresi folder sub_dataset/ menjadi file ZIP yang siap diunggah
ke Kaggle Datasets (baik melalui Web UI maupun Kaggle CLI).
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

# Fix encoding di terminal Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def create_kaggle_metadata(output_dir: Path, title: str = "Phytoplankton 7 Classes Dataset"):
    """Membuat file dataset-metadata.json standar Kaggle."""
    slug = title.lower().replace(" ", "-")
    metadata = {
        "title": title,
        "id": f"YOUR_KAGGLE_USERNAME/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = output_dir / "dataset-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def package_dataset(dataset_dir: Path, output_zip: Path, include_metadata: bool = True):
    """
    Mengompresi folder dataset ke dalam file ZIP.
    Menjaga struktur folder internal:
      - Annotations/*.xml
      - JPEGImages/*.jpg
      - ImageSets/Main/*.txt
      - classes.yaml
      - dataset-metadata.json (opsional untuk Kaggle CLI)
    """
    if not dataset_dir.exists():
        print(f"[ERROR] Folder dataset tidak ditemukan di: {dataset_dir}")
        sys.exit(1)

    annot_dir = dataset_dir / "Annotations"
    images_dir = dataset_dir / "JPEGImages"

    if not annot_dir.exists() or not images_dir.exists():
        print(f"[ERROR] Folder dataset harus memiliki subfolder 'Annotations' dan 'JPEGImages'.")
        sys.exit(1)

    xml_files = list(annot_dir.glob("*.xml"))
    jpg_files = list(images_dir.glob("*.*"))

    print("=" * 60)
    print("  MENYIAPKAN DATASET UNTUK KAGGLE")
    print("=" * 60)
    print(f"  Source directory : {dataset_dir.resolve()}")
    print(f"  Total XML        : {len(xml_files)} file")
    print(f"  Total Citra      : {len(jpg_files)} file")
    print(f"  Target ZIP       : {output_zip.resolve()}")
    print("=" * 60)

    # Buat metadata jika diminta
    meta_path = None
    if include_metadata:
        meta_path = create_kaggle_metadata(dataset_dir)
        print(f"  [+] Metadata Kaggle dibuat: {meta_path.name}")

    # Kumpulkan semua file yang akan dimasukkan ke ZIP
    files_to_pack = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            full_path = Path(root) / f
            # Relatif terhadap dataset_dir agar saat diekstrak di Kaggle posisinya rapi
            rel_path = full_path.relative_to(dataset_dir)
            files_to_pack.append((full_path, rel_path))

    print(f"\n  Mengompresi {len(files_to_pack)} file ke {output_zip.name}...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for full_path, rel_path in tqdm(files_to_pack, desc="Packaging ZIP", unit="file"):
            zipf.write(full_path, arcname=str(rel_path))

    size_mb = output_zip.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("  SUKSES MEMBUNGKUS DATASET!")
    print("=" * 60)
    print(f"  File ZIP      : {output_zip.resolve()}")
    print(f"  Ukuran ZIP    : {size_mb:.2f} MB")
    print(f"  Total Isi     : {len(files_to_pack)} file")
    print("=" * 60)
    print("\n  CARA UPLOAD KE KAGGLE:")
    print("  1. Buka https://www.kaggle.com/datasets/new di browser Anda.")
    print("  2. Beri judul dataset, misal: 'Phytoplankton 7 Classes Dataset'")
    print(f"  3. Drag & drop file: {output_zip.name}")
    print("  4. Klik 'Create' dan tunggu proses upload selesai.")
    print("  5. Setelah selesai, dataset siap dihubungkan ke Notebook Kaggle.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Package dataset ke file ZIP untuk upload Kaggle.")
    parser.add_argument("--source", type=str, default="sub_dataset", help="Folder dataset yang akan dipaketkan (default: sub_dataset).")
    parser.add_argument("--output", type=str, default="sub_dataset_kaggle.zip", help="Nama file ZIP output (default: sub_dataset_kaggle.zip).")
    parser.add_argument("--no_meta", action="store_true", help="Jangan sertakan dataset-metadata.json.")
    args = parser.parse_args()

    package_dataset(
        dataset_dir=Path(args.source),
        output_zip=Path(args.output),
        include_metadata=not args.no_meta,
    )


if __name__ == "__main__":
    main()
