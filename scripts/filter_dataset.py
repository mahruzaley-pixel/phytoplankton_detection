"""
Filter Dataset Fitoplankton Berdasarkan Jenis Tertentu
=======================================================
Script ini memfilter dataset Pascal VOC (Annotations + JPEGImages)
berdasarkan kelas fitoplankton yang dipilih, lalu menyalin hasilnya
ke folder sub_dataset/ dengan struktur yang sama.

Fitur:
  - Otomatis menghapus bounding box kelas lain yang tidak dipilih (clean isolation).
  - Menggabungkan varian (misal: 'coscinodiscus flank' -> 'coscinodiscus').
  - Mendukung alias genus (misal: 'pleurosigma' -> 'pleurosigma pelagicum').
  - Otomatis membuat sub_dataset/classes.yaml siap pakai untuk training YOLO.
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Fix encoding di terminal Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ============================
# Daftar 24 kelas fitoplankton asli
# ============================
ALL_CLASSES = [
    "bacteriastrum",
    "biddulphia",
    "ceratium carriense",
    "ceratium furca",
    "ceratium fusus",
    "ceratium trichoceros",
    "ceratium tripos",
    "chaetoceros",
    "corethron",
    "coscinodiscus",
    "coscinodiscus flank",
    "detonula pumila",
    "dinophysis caudata",
    "ditylum",
    "eucampia",
    "guinardia flaccida",
    "helicotheca",
    "navicula",
    "pleurosigma pelagicum",
    "protoperidinium",
    "rhizosolenia",
    "skeletonema",
    "thalassionema frauenfeldii",
    "thalassionema nitzschioides",
]

# Pemetaan alias nama pendek/genus ke nama kelas dataset
GENUS_ALIASES = {
    "pleurosigma": "pleurosigma pelagicum",
}

# Pemetaan default penggabungan varian/orientasi
DEFAULT_CLASS_REMAP = {
    "coscinodiscus flank": "coscinodiscus",
}


def scan_dataset(annot_dir: Path, class_remap: dict = None) -> Counter:
    """Menghitung jumlah objek per kelas di seluruh dataset."""
    class_counts = Counter()
    for xml_file in annot_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall("object"):
                name = obj.find("name")
                if name is not None and name.text:
                    cls = name.text.strip().lower()
                    if class_remap and cls in class_remap:
                        cls = class_remap[cls]
                    class_counts[cls] += 1
        except ET.ParseError:
            pass
    return class_counts


def filter_xml(xml_path: Path, selected_classes: set, class_remap: dict = None) -> bool:
    """
    Cek apakah file XML mengandung setidaknya satu objek
    dari kelas yang dipilih.
    """
    try:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name = obj.find("name")
            if name is not None and name.text:
                cls = name.text.strip().lower()
                if class_remap and cls in class_remap:
                    cls = class_remap[cls]
                if cls in selected_classes:
                    return True
    except ET.ParseError:
        pass
    return False


def interactive_select(class_counts: Counter) -> list:
    """Mode interaktif: tampilkan menu dan biarkan user memilih."""
    print("\n" + "=" * 60)
    print("  DAFTAR KELAS FITOPLANKTON YANG TERSEDIA")
    print("=" * 60)

    unique_classes = sorted(list(class_counts.keys()))
    for idx, cls_name in enumerate(unique_classes, 1):
        count = class_counts.get(cls_name, 0)
        marker = f"({count:5d} objek)" if count > 0 else "(    0 objek)"
        print(f"  [{idx:2d}] {cls_name:30s}  {marker}")

    print("\n" + "-" * 60)
    print("  Masukkan nomor kelas yang ingin diambil (pisahkan koma)")
    print("  Contoh: 1,7,14  atau  all  untuk semua")
    print("-" * 60)

    while True:
        user_input = input("\n  Pilihan Anda: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "all":
            return unique_classes.copy()

        try:
            indices = [int(x.strip()) for x in user_input.split(",")]
            selected = []
            for i in indices:
                if 1 <= i <= len(unique_classes):
                    selected.append(unique_classes[i - 1])
                else:
                    print(f"  [!] Nomor {i} tidak valid (harus 1-{len(unique_classes)})")
                    selected = []
                    break
            if selected:
                return selected
        except ValueError:
            print("  [!] Format salah. Gunakan angka dipisahkan koma, contoh: 1,7,14")


def copy_filtered_dataset(
    dataset_dir: Path,
    output_dir: Path,
    selected_classes: list,
    keep_only_selected: bool = True,
    class_remap: dict = None,
):
    """
    Menyalin gambar dan anotasi yang mengandung kelas terpilih
    ke folder output. Jika keep_only_selected=True, bounding box
    dari kelas yang tidak termasuk akan dihilangkan dari file XML.

    Args:
        dataset_dir: Path ke folder dataset (berisi Annotations/ dan JPEGImages/)
        output_dir: Path ke folder output (sub_dataset/)
        selected_classes: Daftar nama kelas yang dipilih
        keep_only_selected: Jika True, hapus bounding box kelas lain dari XML.
        class_remap: Dictionary untuk memetakan nama kelas (misal flank -> genus)
    """
    annot_dir = dataset_dir / "Annotations"
    images_dir = dataset_dir / "JPEGImages"
    imagesets_dir = dataset_dir / "ImageSets" / "Main"

    out_annot = output_dir / "Annotations"
    out_images = output_dir / "JPEGImages"
    out_imagesets = output_dir / "ImageSets" / "Main"

    # Buat folder output
    out_annot.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)
    out_imagesets.mkdir(parents=True, exist_ok=True)

    selected_set = {c.strip().lower() for c in selected_classes}
    copied_stems = set()
    total_boxes = 0
    class_box_counts = Counter()
    removed_outside_boxes = 0

    print(f"\n  Memproses anotasi...")
    xml_files = sorted(annot_dir.glob("*.xml"))

    for xml_path in xml_files:
        if not filter_xml(xml_path, selected_set, class_remap=class_remap):
            continue

        stem = xml_path.stem

        # --- Proses XML ---
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if keep_only_selected:
            # Salin XML tapi hilangkan objek yang tidak termasuk dalam selected_set
            objects_to_remove = []
            for obj in root.findall("object"):
                name = obj.find("name")
                if name is not None and name.text:
                    cls = name.text.strip().lower()
                    if class_remap and cls in class_remap:
                        cls = class_remap[cls]
                        name.text = cls  # Rename di tag XML

                    if cls not in selected_set:
                        objects_to_remove.append(obj)
                        removed_outside_boxes += 1
                    else:
                        class_box_counts[cls] += 1
                        total_boxes += 1
                else:
                    objects_to_remove.append(obj)

            for obj in objects_to_remove:
                root.remove(obj)

            # Tulis file XML bersih berformat UTF-8
            tree.write(str(out_annot / xml_path.name), encoding="utf-8", xml_declaration=True)
        else:
            # Salin XML utuh, update nama remap jika ada
            if class_remap:
                for obj in root.findall("object"):
                    name = obj.find("name")
                    if name is not None and name.text:
                        cls = name.text.strip().lower()
                        if cls in class_remap:
                            cls = class_remap[cls]
                            name.text = cls
                        if cls in selected_set:
                            class_box_counts[cls] += 1
                            total_boxes += 1
                tree.write(str(out_annot / xml_path.name), encoding="utf-8", xml_declaration=True)
            else:
                for obj in root.findall("object"):
                    name = obj.find("name")
                    if name is not None and name.text:
                        cls = name.text.strip().lower()
                        if cls in selected_set:
                            class_box_counts[cls] += 1
                            total_boxes += 1
                shutil.copy2(xml_path, out_annot / xml_path.name)

        # --- Salin gambar ---
        img_path = images_dir / f"{stem}.jpg"
        if img_path.exists():
            shutil.copy2(img_path, out_images / img_path.name)

        copied_stems.add(stem)

    # --- Buat ulang file split (train/val/test) ---
    for split_name in ["train", "val", "test", "trainval"]:
        split_file = imagesets_dir / f"{split_name}.txt"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                original_stems = {line.strip() for line in f if line.strip()}

            filtered_stems = sorted(original_stems & copied_stems)
            with open(out_imagesets / f"{split_name}.txt", "w", encoding="utf-8") as f:
                for s in filtered_stems:
                    f.write(s + "\n")

            print(f"  Split '{split_name}': {len(filtered_stems)} gambar (dari {len(original_stems)} asli)")

    return copied_stems, total_boxes, class_box_counts, removed_outside_boxes


def resolve_class_names(classes_input: list) -> list:
    """Menyelesaikan alias genus (seperti 'pleurosigma' -> 'pleurosigma pelagicum')."""
    resolved = []
    all_lower = [x.lower() for x in ALL_CLASSES]

    for c in classes_input:
        c_clean = c.strip().lower()
        if c_clean in all_lower:
            resolved.append(c_clean)
        elif c_clean in GENUS_ALIASES:
            resolved.append(GENUS_ALIASES[c_clean])
        else:
            # Cari apakah c_clean adalah prefix/genus yang unik
            matches = [x for x in all_lower if x.startswith(c_clean + " ") or x == c_clean]
            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                resolved.extend(matches)
            else:
                resolved.append(c_clean)

    return list(dict.fromkeys(resolved))


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset fitoplankton berdasarkan kelas tertentu."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=str,
        default=None,
        help='Nama kelas yang akan diambil. Contoh: --classes "chaetoceros" "eucampia"',
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help='Ambil semua kelas yang mengandung kata tertentu. Contoh: --search "ceratium"',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path ke folder dataset VOC (default: dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sub_dataset",
        help="Path folder output (default: sub_dataset)",
    )
    parser.add_argument(
        "--no_keep_only",
        action="store_true",
        help="Jika diset, objek kelas di luar pilihan TIDAK dihapus dari XML.",
    )
    parser.add_argument(
        "--no_merge_flank",
        action="store_true",
        help="Jangan gabungkan 'coscinodiscus flank' ke 'coscinodiscus'.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    annot_dir = dataset_dir / "Annotations"

    if not annot_dir.exists():
        print(f"[ERROR] Folder Annotations tidak ditemukan di: {annot_dir}")
        sys.exit(1)

    class_remap = None if args.no_merge_flank else DEFAULT_CLASS_REMAP.copy()
    keep_only_selected = not args.no_keep_only

    # Scan distribusi kelas
    print("\n  Memindai dataset...")
    class_counts = scan_dataset(annot_dir, class_remap=class_remap)
    total_annotations = sum(class_counts.values())
    print(f"  Ditemukan {total_annotations} objek dari {len(class_counts)} kelas.\n")

    # Tentukan kelas yang dipilih
    if args.search:
        keyword = args.search.strip().lower()
        selected = [c for c in class_counts.keys() if keyword in c.lower()]
        if not selected:
            print(f"[ERROR] Tidak ada kelas yang mengandung '{args.search}'")
            sys.exit(1)
        print(f"  Kelas yang cocok dengan '{args.search}':")
        for c in selected:
            print(f"    - {c} ({class_counts.get(c, 0)} objek)")
    elif args.classes:
        raw_selected = [c.strip().lower() for c in args.classes]
        selected = resolve_class_names(raw_selected)
        # Validasi
        available_classes = set(class_counts.keys())
        invalid = [c for c in selected if c not in available_classes]
        if invalid:
            print(f"[ERROR] Kelas tidak dikenal: {invalid}")
            print(f"  Kelas yang tersedia: {sorted(list(available_classes))}")
            sys.exit(1)
    else:
        # Mode interaktif
        selected = interactive_select(class_counts)

    if not selected:
        print("[ERROR] Tidak ada kelas yang dipilih.")
        sys.exit(1)

    # Konfirmasi
    print("\n" + "=" * 60)
    print("  KELAS YANG DIPILIH:")
    print("=" * 60)
    for c in selected:
        print(f"    * {c} ({class_counts.get(c, 0)} objek)")

    if keep_only_selected:
        print("  [!] Bounding box dari kelas di luar daftar di atas AKAN DIHAPUS dari XML.")
    if class_remap:
        print(f"  [!] Penggabungan varian aktif: {class_remap}")

    # Bersihkan output jika sudah ada
    if output_dir.exists():
        print(f"\n  [INFO] Folder '{output_dir}' sudah ada, akan ditimpa.")
        shutil.rmtree(output_dir)

    # Jalankan filter
    print(f"\n  Menyalin & membersihkan anotasi ke '{output_dir}'...")
    copied_stems, total_boxes, box_counts, removed_boxes = copy_filtered_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        selected_classes=selected,
        keep_only_selected=keep_only_selected,
        class_remap=class_remap,
    )

    # Simpan file konfigurasi classes.yaml khusus sub_dataset
    classes_yaml_path = output_dir / "classes.yaml"
    sorted_selected = sorted(selected)
    with open(classes_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# Sub-dataset classes definition ({len(sorted_selected)} classes)\n")
        f.write(f"nc: {len(sorted_selected)}\n\n")
        f.write("names:\n")
        for i, c in enumerate(sorted_selected):
            f.write(f'  {i}: "{c}"\n')

    # Ringkasan
    print("\n" + "=" * 60)
    print("  RINGKASAN HASIL FILTER")
    print("=" * 60)
    print(f"  Total gambar disalin           : {len(copied_stems)}")
    print(f"  Total bounding boxes terpilih  : {total_boxes}")
    print(f"  Bounding boxes luar dihilangkan: {removed_boxes}")
    print(f"  Output folder                  : {output_dir.resolve()}")
    print(f"  Config kelas                   : {classes_yaml_path.resolve()}")
    print(f"\n  Distribusi per kelas:")
    for cls_name in sorted(box_counts.keys()):
        print(f"    {cls_name:30s}: {box_counts[cls_name]:5d} boxes")
    print("=" * 60)
    print("  SELESAI! Langkah selanjutnya untuk melatih:")
    print(f"  1. Konversi ke format YOLO:")
    print(f"     py prepare_data.py --raw_dir {output_dir} --output_dir data/yolo_sub --classes_config {output_dir}/classes.yaml")
    print(f"  2. Jalankan training:")
    print(f"     py train.py --data data/yolo_sub/data.yaml --epochs 15 --batch 4 --imgsz 416 --device cpu")
    print("=" * 60)


if __name__ == "__main__":
    main()
