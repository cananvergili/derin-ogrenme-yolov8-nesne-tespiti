import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
import yaml

def main():
    random.seed(42)
    base_dir = Path(__file__).resolve().parent
    raw_zip = base_dir / "coco128.zip"
    extract_dir = base_dir / "raw_coco"
    yolo_dir = base_dir / "datasets" / "coco_split"

    # 1. Resmi COCO128 veri setini otomatik indir (~6.7 MB)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    if not raw_zip.exists():
        print("[1/3] COCO veri seti indiriliyor (~6.7 MB)...")
        urllib.request.urlretrieve(url, raw_zip)

    if not extract_dir.exists():
        print("[2/3] İndirilen arşiv açılıyor...")
        with zipfile.ZipFile(raw_zip, "r") as zf:
            zf.extractall(extract_dir)

    src_img_dir = extract_dir / "coco128" / "images" / "train2017"
    src_lbl_dir = extract_dir / "coco128" / "labels" / "train2017"

    # Etiketi olan tüm görüntüleri bul
    valid_pairs = []
    for img_path in sorted(src_img_dir.glob("*.jpg")):
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            valid_pairs.append((img_path, lbl_path))

    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)

    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train : n_train + n_val],
        "test": valid_pairs[n_train + n_val :],
    }

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    print("[3/3] Eğitim (Train), Doğrulama (Val) ve Test kümeleri ayrılıyor...")
    for split_name, pairs in splits.items():
        img_out = yolo_dir / "images" / split_name
        lbl_out = yolo_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, img_out / img_p.name)
            shutil.copy2(lbl_p, lbl_out / lbl_p.name)

    # COCO 80 sınıf isimleri
    coco_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
        6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
        11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
        16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
        27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
        32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
        36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
        40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
        62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
        67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
        72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
        77: "teddy bear", 78: "hair drier", 79: "toothbrush"
    }

    dataset_yaml = {
        "path": yolo_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 80,
        "names": coco_names,
    }

    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n✅ VERİ SETİ HAZIRLANDI!")
    print(f"   - Eğitim (Train)   : {len(splits['train'])} görüntü")
    print(f"   - Doğrulama (Val)  : {len(splits['val'])} görüntü")
    print(f"   - Test             : {len(splits['test'])} görüntü")
    print(f"   - Yapılandırma     : {yaml_path}")

if __name__ == "__main__":
    main()