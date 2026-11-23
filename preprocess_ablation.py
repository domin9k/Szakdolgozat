#!/usr/bin/env python3.11
"""
DermAI Scan – Pre-processing Ablation Study Script
====================================================
Összehasonlítja a különböző előfeldolgozási paramétereket mindkét modellen,
és egy átlátható eredménytáblázatot generál.

Használat:
    # Szintetikus képekkel (nem kell külön tesztkép):
    python3.11 preprocess_ablation.py

    # Saját tesztképpel:
    python3.11 preprocess_ablation.py --image /path/to/lesion.jpg

    # Csak egy modellt tesztel:
    python3.11 preprocess_ablation.py --model mobilenet

    # CSV export:
    python3.11 preprocess_ablation.py --csv results.csv
"""

import os
import sys
import io
import time
import glob
import argparse
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet



# 1. KONFIGURÁLHATÓ PRE-PROCESSING VARIÁNSOK
IMG_SIZE = 224
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Interpolációs szűrők
INTERPOLATION_METHODS = {
    "NEAREST":  Image.Resampling.NEAREST,
    "BILINEAR": Image.Resampling.BILINEAR,
    "BICUBIC":  Image.Resampling.BICUBIC,
    "LANCZOS":  Image.Resampling.LANCZOS,   # ← a DermAI Scan választása
}

# Normalizációs stratégiák
def norm_mobilenet(batch):
    """MobileNetV2 standard: [-1, 1]"""
    return preprocess_mobilenet(batch.copy())

def norm_efficientnet(batch):
    """EfficientNet standard: [0, 1] (internal /255)"""
    return preprocess_efficientnet(batch.copy())

def norm_raw_255(batch):
    """Nyers 0-255 – nincs normalizáció (hibás baseline)"""
    return batch.copy()

def norm_divide_255(batch):
    """Egyszerű /255 – mindkét modellre egységesen"""
    return batch.copy() / 255.0

def norm_zero_center(batch):
    """Zéró-közepesített: (x / 127.5) - 1.0"""
    return batch.copy() / 127.5 - 1.0

NORMALIZATION_METHODS = {
    "mobilenet_std":    ("MobileNet [-1,1]",    norm_mobilenet),
    "efficientnet_std": ("EfficientNet [0,1]",  norm_efficientnet),
    "raw_255":          ("Nyers [0,255]",       norm_raw_255),
    "divide_255":       ("Osztás /255 [0,1]",   norm_divide_255),
    "zero_center":      ("Zero-center [-1,1]",  norm_zero_center),
}

# Crop stratégiák
def crop_center_square(img: Image.Image) -> Image.Image:
    """Központi négyzetes vágás (DermAI Scan választása)"""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def crop_stretch(img: Image.Image) -> Image.Image:
    """Nincs vágás – a teljes kép közvetlenül méretezve (anamorfikus torzítás)"""
    return img  # a resize majd torzítja

def crop_top_left(img: Image.Image) -> Image.Image:
    """Bal felső négyzetes vágás (nem centralizált)"""
    w, h = img.size
    side = min(w, h)
    return img.crop((0, 0, side, side))

def crop_pad_black(img: Image.Image) -> Image.Image:
    """Fekete padding: a kép négyzetté bővítése fekete szegéllyel"""
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    return padded

CROP_METHODS = {
    "center_square": ("Központi négyzet",   crop_center_square),  # ← DermAI Scan
    "stretch":       ("Nyújtás (Stretch)",  crop_stretch),
    "top_left":      ("Bal felső vágás",    crop_top_left),
    "pad_black":     ("Fekete padding",     crop_pad_black),
}

# TTA variánsok
TTA_STEPS_OPTIONS = [1, 3, 5, 7]  # 1 = nincs TTA (egyetlen futtatás)

# DullRazor szőreltávolítás (Továbbfejlesztett, geometriailag szűrt változat)
def apply_dull_razor(img: Image.Image) -> Image.Image:
    """Intelligens (Advanced) DullRazor szőreltávolítás geometriai szűréssel és pigmentvédelemmel."""
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, raw_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    
    # Geometriai kontúrszűrés: kizárja a körkörös/szabálytalan pigmenteket és léziószéleket
    filtered_mask = np.zeros_like(raw_mask)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-3)
        if aspect_ratio >= 3.0 and max(w, h) >= 15:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    hair_pixels = int(cv2.countNonZero(filtered_mask))
    max_hair_pixels = int(0.20 * img_np.shape[0] * img_np.shape[1])
    if 200 <= hair_pixels < max_hair_pixels:
        inpainted = cv2.inpaint(img_np, filtered_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted)
    return img

HAIR_REMOVAL_METHODS = {
    "dullrazor": ("Advanced DullRazor (Geometriai szűréssel)", apply_dull_razor),
    "none":      ("Nyers (Kikapcsolva)", lambda img: img),
}


# 2. VALÓS TESZTKÉPEK BETÖLTÉSE
def load_real_test_images(max_per_class: int = 3) -> dict[str, Image.Image]:
    """Valós klinikai dermoszkópos tesztképeket tölt be a valos_tesztkepek mappából."""
    images = {}
    base_dir = os.path.join(PROJECT_DIR, "valos_tesztkepek")
    benign_dir = os.path.join(base_dir, "Benign")
    malig_dir = os.path.join(base_dir, "Malignant")

    if os.path.exists(benign_dir):
        b_files = sorted(glob.glob(os.path.join(benign_dir, "*.jpg")))[:max_per_class]
        for f in b_files:
            name = f"benign_{os.path.splitext(os.path.basename(f))[0]}"
            images[name] = Image.open(f).convert("RGB")

    if os.path.exists(malig_dir):
        m_files = sorted(glob.glob(os.path.join(malig_dir, "*.jpg")))[:max_per_class]
        for f in m_files:
            name = f"malignant_{os.path.splitext(os.path.basename(f))[0]}"
            images[name] = Image.open(f).convert("RGB")

    return images


# 3. MODELL BETÖLTÉS ÉS PREDIKCIÓ
def load_models(model_filter: Optional[str] = None):
    """Betölti a Keras modelleket."""
    models = {}
    model_paths = {
        "mobilenet":    os.path.join(PROJECT_DIR, "mobilenet_skin_model.keras"),
        "efficientnet": os.path.join(PROJECT_DIR, "efficientnet_skin_model.keras"),
    }

    for name, path in model_paths.items():
        if model_filter and name != model_filter:
            continue
        if os.path.exists(path):
            print(f"  Betöltés: {name} ({os.path.getsize(path) / 1e6:.1f} MB)...")
            models[name] = tf.keras.models.load_model(path)
        else:
            print(f"  ⚠ Nem található: {path}")

    return models


def predict_with_tta(img_batch, model, steps=5):
    """Test-Time Augmentation: tükrözéses ensemble átlagolás."""
    preds = []
    for i in range(steps):
        if i == 0:
            aug = img_batch
        else:
            aug = img_batch.copy()
            if i % 2 == 0:
                aug = np.flip(aug, axis=2)  # Horizontal flip
            if i % 3 == 0:
                aug = np.flip(aug, axis=1)  # Vertical flip
        preds.append(model.predict(aug, verbose=0))
    return np.mean(preds, axis=0)


def run_single_prediction(img, model, model_name,
                          crop_fn, interp_method, norm_fn,
                          hair_fn=apply_dull_razor,
                          tta_steps=1):
    """Egyetlen predikció a megadott preprocessing-beállításokkal."""
    t0 = time.perf_counter()

    # 1. Crop
    cropped = crop_fn(img.copy())

    # 2. Resize
    resized = cropped.resize((IMG_SIZE, IMG_SIZE), interp_method)

    # 3. DullRazor szőreltávolítás
    cleaned = hair_fn(resized)

    # 4. Array + Batch dim
    arr = np.array(cleaned, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)

    # 5. Normalizáció
    batch = norm_fn(batch)

    # 6. Predikció (TTA vagy egyszerű)
    if tta_steps > 1:
        raw_output = predict_with_tta(batch, model, steps=tta_steps)
    else:
        raw_output = model.predict(batch, verbose=0)

    score = float(raw_output[0][0])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return score, elapsed_ms

# 4. ABLÁCIÓS TESZTEK DEFINÍCIÓJA
@dataclass
class AblationResult:
    test_name: str
    image_name: str
    model_name: str
    crop_method: str
    interpolation: str
    normalization: str
    tta_steps: int
    score: float
    label: str
    confidence: float
    time_ms: float
    hair_removal: str = "DullRazor"
    is_baseline: bool = False


def score_to_label(score):
    if score < 0.35:
        return "Benign", round((1.0 - score) * 100, 1)
    elif score > 0.65:
        return "Suspicious", round(score * 100, 1)
    else:
        return "Uncertain", round((1.0 - abs(score - 0.5) * 2) * 100, 1)


def run_ablation_tests(models, test_images):
    """Futtatja az összes ablációs tesztet."""
    results: List[AblationResult] = []

    # A baseline beállítások (DermAI Scan produkciós konfiguráció)
    BASELINE_CROP = "center_square"
    BASELINE_INTERP = "LANCZOS"

    correct_norms = {
        "mobilenet":    "mobilenet_std",
        "efficientnet": "efficientnet_std",
    }

    total_tests = 0

    for model_name, model in models.items():
        correct_norm_key = correct_norms[model_name]
        _, correct_norm_fn = NORMALIZATION_METHODS[correct_norm_key]
        _, baseline_crop_fn = CROP_METHODS[BASELINE_CROP]
        baseline_interp = INTERPOLATION_METHODS[BASELINE_INTERP]

        for img_name, img in test_images.items():

            # ── A) BASELINE (produkciós konfiguráció) ─────────────────
            score, ms = run_single_prediction(
                img, model, model_name,
                crop_fn=baseline_crop_fn,
                interp_method=baseline_interp,
                norm_fn=correct_norm_fn,
                tta_steps=1
            )
            label, conf = score_to_label(score)
            results.append(AblationResult(
                test_name="BASELINE (Produkciós)",
                image_name=img_name, model_name=model_name,
                crop_method=CROP_METHODS[BASELINE_CROP][0],
                interpolation=BASELINE_INTERP,
                normalization=NORMALIZATION_METHODS[correct_norm_key][0],
                tta_steps=1, score=score, label=label,
                confidence=conf, time_ms=ms, is_baseline=True
            ))
            total_tests += 1

            # ── B) INTERPOLÁCIÓ ABLÁCIÓJA ─────────────────────────────
            for interp_name, interp_val in INTERPOLATION_METHODS.items():
                if interp_name == BASELINE_INTERP:
                    continue
                score, ms = run_single_prediction(
                    img, model, model_name,
                    crop_fn=baseline_crop_fn,
                    interp_method=interp_val,
                    norm_fn=correct_norm_fn,
                    tta_steps=1
                )
                label, conf = score_to_label(score)
                results.append(AblationResult(
                    test_name=f"Interpoláció: {interp_name}",
                    image_name=img_name, model_name=model_name,
                    crop_method=CROP_METHODS[BASELINE_CROP][0],
                    interpolation=interp_name,
                    normalization=NORMALIZATION_METHODS[correct_norm_key][0],
                    tta_steps=1, score=score, label=label,
                    confidence=conf, time_ms=ms
                ))
                total_tests += 1

            # ── C) CROP STRATÉGIA ABLÁCIÓJA ───────────────────────────
            for crop_key, (crop_label, crop_fn) in CROP_METHODS.items():
                if crop_key == BASELINE_CROP:
                    continue
                score, ms = run_single_prediction(
                    img, model, model_name,
                    crop_fn=crop_fn,
                    interp_method=baseline_interp,
                    norm_fn=correct_norm_fn,
                    tta_steps=1
                )
                label, conf = score_to_label(score)
                results.append(AblationResult(
                    test_name=f"Crop: {crop_label}",
                    image_name=img_name, model_name=model_name,
                    crop_method=crop_label,
                    interpolation=BASELINE_INTERP,
                    normalization=NORMALIZATION_METHODS[correct_norm_key][0],
                    tta_steps=1, score=score, label=label,
                    confidence=conf, time_ms=ms
                ))
                total_tests += 1

            # ── D) NORMALIZÁCIÓ ABLÁCIÓJA ─────────────────────────────
            for norm_key, (norm_label, norm_fn) in NORMALIZATION_METHODS.items():
                if norm_key == correct_norm_key:
                    continue
                score, ms = run_single_prediction(
                    img, model, model_name,
                    crop_fn=baseline_crop_fn,
                    interp_method=baseline_interp,
                    norm_fn=norm_fn,
                    tta_steps=1
                )
                label, conf = score_to_label(score)
                results.append(AblationResult(
                    test_name=f"Normalizáció: {norm_label}",
                    image_name=img_name, model_name=model_name,
                    crop_method=CROP_METHODS[BASELINE_CROP][0],
                    interpolation=BASELINE_INTERP,
                    normalization=norm_label,
                    tta_steps=1, score=score, label=label,
                    confidence=conf, time_ms=ms
                ))
                total_tests += 1

            # ── E) TTA LÉPÉSSZÁM ABLÁCIÓJA (csak EfficientNet) ───────
            if model_name == "efficientnet":
                for tta_s in TTA_STEPS_OPTIONS:
                    if tta_s == 1:
                        continue  # baseline-nál már volt
                    score, ms = run_single_prediction(
                        img, model, model_name,
                        crop_fn=baseline_crop_fn,
                        interp_method=baseline_interp,
                        norm_fn=correct_norm_fn,
                        tta_steps=tta_s
                    )
                    label, conf = score_to_label(score)
                    results.append(AblationResult(
                        test_name=f"TTA lépések: {tta_s}",
                        image_name=img_name, model_name=model_name,
                        crop_method=CROP_METHODS[BASELINE_CROP][0],
                        interpolation=BASELINE_INTERP,
                        normalization=NORMALIZATION_METHODS[correct_norm_key][0],
                        tta_steps=tta_s, score=score, label=label,
                        confidence=conf, time_ms=ms
                    ))
                    total_tests += 1

            # ── F) DULLRAZOR SZŐRELTÁVOLÍTÁS ABLÁCIÓJA ──────────────
            score, ms = run_single_prediction(
                img, model, model_name,
                crop_fn=baseline_crop_fn,
                interp_method=baseline_interp,
                norm_fn=correct_norm_fn,
                hair_fn=HAIR_REMOVAL_METHODS["none"][1],
                tta_steps=1
            )
            label, conf = score_to_label(score)
            results.append(AblationResult(
                test_name="Szőreltávolítás: Nyers (Kikapcsolva)",
                image_name=img_name, model_name=model_name,
                crop_method=CROP_METHODS[BASELINE_CROP][0],
                interpolation=BASELINE_INTERP,
                normalization=NORMALIZATION_METHODS[correct_norm_key][0],
                tta_steps=1, score=score, label=label,
                confidence=conf, time_ms=ms,
                hair_removal="Nyers (Kikapcsolva)"
            ))
            total_tests += 1

    print(f"\n  Összesen {total_tests} tesztkonfigurációt futtattunk le.")
    return results

# 5. EREDMÉNYEK MEGJELENÍTÉSE
def print_results_table(results: List[AblationResult]):
    """Terminálban formázott eredménytáblázat."""

    # Csoportosítás modell és tesztkép szerint
    groups = {}
    for r in results:
        key = (r.model_name, r.image_name)
        groups.setdefault(key, []).append(r)

    for (model_name, img_name), group in sorted(groups.items()):
        # Fejléc
        model_display = "MobileNetV2" if model_name == "mobilenet" else "EfficientNetB0"
        print()
        print(f"{'═' * 120}")
        print(f"  MODELL: {model_display}  │  TESZTKÉP: {img_name}")
        print(f"{'═' * 120}")

        # Baseline score megkeresése összehasonlításhoz
        baseline = [r for r in group if r.is_baseline]
        baseline_score = baseline[0].score if baseline else None

        # Táblázat fejléc
        header = (f"  {'Teszt konfiguráció':<35} │ {'Crop':<18} │ {'Interp.':<10} │ "
                  f"{'Normalizáció':<22} │ {'TTA':<4} │ {'Score':>7} │ {'Δ':>7} │ "
                  f"{'Címke':<12} │ {'Konf.':>6} │ {'ms':>7}")
        print(f"{'─' * 120}")
        print(header)
        print(f"{'─' * 120}")

        for r in group:
            # Delta a baseline-hoz képest
            if baseline_score is not None:
                delta = r.score - baseline_score
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "  ---"

            # Jelölés
            marker = " ★" if r.is_baseline else "  "
            test_name = r.test_name

            row = (f"{marker}{test_name:<34} │ {r.crop_method:<18} │ {r.interpolation:<10} │ "
                   f"{r.normalization:<22} │ {r.tta_steps:>4} │ {r.score:>7.4f} │ {delta_str:>7} │ "
                   f"{r.label:<12} │ {r.confidence:>5.1f}% │ {r.time_ms:>6.1f}")
            print(row)

        print(f"{'─' * 120}")


def print_summary_comparison(results: List[AblationResult]):
    """Összefoglaló: ablációs kategóriánként a legnagyobb eltérések."""
    print()
    print(f"{'═' * 100}")
    print(f"  ÖSSZEFOGLALÓ – LEGNAGYOBB HATÁSÚ PREPROCESSING VÁLTOZTATÁSOK")
    print(f"{'═' * 100}")

    categories = {
        "Interpoláció":    [r for r in results if r.test_name.startswith("Interpoláció:")],
        "Crop":            [r for r in results if r.test_name.startswith("Crop:")],
        "Normalizáció":    [r for r in results if r.test_name.startswith("Normalizáció:")],
        "TTA":             [r for r in results if r.test_name.startswith("TTA")],
        "Szőreltávolítás": [r for r in results if r.test_name.startswith("Szőreltávolítás:")],
    }

    baselines = {(r.model_name, r.image_name): r.score
                 for r in results if r.is_baseline}

    for cat_name, cat_results in categories.items():
        if not cat_results:
            continue
        print(f"\n  ── {cat_name} ──")

        # Legnagyobb abszolút eltérés
        max_delta = 0
        max_entry = None
        for r in cat_results:
            bl = baselines.get((r.model_name, r.image_name), 0)
            delta = abs(r.score - bl)
            if delta > max_delta:
                max_delta = delta
                max_entry = r

        if max_entry:
            bl = baselines.get((max_entry.model_name, max_entry.image_name), 0)
            model_d = "MobileNetV2" if max_entry.model_name == "mobilenet" else "EfficientNetB0"
            print(f"    Legnagyobb eltérés: Δ = {max_entry.score - bl:+.4f} "
                  f"({max_entry.test_name}, {model_d}, kép: {max_entry.image_name})")
            print(f"    Baseline score: {bl:.4f} → Módosított: {max_entry.score:.4f} "
                  f"({max_entry.label}, {max_entry.confidence}%)")

    # Végső konklúzió
    wrong_norm = [r for r in results if r.test_name.startswith("Normalizáció:")]
    if wrong_norm:
        avg_delta = np.mean([abs(r.score - baselines.get((r.model_name, r.image_name), 0))
                             for r in wrong_norm])
        print(f"\n  ⚠  Átlagos score-eltérés hibás normalizációval: Δ = {avg_delta:.4f}")
        print(f"     → Ez igazolja a modell-specifikus normalizáció szükségességét.")

    print(f"\n{'═' * 100}")


def export_csv(results: List[AblationResult], filepath: str):
    """CSV export az eredményekről."""
    import csv
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Teszt", "Kép", "Modell", "Crop", "Interpoláció",
            "Normalizáció", "Szőreltávolítás", "TTA_lépések", "Score", "Címke",
            "Konfidencia_%", "Idő_ms", "Baseline"
        ])
        for r in results:
            writer.writerow([
                r.test_name, r.image_name, r.model_name,
                r.crop_method, r.interpolation, r.normalization,
                r.hair_removal,
                r.tta_steps, f"{r.score:.6f}", r.label,
                f"{r.confidence:.1f}", f"{r.time_ms:.1f}",
                "igen" if r.is_baseline else "nem"
            ])
    print(f"\n  CSV exportálva: {filepath}")

# 6. MAIN
def main():
    parser = argparse.ArgumentParser(
        description="DermAI Scan – Pre-processing Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python3.11 preprocess_ablation.py                      # szintetikus képek, mindkét modell
  python3.11 preprocess_ablation.py --image foto.jpg     # saját kép tesztelése
  python3.11 preprocess_ablation.py --model mobilenet    # csak MobileNetV2
  python3.11 preprocess_ablation.py --csv eredmenyek.csv # CSV export
        """
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Saját tesztkép elérési útja (JPEG/PNG). Ha nincs megadva, szintetikus képeket generál.")
    parser.add_argument("--model", type=str, choices=["mobilenet", "efficientnet"],
                        default=None, help="Csak egy modellt tesztel (alapértelmezett: mindkettő).")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV fájl elérési útja az eredmények exportálásához.")

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DermAI Scan – Pre-processing Ablációs Vizsgálat           ║")
    print("║  Összehasonlító elemzés a preprocessing paraméterekről     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Tesztképek ──
    print("\n[1/3] Tesztképek előkészítése...")
    if args.image:
        if not os.path.exists(args.image):
            print(f"  ✗ Nem található: {args.image}")
            sys.exit(1)
        img = Image.open(args.image).convert("RGB")
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        test_images = {img_name: img}
        print(f"  Betöltve: {args.image} ({img.size[0]}x{img.size[1]})")
    else:
        test_images = load_real_test_images()
        print(f"  {len(test_images)} valós klinikai tesztkép betöltve a valos_tesztkepek mappából:")
        for name, img in test_images.items():
            print(f"    • {name}: {img.size[0]}x{img.size[1]}")

    # ── Modellek betöltése ──
    print("\n[2/3] Modellek betöltése...")
    models = load_models(model_filter=args.model)
    if not models:
        print("  ✗ Nincs betölthető modell! Ellenőrizd a .keras fájlokat.")
        sys.exit(1)
    print(f"  ✓ {len(models)} modell kész: {', '.join(models.keys())}")

    # ── Ablációs futtatás ──
    print("\n[3/3] Ablációs tesztek futtatása...")
    results = run_ablation_tests(models, test_images)

    # ── Eredmények ──
    print_results_table(results)
    print_summary_comparison(results)

    if args.csv:
        csv_path = os.path.join(PROJECT_DIR, args.csv)
        export_csv(results, csv_path)

    print("\n  Kész! ✓\n")


if __name__ == "__main__":
    main()
