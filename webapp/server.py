"""
Skin Lesion Screening API Server
================================
FastAPI backend serving MobileNetV2 (fast screening) and EfficientNet (deep analysis with TTA).
Includes image quality validation, proper preprocessing, and performance metrics.
"""

import os
import io
import time
import logging
from pathlib import Path
from enum import Enum
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Suppress TF warnings before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-screening-api")

# --- Constants ---
IMG_SIZE = 224
MODEL_DIR = Path(__file__).resolve().parent.parent  # Points to Projektmunka_2/
CONFIDENCE_THRESHOLD_LOW = 0.35   # Below this = benign with high confidence
CONFIDENCE_THRESHOLD_HIGH = 0.65  # Above this = suspicious with high confidence
# Between 0.35 and 0.65 = uncertain/low confidence zone

# Image quality thresholds
BLUR_THRESHOLD = 80.0       # Laplacian variance below this = blurry
BRIGHTNESS_LOW = 40         # Mean brightness below this = too dark
BRIGHTNESS_HIGH = 220       # Mean brightness above this = too bright
MIN_IMAGE_SIZE = 100        # Minimum dimension in pixels

# --- Model Loading ---
def load_models():
    """Load both Keras models at startup."""
    models = {}

    mobilenet_path = MODEL_DIR / "mobilenet_skin_model.keras"
    efficientnet_path = MODEL_DIR / "efficientnet_skin_model.keras"

    try:
        logger.info(f"Loading MobileNetV2 from {mobilenet_path}...")
        models["mobilenet"] = tf.keras.models.load_model(str(mobilenet_path))
        logger.info("MobileNetV2 loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load MobileNetV2: {e}")

    try:
        logger.info(f"Loading EfficientNet from {efficientnet_path}...")
        models["efficientnet"] = tf.keras.models.load_model(str(efficientnet_path))
        logger.info("EfficientNet loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load EfficientNet: {e}")

    return models


# --- Image Quality Assessment ---
def assess_image_quality(img_array: np.ndarray) -> dict:
    """
    Assess image quality for suitability in skin lesion analysis.
    Returns quality metrics and warnings.
    """
    warnings = []
    quality_score = 100  # Start at 100, deduct for issues

    h, w = img_array.shape[:2]

    # 1. Check minimum size
    if min(h, w) < MIN_IMAGE_SIZE:
        warnings.append(f"Image too small ({w}x{h}px). Minimum {MIN_IMAGE_SIZE}px recommended.")
        quality_score -= 40

    # 2. Check blur (Laplacian variance)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        severity = "very blurry" if laplacian_var < BLUR_THRESHOLD / 2 else "slightly blurry"
        warnings.append(f"Image is {severity} (sharpness: {laplacian_var:.1f}). Please retake with better focus.")
        quality_score -= 30 if laplacian_var < BLUR_THRESHOLD / 2 else 15

    # 3. Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < BRIGHTNESS_LOW:
        warnings.append(f"Image is too dark (brightness: {mean_brightness:.0f}). Use better lighting.")
        quality_score -= 20
    elif mean_brightness > BRIGHTNESS_HIGH:
        warnings.append(f"Image is too bright/overexposed (brightness: {mean_brightness:.0f}). Reduce lighting.")
        quality_score -= 20

    # 4. Check contrast
    std_brightness = np.std(gray)
    if std_brightness < 20:
        warnings.append("Image has very low contrast. The lesion may not be distinguishable.")
        quality_score -= 15

    # 5. Check if image is mostly one color (likely not a skin lesion photo)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation_mean = np.mean(hsv[:, :, 1])
    if saturation_mean < 10:
        warnings.append("Image appears to be grayscale or has very low color saturation.")
        quality_score -= 10

    quality_score = max(0, quality_score)
    is_acceptable = quality_score >= 40

    return {
        "quality_score": quality_score,
        "is_acceptable": is_acceptable,
        "sharpness": round(laplacian_var, 1),
        "brightness": round(mean_brightness, 1),
        "contrast": round(std_brightness, 1),
        "warnings": warnings,
    }


# --- Image Preprocessing ---
def square_crop_center(img: Image.Image) -> Image.Image:
    """Crop the image to a square from the center."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def apply_dull_razor(img: Image.Image) -> tuple[Image.Image, dict]:
    """
    DullRazor digital hair removal algorithm (Lee et al., 1997).
    
    Removes hair occlusions that can distort lesion borders and cause false positives:
    1. Grayscale conversion
    2. Morphological Black-Hat transform with a 7x7 rectangular structuring element
    3. Binary thresholding to generate hair mask
    4. Telea inpainting to reconstruct occluded skin patterns
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Structuring element optimized for 224x224 input
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Isolate pixels noticeably darker than the surrounding skin/lesion
    _, mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    
    hair_pixels = int(cv2.countNonZero(mask))
    max_hair_pixels = int(0.20 * img_np.shape[0] * img_np.shape[1])
    
    # Only inpaint if hair is actually detected (>30 px), but avoid masking large dark lesions (<20% of image)
    if 30 < hair_pixels < max_hair_pixels:
        inpainted = cv2.inpaint(img_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted), {"hair_detected": True, "hair_pixels": hair_pixels}
    
    return img, {"hair_detected": False, "hair_pixels": hair_pixels}


def preprocess_for_model(img: Image.Image, model_name: str, enable_hair_removal: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Square crop from center (if not square)
    2. Resize to 224x224 (Lanczos-3)
    3. DullRazor digital hair removal (if enabled)
    4. Apply model-specific normalization
    """
    # Square crop
    if img.size[0] != img.size[1]:
        img = square_crop_center(img)
    # Resize to model input size
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Digital hair removal
    if enable_hair_removal:
        img, _ = apply_dull_razor(img)

    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # Apply model-specific preprocessing
    # Mindkét modell belső Keras rétegeiben integrálva van a normalizálás:
    # - MobileNetV2: TrueDivide (/127.5) + Subtract (-1.0)
    # - EfficientNetB0: beépített Rescaling réteg
    # Így mindkét modell a standard [0, 255] RGB float32 bemenetet fogadja közvetlenül.
    return img_batch


# --- TTA (Test-Time Augmentation) ---
def predict_with_tta(img_batch: np.ndarray, model, steps: int = 5) -> np.ndarray:
    """
    Test-Time Augmentation: run multiple predictions and average.
    Applies small random augmentations for more robust predictions.
    """
    preds = []
    for i in range(steps):
        if i == 0:
            # Original image (no augmentation)
            augmented = img_batch
        else:
            augmented = img_batch.copy()
            # Apply random augmentations
            if i % 2 == 0:
                augmented = np.flip(augmented, axis=2)  # Horizontal flip
            if i % 3 == 0:
                augmented = np.flip(augmented, axis=1)  # Vertical flip

        p = model.predict(augmented, verbose=0)
        preds.append(p)

    return np.mean(preds, axis=0)


# --- Prediction Result Formatting ---
def format_prediction(score: float, model_name: str, processing_time: float) -> dict:
    """Format the raw model score into a structured prediction result."""
    is_suspicious = score >= 0.5
    confidence = score if is_suspicious else (1.0 - score)
    confidence_pct = round(confidence * 100, 1)

    # Determine risk level and confidence category
    if score < CONFIDENCE_THRESHOLD_LOW:
        risk_level = "low"
        label = "Benign"
        confidence_label = "High Confidence"
    elif score > CONFIDENCE_THRESHOLD_HIGH:
        risk_level = "high"
        label = "Suspicious (Potential Malignancy)"
        confidence_label = "High Confidence"
    else:
        risk_level = "uncertain"
        label = "Uncertain — Further examination recommended"
        confidence_label = "Low Confidence"
        confidence_pct = round((1.0 - abs(score - 0.5) * 2) * 100, 1)  # How uncertain

    return {
        "model": model_name,
        "raw_score": round(score, 4),
        "label": label,
        "risk_level": risk_level,
        "confidence_percent": confidence_pct,
        "confidence_label": confidence_label,
        "is_suspicious": is_suspicious,
        "processing_time_ms": round(processing_time * 1000, 1),
    }


# --- FastAPI App ---
app = FastAPI(
    title="Skin Lesion Screening API",
    description="AI-powered skin lesion screening using MobileNetV2 and EfficientNet",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}


@app.on_event("startup")
async def startup_event():
    global models
    models = load_models()
    logger.info(f"Models loaded: {list(models.keys())}")


# Serve static files (PWA frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the PWA frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/manifest.json")
async def manifest():
    """Serve the PWA manifest."""
    return FileResponse(str(STATIC_DIR / "manifest.json"))


@app.get("/sw.js")
async def service_worker():
    """Serve the service worker."""
    return FileResponse(str(STATIC_DIR / "sw.js"), media_type="application/javascript")


@app.get("/api/health")
async def health_check():
    """Check API and model status."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "mobilenet_ready": "mobilenet" in models,
        "efficientnet_ready": "efficientnet" in models,
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Query(default="both", description="Model to use: mobilenet, efficientnet, or both"),
    tta_steps: int = Query(default=5, ge=1, le=10, description="TTA steps for EfficientNet"),
):
    """
    Analyze a skin lesion image.

    Pipeline: validate → square crop → resize 224×224 → preprocess → predict → return results
    """
    total_start = time.time()

    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are accepted.")

    # 2. Read and decode image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    # 3. Quality assessment
    img_array_raw = np.array(img)
    quality = assess_image_quality(img_array_raw)

    if not quality["is_acceptable"]:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Image quality too low for reliable analysis.",
                "quality": quality,
                "predictions": None,
            },
        )

    # 4. Digital hair removal check
    test_cropped = square_crop_center(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    _, hair_info = apply_dull_razor(test_cropped)

    results = {
        "quality": quality,
        "predictions": {},
        "image_info": {
            "original_size": f"{img.size[0]}x{img.size[1]}",
            "processed_size": f"{IMG_SIZE}x{IMG_SIZE}",
            "hair_removal": hair_info,
        },
    }

    # MobileNet prediction
    if model in ("mobilenet", "both") and "mobilenet" in models:
        start = time.time()
        mn_input = preprocess_for_model(img, "mobilenet")
        mn_score = models["mobilenet"].predict(mn_input, verbose=0)[0][0]
        mn_time = time.time() - start
        results["predictions"]["mobilenet"] = format_prediction(float(mn_score), "MobileNetV2 (Fast Screening)", mn_time)

    # EfficientNet prediction (with TTA)
    if model in ("efficientnet", "both") and "efficientnet" in models:
        start = time.time()
        en_input = preprocess_for_model(img, "efficientnet")
        en_score = predict_with_tta(en_input, models["efficientnet"], steps=tta_steps)[0][0]
        en_time = time.time() - start
        results["predictions"]["efficientnet"] = format_prediction(float(en_score), "EfficientNet (Deep Analysis + TTA)", en_time)

    # Total timing
    results["total_processing_time_ms"] = round((time.time() - total_start) * 1000, 1)

    # Overall recommendation (when both models are used)
    if "mobilenet" in results["predictions"] and "efficientnet" in results["predictions"]:
        mn_risk = results["predictions"]["mobilenet"]["risk_level"]
        en_risk = results["predictions"]["efficientnet"]["risk_level"]

        if en_risk == "high" or mn_risk == "high":
            results["overall_recommendation"] = "Consult a dermatologist. One or both models indicate elevated risk."
        elif en_risk == "uncertain" or mn_risk == "uncertain":
            results["overall_recommendation"] = "Results are uncertain. Consider retaking the image or consulting a professional."
        else:
            results["overall_recommendation"] = "Low risk detected. Continue regular self-monitoring."

    return results


@app.post("/api/quality-check")
async def quality_check(file: UploadFile = File(...)):
    """Check image quality without running model prediction."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    img_array = np.array(img)
    quality = assess_image_quality(img_array)

    return {
        "quality": quality,
        "image_info": {
            "size": f"{img.size[0]}x{img.size[1]}",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
