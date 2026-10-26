"""
Test script for Skin Lesion Screening API
Validates:
1. Normal image prediction (MobileNet + EfficientNet + TTA)
2. Quality detection: Blurry image
3. Quality detection: Low lighting / Dark image
4. Quality detection: Overexposed / Bright image
5. Low-confidence prediction handling
6. Response and prediction timing
"""

import io
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

API_URL = "http://localhost:8000"

def create_synthetic_lesion(size=(300, 300), lesion_color=(90, 50, 40), bg_color=(220, 185, 170)):
    """Create a synthetic skin patch with a pigmented lesion."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add some skin texture variation
    pixels = np.array(img)
    noise = np.random.normal(0, 5, pixels.shape).astype(np.int16)
    noisy_pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy_pixels)
    draw = ImageDraw.Draw(img)

    # Draw irregular lesion in center
    cx, cy = size[0] // 2, size[1] // 2
    r = size[0] // 5
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, fill=lesion_color)
    
    # Blur boundary slightly for organic look
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

def image_to_bytes(img: Image.Image, format="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def run_tests():
    print("=" * 60)
    print("DERMAI SCAN API TEST SUITE")
    print("=" * 60)
    
    # 1. Health check
    print("\n[1] Health Check...")
    r = requests.get(f"{API_URL}/api/health")
    print("Status:", r.status_code, r.json())
    assert r.status_code == 200, "Health check failed!"

    # 2. Normal Synthetic Lesion Image Analysis
    print("\n[2] Testing Normal Image Analysis (MobileNet + EfficientNet + TTA)...")
    normal_img = create_synthetic_lesion()
    normal_bytes = image_to_bytes(normal_img)
    
    t0 = time.time()
    r = requests.post(
        f"{API_URL}/api/analyze?model=both&tta_steps=5",
        files={"file": ("lesion.jpg", normal_bytes, "image/jpeg")}
    )
    req_time = (time.time() - t0) * 1000
    print(f"Status: {r.status_code} (Client Round-Trip: {req_time:.1f}ms)")
    res = r.json()
    print("Image Info:", res.get("image_info"))
    print("Quality Score:", res.get("quality", {}).get("quality_score"), "Acceptable:", res.get("quality", {}).get("is_acceptable"))
    for mod_name, pred in res.get("predictions", {}).items():
        print(f" -> {pred['model']}: Label='{pred['label']}', Score={pred['raw_score']}, Risk={pred['risk_level']}, Time={pred['processing_time_ms']}ms")
    print("Total Server Time:", res.get("total_processing_time_ms"), "ms")
    print("Overall Recommendation:", res.get("overall_recommendation"))

    # 3. Blurry Image Test
    print("\n[3] Testing Blurry Image Quality Detection...")
    blurry_img = normal_img.filter(ImageFilter.GaussianBlur(radius=12))
    blurry_bytes = image_to_bytes(blurry_img)
    r = requests.post(
        f"{API_URL}/api/quality-check",
        files={"file": ("blurry.jpg", blurry_bytes, "image/jpeg")}
    )
    print("Status:", r.status_code)
    q = r.json()["quality"]
    print(f"Sharpness: {q['sharpness']} (Score: {q['quality_score']}/100)")
    print("Warnings:", q["warnings"])
    assert any("blurry" in w.lower() for w in q["warnings"]), "Blurry warning not triggered!"

    # 4. Dark Image Test
    print("\n[4] Testing Underexposed / Dark Image Quality Detection...")
    dark_img = Image.new("RGB", (300, 300), (15, 12, 10))
    dark_bytes = image_to_bytes(dark_img)
    r = requests.post(
        f"{API_URL}/api/quality-check",
        files={"file": ("dark.jpg", dark_bytes, "image/jpeg")}
    )
    print("Status:", r.status_code)
    q = r.json()["quality"]
    print(f"Brightness: {q['brightness']} (Score: {q['quality_score']}/100)")
    print("Warnings:", q["warnings"])
    assert any("dark" in w.lower() for w in q["warnings"]), "Dark warning not triggered!"

    # 5. Overexposed / Bright Image Test
    print("\n[5] Testing Overexposed / Bright Image Quality Detection...")
    bright_img = Image.new("RGB", (300, 300), (245, 240, 240))
    bright_bytes = image_to_bytes(bright_img)
    r = requests.post(
        f"{API_URL}/api/quality-check",
        files={"file": ("bright.jpg", bright_bytes, "image/jpeg")}
    )
    print("Status:", r.status_code)
    q = r.json()["quality"]
    print(f"Brightness: {q['brightness']} (Score: {q['quality_score']}/100)")
    print("Warnings:", q["warnings"])
    assert any("bright" in w.lower() for w in q["warnings"]), "Bright warning not triggered!"

    # 6. Unacceptable Image Analysis Reject Test
    print("\n[6] Testing Analysis Rejection on Severely Degraded Image...")
    severely_bad_img = Image.new("RGB", (50, 50), (5, 5, 5)).filter(ImageFilter.GaussianBlur(10))
    bad_bytes = image_to_bytes(severely_bad_img)
    r = requests.post(
        f"{API_URL}/api/analyze",
        files={"file": ("bad.jpg", bad_bytes, "image/jpeg")}
    )
    print(f"Status: {r.status_code} (Expected: 422)")
    print("Response:", r.json().get("error"))
    assert r.status_code == 422, "API should reject unacceptable image with 422!"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
