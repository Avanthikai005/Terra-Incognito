# Terra Incognita — API Server for Live Model Inference & Frontend Integration
import base64
import io
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import make_model, NUM_CLASSES
from src.utils import (
    RESULTS_ROOT,
    PATCHES_ROOT,
    CONFIGS_ROOT,
    DEFAULT_REGIONS,
    load_task_sequences,
    load_json,
)

app = FastAPI(
    title="Terra Incognita Model API",
    description="Live inference and continual learning comparison API for satellite building-damage assessment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "undamaged", 1: "damaged", 2: "destroyed"}
INVERSE_CLASS_NAMES = {v: k for k, v in CLASS_NAMES.items()}

# Cache loaded PyTorch models in memory for fast real-time inference
_LOADED_MODELS: dict[str, torch.nn.Module] = {}


def get_inference_transform():
    return T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
    ])


def load_or_get_model(checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """Loads and caches model weights from checkpoint or returns pretrained fallback."""
    key = checkpoint_path or "__base__"
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]

    model = make_model(pretrained=True, device=DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state)
            print(f"[api] Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"[api] Warning: Failed to load checkpoint {checkpoint_path}: {e}")
    model.eval()
    _LOADED_MODELS[key] = model
    return model


def find_checkpoint(sequence: str, regime: str, task_idx: Optional[int] = None) -> Optional[str]:
    """Finds existing checkpoint for sequence + regime."""
    if not os.path.exists(RESULTS_ROOT):
        return None

    # Check specific task
    if task_idx is not None:
        p = os.path.join(RESULTS_ROOT, f"ckpt_{sequence}_{regime}_task{task_idx}.pt")
        if os.path.exists(p):
            return p

    # Fallback to the latest available task checkpoint
    candidates = [
        f for f in os.listdir(RESULTS_ROOT)
        if f.startswith(f"ckpt_{sequence}_{regime}_task") and f.endswith(".pt")
    ]
    if candidates:
        candidates.sort()
        return os.path.join(RESULTS_ROOT, candidates[-1])

    # Check without sequence prefix
    candidates_any = [
        f for f in os.listdir(RESULTS_ROOT)
        if f.startswith(f"ckpt_") and f"_{regime}_" in f and f.endswith(".pt")
    ]
    if candidates_any:
        candidates_any.sort()
        return os.path.join(RESULTS_ROOT, candidates_any[-1])

    return None


@app.get("/api/status")
def get_status():
    """Returns runtime info, active device, available checkpoints and dataset status."""
    checkpoints = []
    if os.path.exists(RESULTS_ROOT):
        checkpoints = [f for f in os.listdir(RESULTS_ROOT) if f.endswith(".pt")]

    result_files = []
    if os.path.exists(RESULTS_ROOT):
        result_files = [f for f in os.listdir(RESULTS_ROOT) if f.endswith("_matrix.json")]

    has_patches = os.path.exists(PATCHES_ROOT) and len(os.listdir(PATCHES_ROOT)) > 0

    return {
        "status": "online",
        "device": DEVICE,
        "torch_version": torch.__version__,
        "model_architecture": "ResNet-18 (3-class: undamaged, damaged, destroyed)",
        "cached_models_in_memory": len(_LOADED_MODELS),
        "checkpoints": checkpoints,
        "result_matrices": result_files,
        "has_patches": has_patches,
        "sequences": load_task_sequences(),
    }


@app.get("/api/results")
def get_results(sequence: str = "similar_domain"):
    """Returns the matrix results for the specified sequence."""
    seqs = load_task_sequences()
    seq_info = seqs.get(sequence, {"regions": DEFAULT_REGIONS, "description": ""})
    regions = seq_info.get("regions", DEFAULT_REGIONS)

    modes = ["naive", "joint", "cl", "cl_adaptive", "baseline"]
    matrices = {}
    avg_forgetting = {}

    for mode in modes:
        # Try both sequence-prefixed and non-prefixed paths
        paths = [
            os.path.join(RESULTS_ROOT, f"{sequence}_{mode}_matrix.json"),
            os.path.join(RESULTS_ROOT, f"{mode}_matrix.json"),
        ]
        found = False
        for p in paths:
            if os.path.exists(p):
                try:
                    data = load_json(p)
                    raw_matrix = data.get("matrix", [])
                    # convert float nan to null for JSON compliance
                    cleaned_matrix = [
                        [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v * 100, 1)
                         for v in row]
                        for row in raw_matrix
                    ]
                    matrices[mode] = cleaned_matrix
                    f = data.get("avg_forgetting")
                    avg_forgetting[mode] = round(f * 100, 2) if f is not None else None
                    found = True
                    break
                except Exception as e:
                    print(f"[api] Error reading {p}: {e}")
        if not found:
            matrices[mode] = None
            avg_forgetting[mode] = None

    return {
        "sequence": sequence,
        "description": seq_info.get("description", ""),
        "regions": regions,
        "matrices": matrices,
        "avg_forgetting": avg_forgetting,
    }


@app.get("/api/samples")
def get_samples(sequence: str = "similar_domain", count_per_region: int = 4):
    """Returns curated test patches with base64 images and ground truth labels."""
    seqs = load_task_sequences()
    regions = seqs.get(sequence, {}).get("regions", DEFAULT_REGIONS)

    samples = []
    sample_id = 0

    # 1. Try loading from data/patches/<region>/test.pt
    for reg in regions:
        test_path = os.path.join(PATCHES_ROOT, reg, "test.pt")
        if os.path.exists(test_path):
            try:
                entries = torch.load(test_path, map_location="cpu", weights_only=False)
                chosen = []
                by_label = {}
                for e in entries:
                    lbl = e.get("label", 0)
                    by_label.setdefault(lbl, []).append(e)

                for lbl in sorted(by_label.keys()):
                    chosen.extend(by_label[lbl][: max(1, count_per_region // len(by_label))])
                if len(chosen) < count_per_region and entries:
                    chosen = entries[:count_per_region]

                for item in chosen:
                    sample_id += 1
                    img_tensor = item["img"]  # [3, H, W] float 0..1
                    arr = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    label_idx = item.get("label", 0)
                    samples.append({
                        "id": f"patch_{sample_id}",
                        "region": reg,
                        "tile": item.get("tile", f"tile_{sample_id}"),
                        "label": label_idx,
                        "label_name": CLASS_NAMES.get(label_idx, "unknown"),
                        "image_b64": f"data:image/png;base64,{b64}",
                    })
            except Exception as e:
                print(f"[api] Error reading test patches for {reg}: {e}")

    # 2. Fallback to frontend qualitative tiles if patch cache is not yet generated
    if not samples:
        qa_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "assets", "qualitative")
        if os.path.exists(qa_dir):
            files = sorted([f for f in os.listdir(qa_dir) if f.endswith(".png")])[:12]
            for i, f in enumerate(files):
                p = os.path.join(qa_dir, f)
                pil_img = Image.open(p).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                lbl = i % 3
                reg = regions[i % len(regions)] if regions else "region-1"
                samples.append({
                    "id": f"qa_{i}",
                    "region": reg,
                    "tile": f"synthetic_{f}",
                    "label": lbl,
                    "label_name": CLASS_NAMES[lbl],
                    "image_b64": f"data:image/png;base64,{b64}",
                })

    return {"samples": samples}


class PredictRequest(BaseModel):
    image_b64: Optional[str] = None
    sample_id: Optional[str] = None
    sequence: str = "similar_domain"
    regime: str = "cl"
    task_idx: Optional[int] = None
    compare_all: bool = False


@torch.no_grad()
def run_single_inference(model: torch.nn.Module, img_tensor: torch.Tensor) -> dict:
    start_t = time.perf_counter()
    x = img_tensor.unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "predicted_class": pred_idx,
        "class_name": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "undamaged": float(probs[0]),
            "damaged": float(probs[1]),
            "destroyed": float(probs[2]),
        },
        "inference_ms": round(elapsed_ms, 2),
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Runs live model inference on uploaded image or sample patch."""
    pil_img = None

    # Option A: decode uploaded base64 image
    if req.image_b64:
        try:
            raw_data = req.image_b64
            if "," in raw_data:
                raw_data = raw_data.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_data)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

    # Option B: resolve from sample_id
    elif req.sample_id:
        all_samples = get_samples(sequence=req.sequence, count_per_region=10).get("samples", [])
        matched = next((s for s in all_samples if s["id"] == req.sample_id), None)
        if matched and matched.get("image_b64"):
            raw_data = matched["image_b64"].split(",", 1)[1]
            image_bytes = base64.b64decode(raw_data)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Must provide either image_b64 or a valid sample_id")

    # Transform image for ResNet-18
    transform = get_inference_transform()
    img_tensor = transform(pil_img)  # [3, 224, 224]

    regimes_to_run = ["naive", "joint", "cl", "cl_adaptive"] if req.compare_all else [req.regime]
    results_by_regime = {}

    for reg in regimes_to_run:
        ckpt = find_checkpoint(req.sequence, reg, req.task_idx)
        model = load_or_get_model(ckpt)
        inf = run_single_inference(model, img_tensor)
        inf["checkpoint"] = os.path.basename(ckpt) if ckpt else "pretrained_fallback"
        results_by_regime[reg] = inf

    primary = results_by_regime.get(req.regime, list(results_by_regime.values())[0])

    response = {
        "regime": req.regime,
        "sequence": req.sequence,
        "task_idx": req.task_idx,
        "prediction": primary,
        "comparisons": results_by_regime if req.compare_all else None,
    }
    return response


@app.post("/api/predict_upload")
async def predict_upload(
    file: UploadFile = File(...),
    regime: str = Form("cl"),
    sequence: str = Form("similar_domain"),
    task_idx: Optional[int] = Form(None),
    compare_all: bool = Form(False),
):
    """Direct multipart file upload endpoint for testing arbitrary satellite images."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")

    transform = get_inference_transform()
    img_tensor = transform(pil_img)

    regimes_to_run = ["naive", "joint", "cl", "cl_adaptive"] if compare_all else [regime]
    results_by_regime = {}

    for reg in regimes_to_run:
        ckpt = find_checkpoint(sequence, reg, task_idx)
        model = load_or_get_model(ckpt)
        inf = run_single_inference(model, img_tensor)
        inf["checkpoint"] = os.path.basename(ckpt) if ckpt else "pretrained_fallback"
        results_by_regime[reg] = inf

    primary = results_by_regime.get(regime, list(results_by_regime.values())[0])

    return {
        "regime": regime,
        "sequence": sequence,
        "task_idx": task_idx,
        "prediction": primary,
        "comparisons": results_by_regime if compare_all else None,
    }


if __name__ == "__main__":
    import uvicorn
    print("[api] Starting Terra Incognita Model API Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
