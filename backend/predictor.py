"""
Dental DSS - Prediction Engine (Predictor)
=======================================
Real mode: classifies the uploaded panoramic radiograph using the
PatchDentalNet CNN trio (seeds 42/123/456) + the 12_1002.py Strategy D
SVM super-ensemble (see cnn_arch.py and svm_pipeline.py for architecture
and pipeline details).

Mock mode: produces a fake-but-plausible result from image statistics,
used to test the UI when model files are unavailable.

NOTE: model weights and the Python venv are not stored in this repository
(multi-GB, excluded via .gitignore). MODELS_DIR below resolves to a
`kds/backend/models` directory expected next to this project's parent
folder - see README.md for how to set this up.
"""

import logging
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageStat

log = logging.getLogger("predictor")

MODE = "real"  # "mock" | "real"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Reuse a sibling app's models/venv instead of duplicating ~1GB of weights.
SHARED_BACKEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "kds", "backend"))
MODELS_DIR = os.path.join(SHARED_BACKEND_DIR, "models")
FEATURES_DIR = os.path.join(MODELS_DIR, "features")
SVM_PIPELINE_PATH = os.path.join(MODELS_DIR, "svm_pipeline.joblib")
SEEDS = [42, 123, 456]

CLASSES = ["No Lesion", "Radiolucent", "Radiopaque", "Mixed"]

# ── Input validation (out-of-distribution guard) ──────────────────────────
# The classifier always assigns one of the 4 classes, so non-panoramic inputs
# (photos, screenshots, charts) must be rejected before a result is shown.
# Thresholds were calibrated on the training data and on external panoramics:
#   - chroma: training panoramics are grayscale (mean chroma <= 0.12 / 255)
#   - aspect: training panoramics span 1.51-3.20 (width / height)
#   - distance: Mahalanobis distance of the 1557-dim feature in the PCA85
#     space; training/test panoramics ~9-25, external panoramics <= 38,
#     non-panoramic images that pass the first two checks >= 76.
MAX_MEAN_CHROMA = 3.0
ASPECT_RANGE = (1.4, 3.5)
MAX_FEATURE_DISTANCE = 50.0


INVALID_IMAGE_MESSAGE = "Invalid image. Please upload a panoramic radiograph."


class InvalidImageError(ValueError):
    """Raised when the uploaded image is not a dental panoramic radiograph.

    The user always sees INVALID_IMAGE_MESSAGE; the specific reason is logged.
    """

    def __init__(self, reason: str):
        log.info(f"Image rejected: {reason}")
        super().__init__(INVALID_IMAGE_MESSAGE)


def _check_image_properties(image: Image.Image):
    w, h = image.size
    aspect = w / h
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        raise InvalidImageError(f"aspect ratio {aspect:.2f} outside {ASPECT_RANGE}")
    small = image.convert("RGB").resize((256, max(1, int(256 / aspect))))
    a = np.asarray(small, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    chroma = float(((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3).mean())
    if chroma > MAX_MEAN_CHROMA:
        raise InvalidImageError(f"color image (mean chroma {chroma:.2f} > {MAX_MEAN_CHROMA})")

TRIAGE_RULES = {
    "No Lesion": {
        "triage": "Low",
        "triage_level": "good",
        "findings": [
            "No significant lesion pattern detected.",
            "Image contrast and edge structures appear generally stable.",
        ],
        "suggestion": "Routine follow-up alongside clinical findings is recommended.",
    },
    "Radiolucent": {
        "triage": "Moderate-High",
        "triage_level": "serious",
        "findings": [
            "Suspected radiolucent area (may be consistent with cystic/resorptive processes).",
            "A different projection or additional imaging may be needed to delineate lesion borders.",
        ],
        "suggestion": "Confirm with periapical imaging and clinical examination; consider advanced imaging (CBCT) if warranted.",
    },
    "Radiopaque": {
        "triage": "Moderate",
        "triage_level": "warning",
        "findings": [
            "Suspected radiopaque focus (may be related to a sclerotic area/retained root, etc.).",
            "Additional evaluation may help localize the finding given the dense structure.",
        ],
        "suggestion": "Regional evaluation alongside clinical history; targeted imaging is recommended if suspicion persists.",
    },
    "Mixed": {
        "triage": "Moderate-High",
        "triage_level": "serious",
        "findings": [
            "Suspected mixed-density pattern (heterogeneous structure).",
            "Clinical and radiological correlation is important for differential diagnosis.",
        ],
        "suggestion": "Targeted radiographic evaluation + clinical examination; consultation is recommended if needed.",
    },
}

# For the "Detailed Guidance" screen - an expanded version of the short
# "suggestion" sentence. General, textbook-level information; not a
# definitive diagnostic or treatment recommendation.
DETAILED_GUIDANCE = {
    "No Lesion": {
        "overview": "The model did not detect a distinct radiolucent, radiopaque, or mixed-density "
                    "lesion pattern in this image. This is consistent with a radiographically "
                    "unremarkable region, but panoramic radiographs have known limitations for "
                    "detecting small or early-stage lesions, especially in areas with anatomical "
                    "superimposition.",
        "next_steps": [
            "Correlate with the patient's clinical presentation and complaint.",
            "Continue routine radiographic follow-up per standard recall intervals.",
            "If clinical symptoms persist despite a clear radiograph, consider a different projection "
            "or CBCT.",
        ],
        "differential_considerations": [
            "Early-stage lesions not yet radiographically visible",
            "Non-odontogenic causes of any reported symptoms",
        ],
        "escalation_note": "If the patient reports pain, swelling, or paresthesia despite this "
                            "result, do not rely on the radiograph alone — a clinical exam and "
                            "possibly further imaging is warranted.",
    },
    "Radiolucent": {
        "overview": "A radiolucent (darker, lower-density) area was detected. Radiolucencies are "
                    "commonly associated with cystic, inflammatory (e.g., periapical), or resorptive "
                    "processes, but can also represent normal anatomical structures (e.g., mental "
                    "foramen, incisive canal) depending on location.",
        "next_steps": [
            "Obtain a periapical radiograph of the involved region for higher resolution.",
            "Perform vitality testing on adjacent teeth if a periapical origin is suspected.",
            "Consider CBCT if the lesion's extent or relationship to adjacent structures is unclear.",
            "Review patient history for prior endodontic treatment, trauma, or infection.",
        ],
        "differential_considerations": [
            "Periapical granuloma / cyst / abscess",
            "Odontogenic keratocyst or other odontogenic cyst",
            "Normal anatomical radiolucency (location-dependent)",
        ],
        "escalation_note": "Rapidly enlarging radiolucencies, cortical perforation, or associated "
                            "paresthesia warrant prompt specialist referral (oral surgery / oral "
                            "pathology).",
    },
    "Radiopaque": {
        "overview": "A radiopaque (brighter, denser) focus was detected. This pattern is often "
                    "associated with sclerotic bone reactions, condensing osteitis, retained root "
                    "fragments, odontomas, or other calcified structures.",
        "next_steps": [
            "Correlate location with adjacent teeth (e.g., a non-vital or heavily restored tooth "
            "nearby may suggest condensing osteitis).",
            "Obtain a periapical or occlusal radiograph for better localization.",
            "Assess whether the finding is stable over time using prior radiographs if available.",
        ],
        "differential_considerations": [
            "Condensing osteitis / sclerotic bone",
            "Retained root fragment or foreign body",
            "Odontoma or other benign calcified lesion",
        ],
        "escalation_note": "Radiopacities associated with pain, swelling, or rapid growth should be "
                            "evaluated further rather than assumed benign.",
    },
    "Mixed": {
        "overview": "A mixed-density (heterogeneous) pattern combining both radiolucent and "
                    "radiopaque components was detected. Mixed lesions often represent a maturation "
                    "stage of certain odontogenic tumors or fibro-osseous lesions and generally "
                    "warrant closer follow-up than a purely radiolucent or radiopaque finding.",
        "next_steps": [
            "Obtain targeted imaging (periapical and/or CBCT) to characterize the internal structure.",
            "Refer to oral & maxillofacial radiology or oral pathology if the lesion is new or "
            "growing.",
            "Document the lesion's size and borders for future comparison.",
        ],
        "differential_considerations": [
            "Fibro-osseous lesion (e.g., cemento-osseous dysplasia)",
            "Calcifying odontogenic cyst / tumor",
            "Complex odontoma at a mixed maturation stage",
        ],
        "escalation_note": "Mixed-density lesions with ill-defined borders, cortical expansion, or "
                            "rapid change should be prioritized for specialist evaluation.",
    },
}


def _confidence_note(confidence: float) -> str:
    pct = confidence * 100
    if confidence >= 0.80:
        return (f"The model's confidence for this prediction is high ({pct:.1f}%), but this does not "
                f"replace clinical and radiographic correlation.")
    if confidence >= 0.55:
        return (f"The model's confidence for this prediction is moderate ({pct:.1f}%). Consider this "
                f"alongside clinical findings rather than in isolation.")
    return (f"The model's confidence for this prediction is relatively low ({pct:.1f}%), meaning the "
            f"probabilities were closer between classes (see Class Probabilities above). Additional "
            f"clinical or radiographic correlation is especially recommended here.")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# 2x8 patch grid: row 0 = upper jaw, row 1 = lower jaw.
# Columns are grouped into 4 zones - image-relative position is used instead
# of clinical left/right, to avoid assuming radiographic orientation.
_JAW_LABELS = ["Upper Jaw", "Lower Jaw"]
_ZONE_LABELS = ["Left End", "Left-Mid", "Right-Mid", "Right End"]
_PATCH_LABELS = [f"{_JAW_LABELS[i // 8]} · Column {i % 8 + 1}" for i in range(16)]


def _region_pixel_stats(image, row: int, col: int) -> dict:
    """Computes pixel statistics for the ORIGINAL image region corresponding to
    the given 2x8 grid cell. This is an approximate match (the attention grid
    is computed on the CLAHE+USM-processed image, this uses the unprocessed
    original) but is reliable for the brightness/contrast DIRECTION.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    h, w = gray.shape
    ph, pw = h // 2, w // 8
    y0 = row * ph
    y1 = (row + 1) * ph if row < 1 else h
    x0 = col * pw
    x1 = (col + 1) * pw if col < 7 else w
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        patch = gray
    return {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "img_mean": float(gray.mean()),
        "img_std": float(gray.std()),
    }


def _build_focus_reason(pred_class: str, stats: dict) -> str:
    """Generates a footnote grounded in real pixel statistics, giving a
    directional explanation when it is consistent with the predicted class.
    Makes no causal claim (the model's decision rests on learned features);
    it only reports an observable correlation.
    """
    mean, std = stats["mean"], stats["std"]
    img_mean, img_std = stats["img_mean"], stats["img_std"]
    mean_diff_pct = abs(mean - img_mean) / (img_mean + 1e-9) * 100
    std_diff_pct = abs(std - img_std) / (img_std + 1e-9) * 100

    if pred_class == "Radiolucent" and mean < img_mean and mean_diff_pct > 3:
        return (f"The average pixel brightness in this region ({mean:.0f}/255) is about "
                f"{mean_diff_pct:.0f}% lower than the image overall (~{img_mean:.0f}/255) — a pattern "
                f"consistent with a low-density (radiolucent) area.")
    if pred_class == "Radiopaque" and mean > img_mean and mean_diff_pct > 3:
        return (f"The average pixel brightness in this region ({mean:.0f}/255) is about "
                f"{mean_diff_pct:.0f}% higher than the image overall (~{img_mean:.0f}/255) — a pattern "
                f"consistent with a dense/radiopaque structure.")
    if pred_class == "Mixed" and std > img_std and std_diff_pct > 3:
        return (f"The local contrast in this region (pixel std={std:.0f}) is about "
                f"{std_diff_pct:.0f}% higher than the image overall (~{img_std:.0f}) — consistent with a "
                f"heterogeneous (mixed) texture combining both dark and light areas.")
    if pred_class == "No Lesion":
        return ("No significant brightness/contrast deviation stood out in this region; the model "
                "evaluated the overall tissue pattern.")
    return ("This is the region with the model's highest attention weight; however, no simple "
            "brightness/contrast difference stands out here — the model's decision may rest on more "
            "complex, learned visual patterns.")


def _describe_focus_region(patch_attention: list, image, pred_class: str) -> dict:
    """Finds the patch with the highest attention weight and converts it into a
    human-readable location plus a pixel-grounded reason."""
    arr = np.array(patch_attention)
    top_idx = int(np.argmax(arr))
    row = top_idx // 8
    col = top_idx % 8
    zone = _ZONE_LABELS[col // 2]
    stats = _region_pixel_stats(image, row, col)
    return {
        "jaw": _JAW_LABELS[row],
        "zone": zone,
        "confidence": float(arr[top_idx]),
        "reason": _build_focus_reason(pred_class, stats),
    }


class DentalPredictor:
    def __init__(self):
        self.mode = MODE
        self.model_version = "mock-v1"
        if self.mode == "real":
            self._load_real_model()

    # ------------------------------------------------------------------
    # MOCK MODE
    # ------------------------------------------------------------------
    def _predict_mock(self, image: Image.Image) -> dict:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        mean = stat.mean[0]
        std = stat.stddev[0]

        seed = int(mean * 997 + std * 131) % (2**32 - 1)
        rng = np.random.default_rng(seed)

        raw = np.array(
            [
                0.20 * std + 0.14 * mean,  # No Lesion
                0.55 * std + 0.08 * mean,  # Radiolucent
                0.45 * std + 0.10 * mean,  # Radiopaque
                0.50 * std + 0.09 * mean,  # Mixed
            ]
        ) + rng.normal(0, 4.0, size=4)

        probs = _softmax(raw)
        return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    # ------------------------------------------------------------------
    # REAL MODE
    # ------------------------------------------------------------------
    def _load_real_model(self):
        import torch

        import svm_pipeline
        from cnn_arch import InferenceConfig, PatchDentalNet

        self.torch = torch
        self.svm_pipeline = svm_pipeline
        self.cfg = InferenceConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")

        self.cnn_models = []
        for seed in SEEDS:
            model_path = os.path.join(MODELS_DIR, f"seed_{seed}", "patchdentalnet.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"This app expects trained models under kds/backend/models - make sure "
                    f"that directory is populated with the seed_{{42,123,456}}/patchdentalnet.pth files."
                )
            model = PatchDentalNet(self.cfg)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model = model.to(self.device)
            model.eval()
            self.cnn_models.append(model)
            log.info(f"  CNN loaded: seed_{seed}")

        if os.path.exists(SVM_PIPELINE_PATH):
            log.info(f"Loading SVM pipeline from disk: {SVM_PIPELINE_PATH}")
            self.svm = svm_pipeline.load_pipeline(SVM_PIPELINE_PATH)
        else:
            log.info("SVM pipeline not found - training from .npy features "
                      "(300 Optuna trials x 2, may take a few minutes)...")
            self.svm = svm_pipeline.build_pipeline(FEATURES_DIR)
            svm_pipeline.save_pipeline(self.svm, SVM_PIPELINE_PATH)
            log.info(f"SVM pipeline saved: {SVM_PIPELINE_PATH}")

        strategy = self.svm.get("overall_best", "D")
        best_metrics = self.svm["metrics"][strategy]
        self.model_version = f"real-v1 (Strategy {strategy}, test F1={best_metrics['f1']*100:.1f}%)"
        log.info(f"Real model ready. {self.model_version}")

        log.info("Preparing SHAP background summary (X_trainval.npy -> PCA85 space, k-means summary)...")
        try:
            self._shap_background = svm_pipeline.build_shap_background(self.svm, FEATURES_DIR)
            log.info("  SHAP background summary ready.")
        except Exception as e:
            log.warning(f"Could not prepare SHAP background summary (SHAP panel will be disabled): {e}")
            self._shap_background = None

    def _feature_distance(self, feature_1557: np.ndarray) -> float:
        """Mahalanobis distance to the training distribution in the PCA85 space."""
        pca = self.svm["pca_85"]
        z = pca.transform(self.svm["scaler"].transform(feature_1557.reshape(1, -1)))[0]
        return float(np.sqrt((z ** 2 / pca.explained_variance_).sum()))

    def _extract_validated_feature(self, image: Image.Image) -> np.ndarray:
        from cnn_arch import extract_feature_vector, image_to_patch_tensor

        _check_image_properties(image)
        patches_tensor = image_to_patch_tensor(image, self.cfg)
        feats = [extract_feature_vector(m, patches_tensor, self.device) for m in self.cnn_models]
        feature_1557 = np.mean(feats, axis=0)  # average of 3 seeds (same as extract_features.py)

        distance = self._feature_distance(feature_1557)
        log.info(f"Feature distance to training distribution: {distance:.1f}")
        if distance > MAX_FEATURE_DISTANCE:
            raise InvalidImageError(
                f"feature distance {distance:.1f} > {MAX_FEATURE_DISTANCE} (out of distribution)"
            )
        return feature_1557

    def _predict_real(self, image: Image.Image):
        feature_1557 = self._extract_validated_feature(image)

        proba = self.svm_pipeline.predict_proba(self.svm, feature_1557)
        probs_dict = {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))}

        # attn_weights (17-dim: [0]=global context patch, [1:17]=2x8 local patch
        # grid) sits at 1536:1553 within the 1557-dim vector (see
        # cnn_arch.extract_feature_vector). For visualization, only the local
        # 16 patches are taken and normalized by the max value.
        attn_17 = feature_1557[1536:1553]
        local_16 = attn_17[1:17]
        patch_attention = (local_16 / (local_16.max() + 1e-9)).tolist()
        pred_class = max(probs_dict, key=probs_dict.get)
        focus_region = _describe_focus_region(patch_attention, image, pred_class)

        shap_groups = None
        if self._shap_background is not None:
            try:
                shap_groups = self.svm_pipeline.explain_shap(
                    self.svm, feature_1557, self._shap_background, CLASSES.index(pred_class)
                )
            except Exception as e:
                log.warning(f"Could not compute SHAP: {e}")

        return probs_dict, patch_attention, focus_region, shap_groups

    # ------------------------------------------------------------------
    def predict(self, image: Image.Image, case_info: dict) -> dict:
        if self.mode == "real":
            probs_dict, patch_attention, focus_region, shap_groups = self._predict_real(image)
        else:
            _check_image_properties(image)
            probs_dict = self._predict_mock(image)
            patch_attention = None
            focus_region = None
            shap_groups = None

        pred_class = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[pred_class]
        rule = TRIAGE_RULES[pred_class]
        guidance = DETAILED_GUIDANCE[pred_class]

        return {
            "timestamp": datetime.now().isoformat(),
            "case": case_info,
            "prediction": pred_class,
            "confidence": confidence,
            "triage": rule["triage"],
            "triage_level": rule["triage_level"],
            "probabilities": probs_dict,
            "findings": rule["findings"],
            "suggestion": rule["suggestion"],
            "patch_attention": patch_attention,  # 16 values (2x8 grid, row-major) or null
            "focus_region": focus_region,  # {"jaw","zone","confidence","reason"} or null
            "shap_feature_groups": shap_groups,  # {group_name: percent} or null (if SHAP failed)
            "detailed_guidance": {
                "overview": guidance["overview"],
                "next_steps": guidance["next_steps"],
                "differential_considerations": guidance["differential_considerations"],
                "escalation_note": guidance["escalation_note"],
                "confidence_note": _confidence_note(confidence),
            },
            "model_mode": self.mode,
            "model_version": self.model_version,
        }

    # ------------------------------------------------------------------
    def build_shap_report(self, image: Image.Image) -> dict:
        """
        For the 'SHAP Analysis' button: explains this single case's feature
        vector with SHAP separately for EACH of the 4 classes (the per-case
        counterpart of feature_group_perclass.png) + returns the patch
        importance ranking. Kept separate from the main predict() flow
        because the 4x SHAP computation (~3-4s) should only run when the
        user opens this report, not on every analysis.
        """
        if self.mode != "real":
            raise RuntimeError("The SHAP report is only available in real model mode.")
        if self._shap_background is None:
            raise RuntimeError("SHAP background summary is not ready (see server logs).")

        feature_1557 = self._extract_validated_feature(image)

        proba = self.svm_pipeline.predict_proba(self.svm, feature_1557)
        probs_dict = {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))}
        pred_class = max(probs_dict, key=probs_dict.get)

        attn_17 = feature_1557[1536:1553]
        local_16 = attn_17[1:17]
        patch_attention = (local_16 / (local_16.max() + 1e-9)).tolist()

        shap_per_class = {}
        for cls in CLASSES:
            shap_per_class[cls] = self.svm_pipeline.explain_shap(
                self.svm, feature_1557, self._shap_background, CLASSES.index(cls)
            )

        return {
            "prediction": pred_class,
            "probabilities": probs_dict,
            "patch_attention": patch_attention,
            "patch_labels": _PATCH_LABELS,
            "shap_per_class": shap_per_class,
        }
