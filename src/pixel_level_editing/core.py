from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageChops


EditMode = Literal["paste", "recolor"]
OverlapPolicy = Literal["current", "previous"]


@dataclass
class PairResult:
    result: Image.Image
    mask: Image.Image
    heatmap_before: Image.Image
    heatmap_after: Image.Image
    metrics: dict


@dataclass
class RoundSpec:
    image: str
    mask: str | None = None
    mode: EditMode = "paste"
    overlap: OverlapPolicy = "current"
    threshold: float = 24.0
    min_area_ratio: float = 0.0005
    dilate: int = 5
    name: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_to_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop to the target aspect ratio, then resize."""
    image = image.convert("RGB")
    target_w, target_h = size
    source_w, source_h = image.size
    source_ratio = source_w / source_h
    target_ratio = target_w / target_h
    if source_ratio > target_ratio:
        new_w = round(source_h * target_ratio)
        left = (source_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, source_h))
    elif source_ratio < target_ratio:
        new_h = round(source_w / target_ratio)
        top = (source_h - new_h) // 2
        image = image.crop((0, top, source_w, top + new_h))
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def load_mask(path: str | Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        source_ratio = mask.width / mask.height
        target_ratio = size[0] / size[1]
        if abs(source_ratio - target_ratio) > 0.01:
            raise ValueError(f"Mask aspect ratio does not match the image: {mask.size} vs {size}")
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value >= 128 else 0, mode="L")


def auto_change_mask(
    before: Image.Image,
    after: Image.Image,
    *,
    threshold: float = 24.0,
    min_area_ratio: float = 0.0005,
    dilate: int = 5,
) -> Image.Image:
    """Estimate intentional edit regions from an aligned before/after pair.

    This is the zero-setup backend. Production workflows can replace the
    returned mask with a SAM 3.1 mask through ``mask_path``/``RoundSpec.mask``.
    """
    if before.size != after.size:
        raise ValueError("before and after must share a canvas")

    a = np.asarray(before, dtype=np.float32)
    b = np.asarray(after, dtype=np.float32)
    delta = np.mean(np.abs(a - b), axis=2)
    delta = cv2.GaussianBlur(delta, (0, 0), 1.2)

    # Ignore small global relighting/compression drift while retaining strong,
    # spatially coherent changes. The explicit threshold remains the floor.
    noise_floor = float(np.median(delta))
    adaptive = max(float(threshold), noise_floor + max(10.0, 3.0 * float(np.median(np.abs(delta - noise_floor)))))
    binary = (delta >= adaptive).astype(np.uint8) * 255

    close_size = max(9, int(round(min(before.size) * 0.006)) | 1)
    close_kernel = np.ones((close_size, close_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(16, round(before.width * before.height * min_area_ratio))
    kept = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == label] = 255

    # A difference map often marks only the strongest texture/color changes
    # inside a newly generated object. Fill enclosed holes so the object is
    # transferred as one coherent region rather than as scattered pixels.
    contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(kept, contours, -1, 255, thickness=cv2.FILLED)

    if dilate > 0 and np.any(kept):
        kernel_size = max(1, dilate * 2 + 1)
        kept = cv2.dilate(kept, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    return Image.fromarray(kept)


def _recolor_from_reference(
    base: Image.Image,
    target: Image.Image,
    prechange: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, dict]:
    """Transfer color statistics while retaining base geometry and texture."""
    base_hsv = np.asarray(base.convert("HSV"), dtype=np.uint8).copy()
    target_hsv = np.asarray(target.convert("HSV"), dtype=np.uint8)
    pre_hsv = np.asarray(prechange.convert("HSV"), dtype=np.uint8)
    selected = np.asarray(mask, dtype=np.uint8) > 0
    valid_target = selected & (target_hsv[..., 2] > 20)
    valid_pre = selected & (pre_hsv[..., 2] > 20)
    if not np.any(valid_target) or not np.any(valid_pre):
        raise ValueError("Recolor mask does not cover valid pixels")

    target_sat = int(np.median(target_hsv[..., 1][valid_target]))
    if target_sat >= 32:
        hue_pixels = valid_target & (target_hsv[..., 1] >= max(20, target_sat // 3))
        target_hue = int(np.median(target_hsv[..., 0][hue_pixels])) if np.any(hue_pixels) else 0
    else:
        target_hue = 0

    target_value = float(np.median(target_hsv[..., 2][valid_target]))
    pre_value = float(np.median(pre_hsv[..., 2][valid_pre]))
    value_ratio = target_value / max(pre_value, 1.0)

    base_hsv[..., 0] = target_hue
    base_hsv[..., 1] = target_sat
    base_hsv[..., 2] = np.clip(base_hsv[..., 2].astype(np.float32) * value_ratio, 0, 255).astype(np.uint8)
    recolored = Image.frombytes("HSV", base.size, base_hsv.tobytes()).convert("RGB")
    return recolored, {
        "target_hue_0_255": target_hue,
        "target_saturation_0_255": target_sat,
        "value_ratio": round(value_ratio, 6),
    }


def difference_heatmap(before: Image.Image, after: Image.Image, clip: float = 64.0) -> Image.Image:
    a = np.asarray(before, dtype=np.int16)
    b = np.asarray(after, dtype=np.int16)
    score = np.mean(np.abs(a - b), axis=2)
    normalized = np.clip(score / clip, 0.0, 1.0)
    # OpenCV TURBO is readable on both dark and bright regions.
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def _metrics(baseline: Image.Image, edited: Image.Image, result: Image.Image, mask: Image.Image) -> dict:
    base = np.asarray(baseline, dtype=np.int16)
    target = np.asarray(edited, dtype=np.int16)
    output = np.asarray(result, dtype=np.int16)
    selected = np.asarray(mask, dtype=np.uint8) > 0
    outside = ~selected
    outside_mismatches = int(np.any(output != base, axis=2)[outside].sum())
    before_delta = np.abs(target - base)
    after_delta = np.abs(output - base)
    return {
        "width": baseline.width,
        "height": baseline.height,
        "mask_pixels": int(selected.sum()),
        "mask_percent": round(float(selected.mean() * 100), 6),
        "outside_mask_pixel_mismatches": outside_mismatches,
        "before_mae_rgb": round(float(before_delta.mean()), 6),
        "after_mae_rgb": round(float(after_delta.mean()), 6),
        "status": "PASS" if outside_mismatches == 0 else "FAIL",
    }


def process_pair(
    before: str | Path | Image.Image,
    after: str | Path | Image.Image,
    *,
    mask_path: str | Path | None = None,
    mode: EditMode = "paste",
    threshold: float = 24.0,
    min_area_ratio: float = 0.0005,
    dilate: int = 5,
) -> PairResult:
    baseline = load_rgb(before) if not isinstance(before, Image.Image) else before.convert("RGB")
    edited_source = load_rgb(after) if not isinstance(after, Image.Image) else after.convert("RGB")
    edited = fit_to_canvas(edited_source, baseline.size)
    mask = load_mask(mask_path, baseline.size) if mask_path else auto_change_mask(
        baseline,
        edited,
        threshold=threshold,
        min_area_ratio=min_area_ratio,
        dilate=dilate,
    )

    if mode == "paste":
        source = edited
        transfer = None
    elif mode == "recolor":
        source, transfer = _recolor_from_reference(baseline, edited, baseline, mask)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result = baseline.copy()
    result.paste(source, (0, 0), mask)
    metrics = _metrics(baseline, edited, result, mask)
    metrics["mode"] = mode
    metrics["mask_source"] = "provided" if mask_path else "automatic_difference"
    metrics["color_transfer"] = transfer
    return PairResult(
        result=result,
        mask=mask,
        heatmap_before=difference_heatmap(baseline, edited),
        heatmap_after=difference_heatmap(baseline, result),
        metrics=metrics,
    )


def save_pair_result(result: PairResult, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output / "result.png",
        "mask": output / "mask.png",
        "heatmap_before": output / "heatmap_before.png",
        "heatmap_after": output / "heatmap_after.png",
        "report": output / "report.json",
    }
    result.result.save(paths["result"], "PNG", optimize=True)
    result.mask.save(paths["mask"], "PNG", optimize=True)
    result.heatmap_before.save(paths["heatmap_before"], "PNG", optimize=True)
    result.heatmap_after.save(paths["heatmap_after"], "PNG", optimize=True)
    report = {
        **result.metrics,
        "files": {key: path.name for key, path in paths.items() if key != "report"},
    }
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def process_sequence(
    original_path: str | Path,
    rounds: list[RoundSpec],
    output_dir: str | Path,
) -> dict:
    """Apply cumulative edit masks while restoring untouched original pixels."""
    original_path = Path(original_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original = load_rgb(original_path)
    cumulative = original.copy()
    cumulative_mask = Image.new("L", original.size, 0)
    previous_generated = original
    manifest_rounds = []

    for index, spec in enumerate(rounds, start=1):
        target_path = Path(spec.image)
        target = fit_to_canvas(load_rgb(target_path), original.size)
        new_mask = load_mask(spec.mask, original.size) if spec.mask else auto_change_mask(
            previous_generated,
            target,
            threshold=spec.threshold,
            min_area_ratio=spec.min_area_ratio,
            dilate=spec.dilate,
        )
        previous_mask = cumulative_mask.copy()
        transfer = None

        if spec.mode == "paste":
            current_pixels = target
        elif spec.mode == "recolor":
            current_pixels, transfer = _recolor_from_reference(cumulative, target, previous_generated, new_mask)
        else:
            raise ValueError(f"Unsupported mode: {spec.mode}")

        result = original.copy()
        if spec.overlap == "current":
            result.paste(cumulative, (0, 0), previous_mask)
            result.paste(current_pixels, (0, 0), new_mask)
        elif spec.overlap == "previous":
            result.paste(current_pixels, (0, 0), new_mask)
            result.paste(cumulative, (0, 0), previous_mask)
        else:
            raise ValueError(f"Unsupported overlap policy: {spec.overlap}")

        cumulative_mask = ImageChops.lighter(previous_mask, new_mask)
        cumulative = result
        previous_generated = target
        stem = f"round_{index:02d}"
        result_path = output / f"{stem}.png"
        mask_path = output / f"{stem}_cumulative_mask.png"
        heatmap_path = output / f"{stem}_heatmap.png"
        result.save(result_path, "PNG", optimize=True)
        cumulative_mask.save(mask_path, "PNG", optimize=True)
        difference_heatmap(original, result).save(heatmap_path, "PNG", optimize=True)

        round_metrics = _metrics(original, target, result, cumulative_mask)
        manifest_rounds.append(
            {
                "round": index,
                "name": spec.name or stem,
                "input": str(target_path),
                "mode": spec.mode,
                "overlap": spec.overlap,
                "mask_source": str(spec.mask) if spec.mask else "automatic_difference",
                "color_transfer": transfer,
                "output": result_path.name,
                "cumulative_mask": mask_path.name,
                "heatmap": heatmap_path.name,
                **round_metrics,
            }
        )

    manifest = {
        "workflow": "cumulative masks with exact original-pixel restoration outside the union",
        "original": str(original_path),
        "original_sha256": _sha256(original_path),
        "canvas": list(original.size),
        "round_count": len(rounds),
        "rounds": manifest_rounds,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_rounds_config(path: str | Path) -> tuple[str | None, list[RoundSpec]]:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    original = data.get("original")
    if original and not Path(original).is_absolute():
        original = str((base / original).resolve())
    rounds: list[RoundSpec] = []
    for item in data["rounds"]:
        normalized = dict(item)
        for key in ("image", "mask"):
            if normalized.get(key) and not Path(normalized[key]).is_absolute():
                normalized[key] = str((base / normalized[key]).resolve())
        rounds.append(RoundSpec(**normalized))
    return original, rounds
