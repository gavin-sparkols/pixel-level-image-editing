from pathlib import Path

import numpy as np
from PIL import Image

from pixel_level_editing.core import RoundSpec, process_pair, process_sequence


def _fixture_pair(size=(180, 120)):
    width, height = size
    x = np.linspace(40, 180, width, dtype=np.uint8)
    base = np.repeat(x[None, :, None], height, axis=0)
    base = np.repeat(base, 3, axis=2)
    before = Image.fromarray(base)
    after_array = np.clip(base.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    after_array[35:85, 70:135] = (230, 35, 40)
    after = Image.fromarray(after_array)
    return before, after


def test_pair_restores_every_pixel_outside_mask():
    before, after = _fixture_pair()
    output = process_pair(before, after, threshold=20, min_area_ratio=0.001, dilate=2)
    mask = np.asarray(output.mask) > 0
    baseline = np.asarray(before)
    result = np.asarray(output.result)
    assert mask.any()
    assert np.array_equal(result[~mask], baseline[~mask])
    assert output.metrics["outside_mask_pixel_mismatches"] == 0
    assert output.metrics["status"] == "PASS"


def test_sequence_accumulates_round_masks(tmp_path: Path):
    original = Image.new("RGB", (80, 80), (100, 100, 100))
    round_one = original.copy()
    round_one.paste((230, 30, 30), (5, 5, 25, 25))
    round_two = round_one.copy()
    round_two.paste((30, 50, 230), (50, 50, 75, 75))
    original_path = tmp_path / "original.png"
    one_path = tmp_path / "one.png"
    two_path = tmp_path / "two.png"
    original.save(original_path)
    round_one.save(one_path)
    round_two.save(two_path)

    manifest = process_sequence(
        original_path,
        [
            RoundSpec(image=str(one_path), threshold=15, min_area_ratio=0, dilate=0),
            RoundSpec(image=str(two_path), threshold=15, min_area_ratio=0, dilate=0),
        ],
        tmp_path / "out",
    )
    final = np.asarray(Image.open(tmp_path / "out" / "round_02.png"))
    assert tuple(final[10, 10]) == (230, 30, 30)
    assert tuple(final[60, 60]) == (30, 50, 230)
    assert manifest["rounds"][-1]["outside_mask_pixel_mismatches"] == 0
