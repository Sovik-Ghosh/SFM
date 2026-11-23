"""Tests of the Python module (run by ctest when SFM_BUILD_PYTHON=ON and SFM_BUILD_TESTS=ON).
Usage: test_epipole.py <source dir> <work dir>. Plain asserts: no pytest needed."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import epipole

src, work = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)

# run(): the bunny ORB config on the CPU (fast, every build) gives the sfm_run result.
cfg = json.loads((src / "configs/bunny.json").read_text())
cfg["dataset"]["root"] = str(src / "data/bunny")
cfg["run"]["output_dir"] = str(work)
r = epipole.run(cfg, backend="cpu")
assert r.metrics["registered"] == 35, r.metrics["registered"]
assert abs(r.metrics["rotation_error_deg"]["median"] - 0.545) < 0.01, r.metrics
assert r.points.shape == (r.metrics["points"], 3) and r.colors.shape == r.points.shape
assert r.colors.dtype == np.uint8 and r.colors.mean() > 50  # real colours (the bunny is orange)
assert r.registered.sum() == 35 and len(r.names) == 36 and r.names[0] == "0000.ppm"
assert np.allclose(np.einsum("nji,njk->nik", r.R, r.R), np.eye(3), atol=1e-9)
assert (r.run_dir / "colmap" / "points3D.bin").exists()

# reconstruct(): a folder of images with other names, focal length from EXIF (none: 1.2 x side).
photos, masks = work / "photos", work / "masks"
photos.mkdir()
masks.mkdir()
for i in range(12):  # 12 neighbouring views, 10 degrees apart, with the silhouettes (black object)
    shutil.copy(src / f"data/bunny/images/{i:04d}.ppm", photos / f"view_{i:02d}.ppm")
    shutil.copy(src / f"data/bunny/silhouettes/{i:04d}.pgm", masks / f"view_{i:02d}.pgm")
r = epipole.reconstruct(photos, output_dir=work, name="folder", backend="cpu", refine_focal=False,
                        masks=masks, mask_object="black")
assert r.metrics["registered"] >= 10, r.metrics["registered"]
assert r.names[0] == "view_00.ppm"
assert (r.run_dir / "scene" / "images" / "view_00.png").exists()
assert abs(r.K[0, 0, 0] - 1.2 * 1024) < 1e-6  # no EXIF in a PPM: the default focal length
assert r.distortion == (0.0, 0.0)

for bad, err in (({"schema_version": 6}, epipole.ConfigError), (work / "missing", epipole.IoError)):
    try:
        epipole.run(bad) if isinstance(bad, dict) else epipole.reconstruct(bad)
        raise AssertionError(f"no {err.__name__}")
    except err:
        pass
shutil.rmtree(work)
print("python tests passed")
