"""Epipole: Structure-from-Motion in C++ and CUDA.

    import epipole
    r = epipole.reconstruct("photos/")            # a folder of JPEG/PNG photos
    print(r.metrics["registered"], r.points.shape)
    r.R, r.t                                      # world -> camera poses (registered images)

    r = epipole.run("configs/bunny_sift_tuned2.json")   # or a full config (path or dict)

Every run writes its usual output folder (model.ply, colmap/, metrics.json, ...): Result.run_dir.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from . import _epipole
from ._epipole import ConfigError, Error, IoError

__version__: str = _epipole.__version__
__all__ = ["run", "reconstruct", "device", "default_config", "Result", "Error", "ConfigError",
           "IoError", "__version__"]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".pgm"}


@dataclass
class Result:
    """Outcome of a run.

    names       image file names, in the run's order (one per image, registered or not)
    registered  bool (N,), which images have a pose
    K, R        (N, 3, 3) intrinsics (pixel centres at integer coordinates) and rotation world->camera
    t           (N, 3) translation: x_camera = R x_world + t
    points      (M, 3) reconstructed 3D points
    colors      (M, 3) uint8 RGB per point (empty when nothing was exported)
    distortion  (k1, k2) radial lens distortion the model was made with (configured or refined)
    metrics     metrics.json as a dict (registered, points, reprojection_px, rotation_error_deg, ...)
    timings     seconds per stage
    run_dir     the output folder
    """

    run_dir: Path
    metrics: Optional[dict]
    timings: dict
    names: list = field(default_factory=list)
    registered: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    K: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    R: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    t: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    distortion: tuple = (0.0, 0.0)

    @property
    def centers(self) -> np.ndarray:
        """(N, 3) camera centres in world coordinates, C = -R^T t."""
        return -np.einsum("nji,nj->ni", self.R, self.t)


def default_config() -> dict:
    """The template reconstruct() starts from (configs/bunny_sift_tuned2.json)."""
    return json.loads(_epipole.default_config)


def device() -> dict:
    """{'cuda_build': bool, 'gpu': bool, 'device': name, 'unusable': reason or ''}."""
    return dict(_epipole.device())


def run(config: Union[str, os.PathLike, Mapping[str, Any]], *, backend: str = "auto",
        log_level: str = "warn") -> Result:
    """Runs a full config: a JSON file (relative paths in it are taken from the current directory,
    as for sfm_run) or a dict with the same content. backend: "auto", "cpu" or "gpu"."""
    if isinstance(config, Mapping):
        text = json.dumps(config)
    else:
        text = Path(config).read_text()
    out = _epipole.run(text, os.getcwd(), backend, log_level)
    return Result(
        run_dir=Path(out["run_dir"]),
        metrics=json.loads(out["metrics"]) if out.get("metrics") is not None else None,
        timings=json.loads(out["timings"]),
        names=out.get("names", []),
        registered=out.get("registered", np.zeros(0, bool)),
        K=out.get("K", np.zeros((0, 3, 3))),
        R=out.get("R", np.zeros((0, 3, 3))),
        t=out.get("t", np.zeros((0, 3))),
        points=out.get("points", np.zeros((0, 3))),
        colors=out.get("colors", np.zeros((0, 3), np.uint8)),
        distortion=(out.get("k1", 0.0), out.get("k2", 0.0)),
    )


def reconstruct(images: Union[str, os.PathLike], *, output_dir: Union[str, os.PathLike] = "outputs",
                name: Optional[str] = None, camera: Optional[Mapping[str, Any]] = None,
                masks: Union[str, os.PathLike, None] = None, mask_object: str = "white",
                refine_focal: bool = True, distortion: Optional[tuple] = None,
                refine_distortion: bool = False, sequential: bool = False, colmap_scene: bool = True,
                backend: str = "auto", log_level: str = "warn") -> Result:
    """Reconstructs a folder of photos (JPEG/PNG, any names, taken in name order, one size, one
    camera).

    camera       {"exif": True} (default: focal length from EXIF, else 1.2 x the longer side) or
                 {"fx": ..., "fy": ..., "cx": ..., "cy": ...} in pixels
    masks        optional folder with <photo stem>.png per photo; features only on the object
                 (white pixels by default; mask_object="black" for black objects)
    refine_focal search the focal length that fits best after the final bundle adjustment
    distortion   known radial distortion (k1, k2), COLMAP's OPENCV convention
    refine_distortion  estimate k1 (needs keypoints across the frame; re-runs until it settles)
    sequential   match each photo with its neighbours (video, walk) instead of all pairs
    colmap_scene also write <run_dir>/scene/{images,sparse/0} for 3DGS / nerfstudio / gsplat
    """
    folder = Path(images).resolve()
    if not folder.is_dir():
        raise IoError(f"{folder}: not a folder")
    count = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)
    if count < 2:
        raise IoError(f"{folder}: needs at least 2 images (.jpg .jpeg .png .ppm .pgm), found {count}")
    cfg = default_config()
    cfg["run"]["name"] = name or folder.name or "run"
    cfg["run"]["output_dir"] = str(Path(output_dir).resolve())
    cfg["run"]["log_level"] = log_level
    cfg["dataset"] = {
        "root": str(folder), "images_dir": ".", "calib_dir": "", "layout": "folder",
        "first": 0, "last": -1, "use_masks": masks is not None,
        "masks_dir": str(Path(masks).resolve()) if masks is not None else "",
        "mask_object": mask_object, "camera": dict(camera) if camera is not None else {"exif": True},
    }
    cfg["memory"]["max_images"] = max(cfg["memory"]["max_images"], count)
    for section in ("features", "matching", "ba"):
        cfg[section]["method"] = "auto"
    cfg["ba"]["refine_focal"] = refine_focal
    cfg["ba"]["refine_distortion"] = refine_distortion
    if distortion is not None:
        cfg["dataset"]["distortion"] = {"k1": float(distortion[0]), "k2": float(distortion[1])}
    if sequential:
        cfg["matching"]["pairs"] = {"mode": "sequential", "window": 10, "keyframe_stride": 10}
    cfg["output"]["colmap_scene"] = colmap_scene
    return run(cfg, backend=backend, log_level=log_level)
