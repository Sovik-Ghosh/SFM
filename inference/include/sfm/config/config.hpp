#pragma once
// Run configuration, schema v3 (example: configs/bunny.json). One JSON file describes a
// run completely. Parsing is strict: every key is required, unknown keys are errors, and every
// error names the JSON path ("dataset.first"). v2 adds the stage sections of the CPU pipeline
// (features, matching, reconstruction, ba, output); tools/migrate_config.py upgrades a v1 file.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "sfm/core/logger.hpp"

namespace sfm::config {

inline constexpr int kSchemaVersion = 6;

struct RunConfig {
  std::string name;                  // [A-Za-z0-9._-]+, used as the output sub-directory
  std::filesystem::path output_dir;  // outputs go to <output_dir>/<name>/
  core::LogLevel log_level = core::LogLevel::kInfo;
  std::uint64_t seed = 0;
};

struct DeviceConfig {
  int index = 0;  // CUDA device ordinal; checked against the device count at startup, not here
};

// One pinhole camera for every image, instead of a calibration file per image (optional
// "dataset.camera"). fx/fy in pixels; cx/cy default to the image centre ((w - 1) / 2, (h - 1) / 2).
// exif: fx = fy from each JPEG's EXIF 35 mm-equivalent focal length (or its focal length in mm and
// sensor_width_mm); images without it use fx_default_factor * max(width, height).
// fx = 0: unset; fy = 0: same as fx; cx, cy < 0: the image centre. sensor_width_mm is used with
// exif when a file has no 35 mm-equivalent focal length; fx_default_factor (COLMAP's default)
// when nothing else is known.
struct CameraConfig {
  double fx = 0.0;
  double fy = 0.0;
  double cx = -1.0;
  double cy = -1.0;
  bool exif = false;
  double sensor_width_mm = 0.0;
  double fx_default_factor = 1.2;

  bool enabled() const { return fx > 0.0 || exif; }
};

struct DatasetConfig {
  std::filesystem::path root;
  std::filesystem::path images_dir;  // relative to root
  std::filesystem::path masks_dir;   // relative to root; ignored when use_masks is false
  std::filesystem::path calib_dir;   // relative to root; ignored when camera.enabled()
  int first = 0;                     // inclusive image range (ids, or indices in "folder" layout)
  int last = 0;                      // "folder" layout: -1 = up to the last image
  bool use_masks = true;
  // Optional keys (absent from older configs):
  // "numbered": <images_dir>/NNNN.ppm, <masks_dir>/NNNN.pgm, <calib_dir>/NNNN.txt for ids
  //   first..last (the bunny layout).
  // "folder": every .jpg/.jpeg/.png/.ppm/.pgm in images_dir (any case), sorted by file name;
  //   first..last index that list. The mask of photo.jpg is <masks_dir>/photo.png or .pgm, its
  //   calibration <calib_dir>/photo.txt.
  std::string layout = "numbered";
  bool mask_object_white = false;  // "mask_object": "black" (default) or "white"
  CameraConfig camera;
  // Optional "distortion": {"k1", "k2"}: known radial lens distortion of every image (COLMAP's
  // OPENCV model without tangential terms; normalised coordinates). Keypoints are undistorted
  // after detection; 0, 0 = none.
  double k1 = 0.0;
  double k2 = 0.0;

  // Image count for a fixed range; -1 when it depends on the folder ("folder" with last = -1).
  int image_count() const { return last < 0 ? -1 : last - first + 1; }
};

struct MemoryConfig {
  std::uint32_t arena_mb = 0;
  std::uint32_t max_images = 0;
  std::uint32_t max_kpts_per_image = 0;
  std::uint32_t max_matches_per_pair = 0;
};

// Keypoints and descriptors. Masks come from dataset.use_masks.
// SIFT detector (methods sift_*; features/sift.hpp). Defaults = COLMAP's.
struct SiftConfig {
  int octaves = 4;
  int octave_resolution = 3;  // scales per octave
  int first_octave = -1;      // -1: start from the image upsampled x2; 0: from the image
  double peak_threshold = 0.02 / 3.0;
  double edge_threshold = 10.0;
  int max_orientations = 2;  // per location
};

struct FeaturesConfig {
  std::string method = "orb_cpu";     // registry name
  std::uint32_t max_features = 2000;  // ORB: keypoints; SIFT: locations (x max_orientations)
  int pyramid_levels = 8;
  double scale = 1.2;       // size ratio between pyramid levels
  int fast_threshold = 10;  // FAST intensity threshold (0..255)
  int edge_border = 31;     // no keypoint closer than this to a level's border (pixels)
  SiftConfig sift;          // used by the sift_* methods only
};

struct RansacConfig {
  // "essential": 5-point E with the calibrated K (not degenerate on planar scenes; v3 default);
  // "fundamental": 8-point F (no K needed; degenerate when the matches lie on a plane).
  std::string model = "essential";
  double threshold_px = 3.0;  // inlier threshold (Sampson / reprojection distance, pixels)
  double confidence = 0.999;
  std::uint32_t max_iters = 2000;
};

// Which image pairs are matched.
struct PairsConfig {
  // "exhaustive": every pair (n (n - 1) / 2: fine for tens of images);
  // "sequential": image i with i+1 .. i+window (video / ordered captures), plus all pairs among
  // every keyframe_stride-th image (to catch revisits); about n window + (n / stride)^2 / 2 pairs.
  std::string mode = "exhaustive";
  std::uint32_t window = 10;
  std::uint32_t keyframe_stride = 10;  // 0 = no keyframe pairs
};

struct MatchingConfig {
  std::string method = "bf_cpu";
  double ratio = 0.75;  // Lowe ratio test: best < ratio * second best
  bool mutual = false;  // keep only matches that are also best in the reverse direction
  RansacConfig ransac;
  PairsConfig pairs;
  std::uint32_t min_inliers = 30;  // a pair is verified with at least this many F inliers
};

struct PnpConfig {
  double threshold_px = 8.0;
  std::uint32_t min_inliers = 20;
  // A camera is registered only if at least this share of its 2D-3D correspondences are inliers
  // (repetitive scenes produce many wrong correspondences that still reach min_inliers).
  double min_inlier_ratio = 0.4;
  std::uint32_t max_iters = 1000;
};

struct TriangulationConfig {
  double min_angle_deg = 2.0;  // smallest ray angle of a new point
  double max_reproj_px = 4.0;  // largest reprojection error of a new point in any view
};

struct ReconstructionConfig {
  std::uint32_t init_min_inliers = 50;
  double init_min_angle_deg = 5.0;  // median triangulation angle of the initial pair
  PnpConfig pnp;
  std::uint32_t min_correspondences = 30;  // 2D-3D matches needed to try registering a camera
  TriangulationConfig triangulation;
  // Rigid camera rig (v6, P9b): <dataset root>/rig.txt (tools/rig_from_dataset.py) gives each
  // image's frame and sensor; one pose per frame is estimated, the sensors' offsets stay fixed, and
  // registering one image of a frame registers the whole frame.
  bool rig = false;
};

struct BaConfig {
  std::string method = "lm_schur_cpu";
  double huber_px = 2.0;
  std::uint32_t max_iters = 50;
  // BA after the initial pair, after every n registrations, and when no camera is left to try;
  // 0 = only the last two. Bunny: 0 -> 14 deg median rotation error, 5 -> 0.6 deg (P2 design).
  std::uint32_t every_n_cameras = 5;
  // Local BA (v5): the periodic BA adjusts only the local_cameras most recently registered
  // cameras and their points, until the model has grown by global_growth (fraction) since the
  // last global BA. 0 = every BA global. delivery_area (237): BA 57 s -> 13 s, same accuracy.
  std::uint32_t local_cameras = 10;
  double global_growth = 0.25;
  // Final global BA (v5) after the incremental loop with a tighter Huber threshold; 0 iterations =
  // none. courtyard: median rotation 0.050 -> 0.035 deg (COLMAP 0.034).
  double final_huber_px = 0.5;
  std::uint32_t final_max_iters = 10;
  // Optional "refine_focal": after the final BA, search one scale factor s for every camera's
  // focal length (fx, fy -> s fx, s fy; s in [0.74, 1.35]) that minimises the final BA cost, for
  // photos whose focal length is only estimated (dataset.camera exif). Needs final_max_iters > 0.
  bool refine_focal = false;
  // Optional "refine_distortion": after the final BA (and refine_focal), search the radial
  // distortion k1 in [-0.4, 0.4] that minimises the reprojection error in the recorded pixels
  // (each candidate re-undistorts the keypoints and re-runs the final BA). k2 stays as configured.
  bool refine_distortion = false;
};

struct OutputConfig {
  bool dump_stages = true;  // .npy of every stage output under <run>/stages/
  bool ply = true;
  bool colmap = true;
  // Optional "colmap_scene": also write <run>/scene/{images,sparse/0}, the folder layout Gaussian
  // splatting and NeRF tools (3DGS, nerfstudio, gsplat) open directly.
  bool colmap_scene = false;
};

struct Config {
  int schema_version = kSchemaVersion;
  RunConfig run;
  DeviceConfig device;
  DatasetConfig dataset;
  MemoryConfig memory;
  FeaturesConfig features;
  MatchingConfig matching;
  ReconstructionConfig reconstruction;
  BaConfig ba;
  OutputConfig output;
};

// JSON text -> Config. Checks structure and types only (missing, unknown, wrong-type keys,
// schema_version != 2, integer range). Throws ConfigError on the first problem.
Config parse_config(std::string_view json_text);

// Reads the file, parses it, makes run.output_dir and dataset.root absolute (relative ones are
// taken against `base_dir`), then validates. Throws IoError if the file cannot be read, ConfigError
// otherwise.
Config load_config(const std::filesystem::path& file,
                   const std::filesystem::path& base_dir = std::filesystem::current_path());

// Semantic checks: value ranges, consistency, dataset directories exist (relative paths resolved
// against base_dir). Collects every problem and throws one ConfigError listing them all.
void validate_config(const Config& config, const std::filesystem::path& base_dir);

// Config -> pretty JSON (2-space indent, keys in schema order, trailing newline). Round-trips:
// to_json(parse_config(to_json(c))) == to_json(c).
std::string to_json(const Config& config);

}  // namespace sfm::config
