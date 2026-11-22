#include "sfm/ba/adjust.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sfm::ba {

namespace {

// Rig mode (rec.rig set, P9b): one parameter block per frame; each observation goes through a
// per-image sensor (the image's K and its sensor's offset). Gauge: the frames held fixed; with
// none, the frame of rec.order[0]. The rig's metric offsets fix the scale (unless every offset has
// a zero translation: then one translation component of the next frame, as without a rig).
Report bundle_adjust_rig(recon::Reconstruction& rec, const recon::TrackIndex& index,
                         const Options& o, Solver solver, const std::vector<int>& free_images) {
  const data::Rig& rig = *rec.rig;
  const bool local = !free_images.empty();
  const auto frame_of = [&](int image) { return rig.frame[static_cast<std::size_t>(image)]; };
  std::vector<char> frame_free(static_cast<std::size_t>(rig.frames), local ? 0 : 1);
  for (const int i : free_images) frame_free[static_cast<std::size_t>(frame_of(i))] = 1;
  const std::vector<recon::PointObs> all = recon::point_observations(rec, index);
  std::vector<char> track_in(rec.points.size(), local ? 0 : 1);
  if (local) {
    for (const recon::PointObs& ob : all) {
      if (frame_free[static_cast<std::size_t>(frame_of(ob.camera))] != 0) {
        track_in[static_cast<std::size_t>(ob.track)] = 1;
      }
    }
  }
  std::vector<char> frame_in(static_cast<std::size_t>(rig.frames), 0);
  for (const recon::PointObs& ob : all) {
    if (track_in[static_cast<std::size_t>(ob.track)] != 0) {
      frame_in[static_cast<std::size_t>(frame_of(ob.camera))] = 1;
    }
  }

  Problem p;
  std::vector<int> block_of_frame(static_cast<std::size_t>(rig.frames), -1);
  std::vector<int> frame_of_block;
  int fixed_blocks = 0;
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {  // first registered image of each frame
    const int f = frame_of(static_cast<int>(i));
    const auto fu = static_cast<std::size_t>(f);
    if (!rec.cameras[i].registered || frame_in[fu] == 0 || block_of_frame[fu] >= 0) continue;
    block_of_frame[fu] = static_cast<int>(frame_of_block.size());
    frame_of_block.push_back(f);
    const recon::Pose fp = recon::rig_frame_pose(rig, static_cast<int>(i), rec.cameras[i].pose);
    p.K.push_back(rec.cameras[i].K);  // unused: every observation has a sensor
    p.R.push_back(fp.R);
    p.t.push_back(fp.t);
    const bool fixed = frame_free[fu] == 0;
    fixed_blocks += fixed ? 1 : 0;
    p.fixed.push_back({fixed, fixed, fixed, fixed, fixed, fixed});
  }
  std::vector<int> sensor_of_image(rec.cameras.size(), -1);
  std::vector<int> point_of_track(rec.points.size(), -1);
  std::vector<int> track_of_point;
  for (const recon::PointObs& ob : all) {
    if (track_in[static_cast<std::size_t>(ob.track)] == 0) continue;
    const auto img = static_cast<std::size_t>(ob.camera);
    auto& sn = sensor_of_image[img];
    if (sn < 0) {
      sn = static_cast<int>(p.sensors.size());
      const auto s = static_cast<std::size_t>(rig.sensor[img]);
      p.sensors.push_back({rec.cameras[img].K, rig.R[s], rig.t[s]});
    }
    auto& pid = point_of_track[static_cast<std::size_t>(ob.track)];
    if (pid < 0) {
      pid = static_cast<int>(track_of_point.size());
      track_of_point.push_back(ob.track);
      p.X.push_back(rec.points[static_cast<std::size_t>(ob.track)]);
    }
    p.obs.push_back(
        {block_of_frame[static_cast<std::size_t>(frame_of(ob.camera))], pid, ob.uv, sn});
  }
  if (p.obs.empty()) return Report{};

  if (fixed_blocks == 0 && !rec.order.empty()) {
    const int b0 = block_of_frame[static_cast<std::size_t>(frame_of(rec.order[0]))];
    if (b0 >= 0) p.fixed[static_cast<std::size_t>(b0)] = {true, true, true, true, true, true};
    double baseline = 0.0;
    for (const auto& t : rig.t) baseline = std::max(baseline, t.norm());
    if (baseline <= 0.0) {  // no metric offsets: fix the scale as without a rig
      for (const int image : rec.order) {
        const int b = block_of_frame[static_cast<std::size_t>(frame_of(image))];
        if (b < 0 || b == b0) continue;
        Eigen::Index axis = 0;
        p.t[static_cast<std::size_t>(b)].cwiseAbs().maxCoeff(&axis);
        p.fixed[static_cast<std::size_t>(b)][3 + static_cast<std::size_t>(axis)] = true;
        break;
      }
    }
  }

  const Report r = solver(p, o);

  std::vector<recon::Pose> frame_pose(static_cast<std::size_t>(rig.frames));
  std::vector<char> moved(static_cast<std::size_t>(rig.frames), 0);
  for (std::size_t b = 0; b < frame_of_block.size(); ++b) {
    if (p.fixed[b][0] && p.fixed[b][3] && p.fixed[b][4] && p.fixed[b][5]) continue;
    const auto f = static_cast<std::size_t>(frame_of_block[b]);
    frame_pose[f] = {p.R[b], p.t[b]};
    moved[f] = 1;
  }
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    const auto f = static_cast<std::size_t>(frame_of(static_cast<int>(i)));
    if (!rec.cameras[i].registered || moved[f] == 0) continue;
    rec.cameras[i].pose = recon::rig_image_pose(rig, static_cast<int>(i), frame_pose[f]);
  }
  for (std::size_t x = 0; x < track_of_point.size(); ++x) {
    rec.points[static_cast<std::size_t>(track_of_point[x])] = p.X[x];
  }
  return r;
}

}  // namespace

Report bundle_adjust(recon::Reconstruction& rec, const recon::TrackIndex& index, const Options& o,
                     Solver solver, const std::vector<int>& free_images) {
  if (rec.rig) return bundle_adjust_rig(rec, index, o, solver, free_images);
  const bool local = !free_images.empty();
  std::vector<char> is_free(rec.cameras.size(), local ? 0 : 1);
  for (const int i : free_images) is_free[static_cast<std::size_t>(i)] = 1;
  const std::vector<recon::PointObs> all = recon::point_observations(rec, index);

  // Points in the problem: all (global) or those seen by a free camera (local).
  std::vector<char> track_in(rec.points.size(), local ? 0 : 1);
  if (local) {
    for (const recon::PointObs& ob : all) {
      if (is_free[static_cast<std::size_t>(ob.camera)] != 0) {
        track_in[static_cast<std::size_t>(ob.track)] = 1;
      }
    }
  }
  // Cameras in the problem: registered ones observing an included point (in image order).
  std::vector<char> cam_in(rec.cameras.size(), 0);
  for (const recon::PointObs& ob : all) {
    if (track_in[static_cast<std::size_t>(ob.track)] != 0) {
      cam_in[static_cast<std::size_t>(ob.camera)] = 1;
    }
  }

  Problem p;
  std::vector<int> cam_of_image(rec.cameras.size(), -1);
  std::vector<int> image_of_cam;
  int fixed_cameras = 0;
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    if (!rec.cameras[i].registered || cam_in[i] == 0) continue;
    cam_of_image[i] = static_cast<int>(image_of_cam.size());
    image_of_cam.push_back(static_cast<int>(i));
    p.K.push_back(rec.cameras[i].K);
    p.R.push_back(rec.cameras[i].pose.R);
    p.t.push_back(rec.cameras[i].pose.t);
    const bool f = is_free[i] == 0;
    fixed_cameras += f ? 1 : 0;
    p.fixed.push_back({f, f, f, f, f, f});
  }
  std::vector<int> point_of_track(rec.points.size(), -1);
  std::vector<int> track_of_point;
  for (const recon::PointObs& ob : all) {
    if (track_in[static_cast<std::size_t>(ob.track)] == 0) continue;
    auto& pid = point_of_track[static_cast<std::size_t>(ob.track)];
    if (pid < 0) {
      pid = static_cast<int>(track_of_point.size());
      track_of_point.push_back(ob.track);
      p.X.push_back(rec.points[static_cast<std::size_t>(ob.track)]);
    }
    p.obs.push_back({cam_of_image[static_cast<std::size_t>(ob.camera)], pid, ob.uv});
  }
  if (p.obs.empty()) return Report{};

  // Gauge.
  const auto cam = [&](int image) { return cam_of_image[static_cast<std::size_t>(image)]; };
  if (fixed_cameras < 2 && rec.order.size() >= 2) {
    const int c0 = cam(rec.order[0]);
    const int c1 = cam(rec.order[1]);
    if (fixed_cameras == 0 && c0 >= 0 && c1 >= 0) {
      p.fixed[static_cast<std::size_t>(c0)] = {true, true, true, true, true, true};
      Eigen::Index axis = 0;
      p.t[static_cast<std::size_t>(c1)].cwiseAbs().maxCoeff(&axis);
      p.fixed[static_cast<std::size_t>(c1)][3 + static_cast<std::size_t>(axis)] = true;
    } else if (fixed_cameras == 1) {
      // Scale about the fixed camera's centre C0 moves a free camera's t by -R (C - C0) per unit.
      std::size_t fixed_c = 0;
      while (!p.fixed[fixed_c][0]) ++fixed_c;
      const Eigen::Vector3d C0 = -p.R[fixed_c].transpose() * p.t[fixed_c];
      for (const int image : rec.order) {
        const int c = cam(image);
        if (c < 0 || p.fixed[static_cast<std::size_t>(c)][0]) continue;
        const auto cc = static_cast<std::size_t>(c);
        const Eigen::Vector3d C = -p.R[cc].transpose() * p.t[cc];
        Eigen::Index axis = 0;
        (p.R[cc] * (C - C0)).cwiseAbs().maxCoeff(&axis);
        p.fixed[cc][3 + static_cast<std::size_t>(axis)] = true;
        break;
      }
    }
  }

  const Report r = solver(p, o);

  for (std::size_t c = 0; c < image_of_cam.size(); ++c) {
    if (p.fixed[c][0] && p.fixed[c][3] && p.fixed[c][4] && p.fixed[c][5]) continue;
    auto& camera = rec.cameras[static_cast<std::size_t>(image_of_cam[c])];
    camera.pose.R = p.R[c];
    camera.pose.t = p.t[c];
  }
  for (std::size_t x = 0; x < track_of_point.size(); ++x) {
    rec.points[static_cast<std::size_t>(track_of_point[x])] = p.X[x];
  }
  return r;
}

}  // namespace sfm::ba
