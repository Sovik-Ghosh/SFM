#pragma once
// Bundle adjustment of a Reconstruction: builds the Problem from the registered cameras and
// reconstructed points, solves, writes the result back.

#include "sfm/ba/ba.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::ba {

// Global (free_images empty): every registered camera and reconstructed point. Gauge:
// rec.order[0] fully fixed; the largest |t| component of rec.order[1] fixed (scale).
//
// Local (free_images = some registered images): the points seen by a free image, the free
// cameras, and every other registered camera seeing one of those points, held fixed. Gauge: the
// fixed cameras; with fewer than two of them, rec.order[0] / rec.order[1] as above if they are
// free, else the scale through one translation component of the first free camera in rec.order
// (the one that moves most when the free part is scaled about the fixed camera).
// Rig (rec.rig set, P9b): one pose per rig frame, the sensor offsets held fixed; see adjust.cpp.
// `solver` runs the optimisation (default: the CPU solve; solve_cuda for the GPU).
using Solver = Report (*)(Problem&, const Options&);
Report bundle_adjust(recon::Reconstruction& rec, const recon::TrackIndex& index, const Options& o,
                     Solver solver = &solve, const std::vector<int>& free_images = {});

}  // namespace sfm::ba
