import numpy as np
import cv2
from pathlib import Path
import logging
from tqdm import tqdm
import json
import sqlite3
import scipy.optimize as optimize
from collections import defaultdict
import pandas as pd
from image_selector import SfMGraphSelector

# Constants for reconstruction parameters
MATCHING_THRESHOLD = 2.0  # pixels
MIN_MATCHES = 20  # minimum matches for PnP
PNP_REPROJECTION_ERROR = 8.0
PNP_MIN_INLIERS = 15
RANSAC_ITERATIONS = 1000
BUNDLE_ADJUST_FREQUENCY = 7

class StructureFromMotion:
    def __init__(self, data_dir):
        """
        Initialize SfM with a single camera intrinsic matrix
        
        Args:
            data_dir: Path to data directory
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.data_dir = Path(data_dir)
        self.selector = SfMGraphSelector('/teamspace/studios/this_studio/SFM/bunny_data/matching_results.csv')
        self.selector.visualize_graph()
        
        # Image parameters
        self.image_width = 1024
        self.image_height = 768
        self.constructed = []
        
        # Camera intrinsic matrix for all images
        self.K = np.array([
            [2393.95, 0, 932.38],
            [0, 2398.12, 628.26],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Data directories
        self.matches_dir = self.data_dir / 'matches'
        self.fund_dir = self.data_dir / 'fundamental'
        self.corr_dir = self.data_dir / 'correspondences'
        
        # Initialize reconstruction state
        self.poses = {}  # camera poses {img_id: (R, t)}
        self.points3D = []  # 3D points
        self.point_tracks = []  # 2D-3D correspondences

    def find_best_initial_pair(self, image_pairs):
        """
        Find best initial image pair for reconstruction
        
        Args:
            image_pairs: List of image pair names
            
        Returns:
            best_pair: Name of best initial pair
        """
        best_score = -1
        best_pair = None
        
        for pair in tqdm(image_pairs, desc="Finding best initial pair"):
            try:
                # Load match data
                match_data = np.load(self.matches_dir / f'{pair}_matches.npz')
                fund_data = np.load(self.fund_dir / f'{pair}_F.npz')
                
                num_inliers = np.sum(match_data['inlier_mask'])
                pts1 = fund_data['pts1']
                pts2 = fund_data['pts2']
                
                # Compute essential matrix
                E = self.K.T @ fund_data['F'] @ self.K
                
                # Recover pose
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                
                # Score based on inliers and baseline
                baseline_score = np.linalg.norm(t)
                pair_score = num_inliers * baseline_score
                
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = pair
                    
            except (FileNotFoundError, ValueError) as e:
                logging.warning(f"Skipping pair {pair}: {e}")
                continue
                
        if best_pair is None:
            raise ValueError("Could not find valid initial pair")
            
        logging.info(f"Best initial pair: {best_pair} (score: {best_score:.2f})")
        return best_pair

    def initialize_reconstruction(self, init_pair):
        """
        Initialize reconstruction from initial pair
        
        Args:
            init_pair: Name of initial image pair
        """
        # Load match data
        match_data = np.load(self.matches_dir / f'{init_pair}_matches.npz')
        fund_data = np.load(self.fund_dir / f'{init_pair}_F.npz')
        
        id1, id2 = map(int, init_pair.split('_')[1:])
        
        pts1 = fund_data['pts1'][match_data['inlier_mask']]
        pts2 = fund_data['pts2'][match_data['inlier_mask']]
        
        # Compute essential matrix
        E = self.K.T @ fund_data['F'] @ self.K
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Set first camera as origin
        self.poses[id1] = (np.eye(3), np.zeros((3, 1)))
        self.poses[id2] = (R, t)
        
        # Triangulate initial points
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3D = (pts4D[:3] / pts4D[3]).T
        
        # Store valid points and initialize tracks
        valid_mask = mask.ravel() > 0
        self.points3D = pts3D[valid_mask].tolist()
        
        logging.info(f"Initialized with {len(self.points3D)} 3D points")
        
        # Initialize point tracks
        for i, (pt3d, pt1, pt2) in enumerate(zip(pts3D[valid_mask], 
                                                pts1[valid_mask], 
                                                pts2[valid_mask])):
            track = {
                int(id1): pt1.tolist(),
                int(id2): pt2.tolist()
            }
            self.point_tracks.append(track)

    def find_2d3d_matches(self, image_id):
        """
        Find 2D-3D correspondences for new image efficiently
        
        Args:
            image_id: ID of the new image to add
            
        Returns:
            Tuple of (points3D, points2D) arrays
        """
        points3D = []
        points2D = []
        
        # Pre-convert points3D to numpy array
        points3D_array = np.array(self.points3D)
        
        # Process each pair containing the new image
        image_pairs = self.find_image_pairs(image_id)
        logging.info(f"Found {len(image_pairs)} pairs for image {image_id}")
        
        for pair in image_pairs:
            try:
                # Load correspondence data
                pts1 = np.load(self.corr_dir / f'{pair}_pts1.npy')
                pts2 = np.load(self.corr_dir / f'{pair}_pts2.npy')
                
                # Get image IDs
                id1, id2 = map(int, pair.split('_')[1:])
                
                # Determine which image is the new one
                if id1 == image_id:
                    new_img_pts = pts1
                    other_img_pts = pts2
                    other_img_id = id2
                else:
                    new_img_pts = pts2
                    other_img_pts = pts1
                    other_img_id = id1
                    
                # Find valid tracks containing other image
                valid_track_indices = []
                valid_track_points = []
                
                for track_idx, track in enumerate(self.point_tracks):
                    if other_img_id in track:
                        valid_track_indices.append(track_idx)
                        valid_track_points.append(track[other_img_id])
                
                if not valid_track_indices:
                    continue
                    
                # Vectorized matching
                valid_track_points = np.array(valid_track_points)
                
                # Improved vectorized matching with batch processing
                distances = np.linalg.norm(valid_track_points[:, None] - other_img_pts, axis=2)
                matches = np.where(distances < MATCHING_THRESHOLD)
                
                for track_idx, match_idx in zip(*matches):
                    track_idx = valid_track_indices[track_idx]
                    points3D.append(points3D_array[track_idx])
                    points2D.append(new_img_pts[match_idx])
                
            except (FileNotFoundError, ValueError) as e:
                logging.warning(f"Failed to process pair {pair}: {e}")
                continue
        
        # Convert to numpy arrays
        points3D = np.array(points3D)
        points2D = np.array(points2D)
        
        logging.info(f"Found {len(points3D)} 2D-3D matches for image {image_id}")
        
        return points3D, points2D

    def pnp_ransac(self, points3D, points2D):
        """
        Compute camera pose using PnP RANSAC
        
        Args:
            points3D: 3D points (Nx3)
            points2D: 2D points (Nx2)
            
        Returns:
            Tuple of (R, t, inliers)
        """
        if len(points3D) < PNP_MIN_INLIERS:
            return None, None, None
            
        success, R_vec, t, inliers = cv2.solvePnPRansac(
            points3D.astype(np.float32),
            points2D.astype(np.float32),
            self.K,
            None,
            iterationsCount=RANSAC_ITERATIONS,
            reprojectionError=PNP_REPROJECTION_ERROR,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
            
        R, _ = cv2.Rodrigues(R_vec)
        return R, t, inliers

    def triangulate_point(self, image_points):
        """
        Triangulate 3D point from multiple views
        
        Args:
            image_points: Dict of {image_id: 2D point}
            
        Returns:
            point3D: Triangulated 3D point or None if invalid
        """
        if len(image_points) < 2:
            return None
            
        # Set up projection matrices
        Ps = []
        points = []
        for img_id, point in image_points.items():
            R, t = self.poses[img_id]
            P = self.K @ np.hstack([R, t])
            Ps.append(P)
            points.append(point)
            
        Ps = np.array(Ps)
        points = np.array(points)
        
        # Triangulate
        point4D = cv2.triangulatePoints(Ps[0], Ps[1], 
                                      points[0].reshape(-1, 2).T,
                                      points[1].reshape(-1, 2).T)
        point3D = (point4D[:3] / point4D[3]).ravel()
        
        # Verify reprojection error
        max_error = 4.0  # pixels
        for P, point2D in zip(Ps, points):
            projected = P @ np.append(point3D, 1)
            projected = projected[:2] / projected[2]
            error = np.linalg.norm(projected - point2D)
            if error > max_error:
                return None
                
        return point3D

    def add_new_image(self, image_id):
        """
        Add new image to reconstruction
        
        Args:
            image_id: ID of image to add
            
        Returns:
            success: Whether image was successfully added
        """
        logging.info(f"Attempting to add image {image_id}")
        
        # Find 2D-3D matches
        points3D, points2D = self.find_2d3d_matches(image_id)
        
        if len(points3D) < MIN_MATCHES:
            logging.warning(f"Insufficient 2D-3D matches ({len(points3D)} < {MIN_MATCHES})")
            return False
            
        # Compute pose using PnP
        R, t, inliers = self.pnp_ransac(points3D, points2D)
        
        if R is None:
            logging.warning("PnP pose estimation failed")
            return False
            
        # Store pose
        self.poses[image_id] = (R, t)
        logging.info(f"Successfully estimated pose with {len(inliers)} inliers")
        
        return True

    def add_new_matches(self, pair, image_id):
        """
        Add new matches for image to reconstruction
        
        Args:
            pair: Image pair name
            image_id: ID of new image
        """
        try:
            # Load and ensure proper shape of points
            pts1 = np.load(self.corr_dir / f'{pair}_pts1.npy')
            pts2 = np.load(self.corr_dir / f'{pair}_pts2.npy')
            pts1 = np.asarray(pts1).reshape(-1, 2)
            pts2 = np.asarray(pts2).reshape(-1, 2)
            id1, id2 = map(int, pair.split('_')[1:])

            # Track existing points to avoid duplicates
            existing_track_points = set()
            for track in self.point_tracks:
                for img_id, point in track.items():
                    point_array = np.asarray(point).ravel()
                    existing_track_points.add((img_id, tuple(point_array)))
            
            # Create unique tracks for each point correspondence
            new_tracks = []
            for pt1, pt2 in zip(pts1, pts2):
                track_point1 = (id1, tuple(pt1.ravel()))
                track_point2 = (id2, tuple(pt2.ravel()))
                
                if track_point1 not in existing_track_points and track_point2 not in existing_track_points:
                    track = {
                        id1: pt1.tolist(),
                        id2: pt2.tolist()
                    }
                    new_tracks.append(track)
            
            # Batch triangulate points
            points3D = []
            valid_tracks = []
            
            for track in new_tracks:
                point3D = self.triangulate_point(track)
                if point3D is not None:
                    points3D.append(point3D)
                    valid_tracks.append(track)
            
            if valid_tracks:
                # Add to reconstruction
                self.points3D.extend(points3D)
                self.point_tracks.extend(valid_tracks)
                logging.info(f"Added {len(valid_tracks)} new tracks from pair {pair}")
            else:
                logging.warning(f"No valid tracks found for pair {pair}")
                
        except Exception as e:
            logging.warning(f"Failed to add matches for pair {pair}: {e}")
            return False
            
        return True

    def bundle_adjust(self):
        """Perform bundle adjustment optimization"""
        logging.info("Starting bundle adjustment...")
        
        if len(self.poses) < 2:
            logging.warning("Not enough cameras for bundle adjustment")
            return False
            
        camera_params = []
        points3D = []
        point2D_idxs = []
        camera_idxs = []
        points2D = []
        
        # Camera parameters including intrinsics
        id_to_idx = {}
        for i, (img_id, (R, t)) in enumerate(self.poses.items()):
            id_to_idx[img_id] = i
            rvec, _ = cv2.Rodrigues(R)
            t = np.asarray(t).reshape(3)  # Ensure t is 1D
            intrinsics = [
                self.K[0,0],  # focal length x
                self.K[1,1],  # focal length y
                self.K[0,2],  # principal point x
                self.K[1,2]   # principal point y
            ]
            camera_params.append(np.concatenate([rvec.ravel(), t.ravel(), intrinsics]))
        
        # Prepare optimization data
        for i, (point, track) in enumerate(zip(self.points3D, self.point_tracks)):
            points3D.append(np.asarray(point).ravel())
            for img_id, pt2D in track.items():
                point2D_idxs.append(i)
                camera_idxs.append(id_to_idx[img_id])
                points2D.append(np.asarray(pt2D).ravel())
        
        # Convert to arrays and validate
        try:
            camera_params = np.array(camera_params)
            points3D = np.array(points3D).reshape(-1, 3)
            point2D_idxs = np.array(point2D_idxs)
            camera_idxs = np.array(camera_idxs)
            points2D = np.array(points2D).reshape(-1, 2)
            
            if len(points2D) == 0:
                logging.warning("No points for bundle adjustment")
                return False
                
        except Exception as e:
            logging.error(f"Error preparing bundle adjustment data: {e}")
            return False
        
        def project_points(points3D, camera_params):
            """Project points with variable intrinsics"""
            rvec = camera_params[:3]
            tvec = camera_params[3:6]
            fx, fy, cx, cy = camera_params[6:]
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            R, _ = cv2.Rodrigues(rvec)
            points_homogeneous = np.hstack([points3D, np.ones((points3D.shape[0], 1))])
            P = K @ np.hstack([R, tvec.reshape(3,1)])
            projected = points_homogeneous @ P.T
            projected = projected[:, :2] / projected[:, 2:3]
            return projected
                
        def objective(params):
            """Bundle adjustment objective with regularization"""
            n_cameras = len(self.poses)
            camera_params = params[:n_cameras * 10].reshape((n_cameras, 10))
            points3D = params[n_cameras * 10:].reshape((len(self.points3D), 3))
            
            # Projection error
            all_projections = []
            for cam_idx in np.unique(camera_idxs):
                mask = camera_idxs == cam_idx
                proj = project_points(points3D[point2D_idxs[mask]], camera_params[cam_idx])
                all_projections.append(proj)
                    
            projected = np.vstack(all_projections)
            reproj_error = (projected - points2D).ravel()
            
            # Add regularization for intrinsics
            intrinsics_reg = []
            for params in camera_params:
                fx, fy, cx, cy = params[6:]
                init_fx = self.K[0,0]
                reg = np.array([
                    (fx - init_fx) / init_fx,
                    (fy - fx) / fx,  # enforce fx ≈ fy
                    (cx - self.K[0,2]) / self.image_width,
                    (cy - self.K[1,2]) / self.image_height
                ]) * 0.1  # regularization weight
                intrinsics_reg.append(reg)
            
            return np.concatenate([reproj_error, np.ravel(intrinsics_reg)])
        
        # Run optimization
        x0 = np.hstack([camera_params.ravel(), points3D.ravel()])
        
        res = optimize.least_squares(
            objective, 
            x0,
            method='trf',
            loss='huber',
            max_nfev=100,
            ftol=1e-4,
            xtol=1e-4
        )

        # After optimization
        if not res.success:
            logging.warning(f"Bundle adjustment failed to converge: {res.message}")
            return False

        # Add optimization stats
        cost_initial = np.linalg.norm(objective(x0))
        cost_final = np.linalg.norm(objective(res.x))
        logging.info(f"Bundle adjustment: cost reduced from {cost_initial:.2f} to {cost_final:.2f}")
        
        # Update parameters
        n_cameras = len(self.poses)
        camera_params = res.x[:n_cameras * 10].reshape((n_cameras, 10))
        points3D_new = res.x[n_cameras * 10:].reshape((len(self.points3D), 3))
        
        # Update camera matrices
        self.K = np.mean([
            np.array([
                [params[6], 0, params[8]],
                [0, params[7], params[9]],
                [0, 0, 1]
            ]) for params in camera_params
        ], axis=0)
        
        # Update poses and points
        for img_id, idx in id_to_idx.items():
            rvec = camera_params[idx, :3]
            tvec = camera_params[idx, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            self.poses[img_id] = (R, tvec)
        
        self.points3D = points3D_new.tolist()
        
        logging.info("Bundle adjustment completed")
        
    def find_image_pairs(self, image_id):
        """
        Find all pairs containing image_id
        
        Args:
            image_id: Target image ID
            
        Returns:
            pairs: List of pair names
        """
        pairs = []
        for path in self.matches_dir.glob('*.npz'):
            pair = path.stem
            
            try:
                if pair.endswith('_matches'):
                    pair = pair.replace('_matches', '')
                
                if not pair.startswith('pair_'):
                    continue
                
                id1, id2 = map(int, pair.split('_')[1:3])
                
                if ((id1 == image_id and f"{id2:04d}.ppm" in self.constructed) or 
                    (id2 == image_id and f"{id1:04d}.ppm" in self.constructed)):
                    pairs.append(pair)
            except (ValueError, IndexError):
                logging.warning(f"Skipping file with unexpected name: {path}")
        
        return pairs
        
    def compute_reconstruction_stats(self):
        """
        Compute reconstruction quality statistics
        
        Returns:
            stats: Dictionary of reconstruction statistics
        """
        reproj_errors = []
        track_lengths = []
        
        for point3D, track in zip(self.points3D, self.point_tracks):
            # Convert point3D to numpy array if it isn't already
            point3D = np.array(point3D)
            
            # Compute reprojection errors
            for img_id, point2D in track.items():
                R, t = self.poses[img_id]
                # Ensure t is 2D array
                t = np.array(t).reshape(3, 1)
                P = self.K @ np.hstack([R, t])
                # Convert point2D to numpy array
                point2D = np.array(point2D)
                # Add homogeneous coordinate
                point3D_h = np.append(point3D, 1)
                projected = P @ point3D_h
                projected = projected[:2] / projected[2]
                error = np.linalg.norm(projected - point2D)
                reproj_errors.append(error)
            track_lengths.append(len(track))
        
        if not reproj_errors:
            return {
                'mean_reproj_error': 0,
                'max_reproj_error': 0,
                'mean_track_length': 0,
                'max_track_length': 0,
                'num_points': len(self.points3D),
                'num_cameras': len(self.poses)
            }
        
        stats = {
            'mean_reproj_error': float(np.mean(reproj_errors)),
            'max_reproj_error': float(np.max(reproj_errors)),
            'mean_track_length': float(np.mean(track_lengths)),
            'max_track_length': float(np.max(track_lengths)),
            'num_points': len(self.points3D),
            'num_cameras': len(self.poses)
        }
        
        return stats

    def run_reconstruction(self, num_images):
        """
        Run complete reconstruction pipeline
        
        Args:
            num_images: Number of images to process
        """
        logging.info("Starting reconstruction...")
        
        # Load match data
        match_data_df = pd.read_csv("/teamspace/studios/this_studio/SFM/bunny_data/matching_results.csv")
        image_pairs = match_data_df['pair_name'].tolist()
        
        if not image_pairs:
            raise ValueError("No image pairs found!")
            
        # Find and initialize from best pair
        init_pair = self.find_best_initial_pair(image_pairs)
        self.initialize_reconstruction(init_pair)
        self.constructed = [f"{i:04d}.ppm" for i in list(self.poses.keys())]
        
        # Add remaining images
        remaining_images = set(range(0, num_images)) - set(self.poses.keys())
        
        while remaining_images:
            logging.info(f"Current reconstruction has {len(self.constructed)} images")
            
            # Get next best images
            next_best_images = self.selector.find_next_best_images(self.constructed, num_images)
            
            if not next_best_images:
                logging.warning("No next best images available")
                break
                
            image_added = False
            candidate_images = [int(img[:4]) for img in next_best_images]
            
            for image_id in candidate_images:
                if image_id not in remaining_images:
                    continue
                
                success = self.add_new_image(image_id)
                
                if success:
                    # Update construction
                    self.constructed.append(f"{image_id:04d}.ppm")
                    
                    # Add new matches
                    image_pairs = self.find_image_pairs(image_id)
                    for pair in image_pairs:
                        self.add_new_matches(pair, image_id)
                    
                    remaining_images.remove(image_id)
                    image_added = True
                    
                    # Periodic bundle adjustment
                    if len(self.poses) % BUNDLE_ADJUST_FREQUENCY == 0:
                        self.bundle_adjust()
                        stats = self.compute_reconstruction_stats()
                        logging.info(f"Reconstruction stats: {stats}")
                    
                    break
            
            if not image_added:
                logging.warning("Failed to add any new images")
                break
        
        # Final bundle adjustment
        if len(self.poses) > 2:
            self.bundle_adjust()
        
        # Final statistics
        stats = self.compute_reconstruction_stats()
        logging.info("Reconstruction complete!")
        logging.info(f"Final statistics: {stats}")
        if remaining_images:
            logging.warning(f"Failed to reconstruct {len(remaining_images)} images: {sorted(remaining_images)}")
            
    def save_reconstruction(self, output_dir):
        """Save reconstruction results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save camera poses
        poses_dict = {}
        for img_id, (R, t) in self.poses.items():
            poses_dict[str(img_id)] = {
                'R': R.tolist(),
                't': t.ravel().tolist()
            }
            
        with open(output_dir / 'poses.json', 'w') as f:
            json.dump(poses_dict, f, indent=2)
            
        # Convert points3D to list if it's numpy array
        points3D_list = [p.tolist() if isinstance(p, np.ndarray) else p for p in self.points3D]
        
        # Convert tracks ensuring all numpy arrays are converted to lists
        tracks_list = []
        for track in self.point_tracks:
            track_dict = {}
            for img_id, point in track.items():
                track_dict[str(img_id)] = point.tolist() if isinstance(point, np.ndarray) else point
            tracks_list.append(track_dict)
        
        # Save 3D points and tracks
        points_dict = {
            'points3D': points3D_list,
            'tracks': tracks_list
        }
        
        with open(output_dir / 'points3D.json', 'w') as f:
            json.dump(points_dict, f, indent=2)
            
        # Save as PLY
        self.save_ply(output_dir / 'reconstruction.ply')
        logging.info(f"Saved reconstruction to {output_dir}")
        
    def save_ply(self, filepath):
        """Save reconstruction as PLY file"""
        points = np.array(self.points3D)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize and run SfM
    sfm = StructureFromMotion("/teamspace/studios/this_studio/SFM/bunny_data")
    sfm.run_reconstruction(num_images=36)
    
    # Save results
    sfm.save_reconstruction("/teamspace/studios/this_studio/SFM/bunny_data/reconstruction")

if __name__ == "__main__":
    main()