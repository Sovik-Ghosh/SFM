import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from itertools import combinations
from scipy.spatial import cKDTree
from .bundle_adjustment import run_bundle_adjustment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustSfM:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.matches_dir = self.data_dir / 'matches'
        self.fund_dir = self.data_dir / 'fundamental'
        self.corr_dir = self.data_dir / 'correspondences'
        
        # Parameters for matching and reconstruction
        self.min_inliers = 50
        self.min_inlier_ratio = 0.4
        self.min_matches = 100
        self.good_inlier_ratio = 0.7
        self.bundle_adjustment_frequency = 5
        
        # Parameters for PnP
        self.min_tracked_points = 10
        self.pnp_reproj_threshold = 16.0
        self.pnp_confidence = 0.99
        self.pnp_min_inlier_ratio = 0.3
        
        # Parameters for point filtering
        self.min_triangulation_angle = np.radians(2.0)
        self.max_point_distance = 100.0
        self.min_point_distance = 1e-3
        
        # Initialize reconstruction data
        self.reconstruction = {
            'cameras': [],
            'points_3d': [],
            'image_points': [],
            'images': [],
            'camera_matrix': np.array([[2.8478542e+03, 0.0000000e+00, 1.7340000e+03],
                              [0.0000000e+00, 5.0982563e+03, 2.3120000e+03],
                              [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
        }
        
        # Track points across views
        self.point_tracks = defaultdict(dict)  # track_id -> {image_id: point_2d}
        self.image_graph = nx.Graph()
        self.registered_images = set()
        
        # Statistics tracking
        self.stats = {
            'added_views': [],
            'point_counts': [],
            'reprojection_errors': [],
            'processing_times': [],
            'track_lengths': []
        }
        
        # Create directories if they don't exist
        for dir_path in [self.matches_dir, self.fund_dir, self.corr_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def predict_next_pose(self, last_poses: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Predict next camera pose based on motion model"""
        if len(last_poses) < 2:
            return None, None
            
        # Estimate motion from last two cameras
        R1 = last_poses[-2][:3, :3]
        t1 = last_poses[-2][:3, 3]
        R2 = last_poses[-1][:3, :3]
        t2 = last_poses[-1][:3, 3]
        
        # Calculate relative motion
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        # Predict next pose
        R_pred = R_rel @ R2
        t_pred = R_rel @ t2 + t_rel
        
        return R_pred, t_pred

    def verify_camera_pose(self, R: np.ndarray, t: np.ndarray, 
                          points_3d: np.ndarray, points_2d: np.ndarray) -> bool:
        """Verify camera pose by checking point depths and reprojection"""
        # Transform points to camera frame
        points_cam = points_3d @ R.T + t.T
        
        # Check point depths
        depth_percentile = np.percentile(points_cam[:, 2], 75)
        if depth_percentile < self.min_point_distance:
            return False
            
        # Check reprojection
        projected = cv2.projectPoints(points_3d, cv2.Rodrigues(R)[0], t, 
                                    self.reconstruction['camera_matrix'], None)[0][:, 0, :]
        errors = np.linalg.norm(projected - points_2d, axis=1)
        median_error = np.median(errors)
        
        # Track statistics
        self.stats['reprojection_errors'].append(median_error)
        
        return median_error < self.pnp_reproj_threshold

    def check_scale_consistency(self, new_points_3d: np.ndarray, 
                              existing_points_3d: np.ndarray) -> bool:
        """Check if new points maintain consistent scale"""
        if len(new_points_3d) < 10 or len(existing_points_3d) < 10:
            return True
            
        # Sample points if there are too many
        max_points = 100
        if len(new_points_3d) > max_points:
            indices = np.random.choice(len(new_points_3d), max_points, replace=False)
            new_points_sample = new_points_3d[indices]
        else:
            new_points_sample = new_points_3d
            
        if len(existing_points_3d) > max_points:
            indices = np.random.choice(len(existing_points_3d), max_points, replace=False)
            existing_points_sample = existing_points_3d[indices]
        else:
            existing_points_sample = existing_points_3d
        
        # Calculate scale using point distances
        new_scale = np.median([np.linalg.norm(p1 - p2) 
                              for p1, p2 in combinations(new_points_sample, 2)])
        old_scale = np.median([np.linalg.norm(p1 - p2) 
                              for p1, p2 in combinations(existing_points_sample, 2)])
                              
        ratio = new_scale / old_scale
        return 0.1 < ratio < 10
    
    def load_pair_data(self, pair_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load precomputed pair data"""
        try:
            pts1 = np.load(self.corr_dir / f'{pair_name}_pts1.npy')
            pts2 = np.load(self.corr_dir / f'{pair_name}_pts2.npy')
            F_data = np.load(self.fund_dir / f'{pair_name}_F.npz')
            matches_data = np.load(self.matches_dir / f'{pair_name}_matches.npz')
            
            # Verify data integrity
            if len(pts1) != len(pts2):
                raise ValueError(f"Mismatched point counts: {len(pts1)} vs {len(pts2)}")
                
            return pts1, pts2, F_data['F'], matches_data['inlier_mask']
        except Exception as e:
            logger.error(f"Failed to load data for {pair_name}: {e}")
            raise

    def build_covisibility_graph(self, matches_df: pd.DataFrame) -> nx.Graph:
        """Build graph representing image connectivity strength"""
        G = nx.Graph()
        
        for _, row in matches_df.iterrows():
            id1 = int(row['img1'].split('.')[0])
            id2 = int(row['img2'].split('.')[0])
            
            # Edge weight combines number of inliers and inlier ratio
            weight = row['num_inliers'] * row['inlier_ratio']
            
            if row['num_inliers'] >= self.min_inliers and \
               row['inlier_ratio'] >= self.min_inlier_ratio:
                G.add_edge(id1, id2, 
                          weight=weight,
                          num_inliers=row['num_inliers'],
                          inlier_ratio=row['inlier_ratio'],
                          well_distributed=row.get('well_distributed', True))
        
        logger.info(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G

    def initialize_reconstruction(self, id1: int, id2: int):
        """Initialize reconstruction from first pair"""
        logger.info(f"Initializing with pair {id1}-{id2}")
        
        try:
            # Load pair data
            pair_name = f'pair_{min(id1,id2)}_{max(id1,id2)}'
            pts1, pts2, F, inlier_mask = self.load_pair_data(pair_name)
            
            # Use only inlier points
            if inlier_mask is not None:
                pts1 = pts1[:min(len(pts1), len(inlier_mask))]
                pts2 = pts2[:min(len(pts1), len(inlier_mask))]
                inlier_mask = inlier_mask[:len(pts1)]
                pts1 = pts1[inlier_mask]
                pts2 = pts2[inlier_mask]
            
            logger.info(f"Using {len(pts1)} points for initialization")
            
            # Normalize image coordinates
            K = self.reconstruction['camera_matrix']
            K_inv = np.linalg.inv(K)
            
            # Convert to normalized coordinates
            pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
            pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)
            pts1_norm = pts1_norm.reshape(-1, 2)
            pts2_norm = pts2_norm.reshape(-1, 2)
            
            # Estimate Essential matrix directly from normalized coordinates
            E, E_mask = cv2.findEssentialMat(pts1_norm, pts2_norm, 
                                            np.eye(3), method=cv2.RANSAC, 
                                            prob=0.999, threshold=3.0/K[0,0])
            
            # Recover pose from Essential matrix using normalized coordinates
            _, R, t, pose_mask = cv2.recoverPose(E, pts1_norm[E_mask.ravel() == 1], 
                                                pts2_norm[E_mask.ravel() == 1], np.eye(3))
            
            logger.info(f"RecoverPose found {np.sum(pose_mask)} inliers")
            
            # Set up projection matrices (in normalized coordinates)
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = np.hstack((R, t))
            
            # Use only inliers from E-matrix estimation
            valid_pts1 = pts1_norm[E_mask.ravel() == 1]
            valid_pts2 = pts2_norm[E_mask.ravel() == 1]
            
            # Triangulate points using normalized coordinates
            points_4d = cv2.triangulatePoints(P1, P2, valid_pts1.T, valid_pts2.T)
            
            # Convert to 3D points
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Filter points based on cheirality (points must be in front of both cameras)
            valid_points = np.ones(len(points_3d), dtype=bool)
            
            # Check depths in both cameras
            depths1 = points_3d[:, 2]
            depths2 = (points_3d @ R.T + t.T)[:, 2]
            valid_points &= (depths1 > 0) & (depths2 > 0)
            
            logger.info(f"Points after cheirality check: {np.sum(valid_points)}/{len(points_3d)}")
            
            # Filter out points too far from cameras
            distances = np.linalg.norm(points_3d, axis=1)
            median_dist = np.median(distances)
            valid_points &= (distances < median_dist * 3)
            
            logger.info(f"Points after distance check: {np.sum(valid_points)}/{len(points_3d)}")
            
            # Apply filtering
            points_3d = points_3d[valid_points]
            valid_pts1 = valid_pts1[valid_points]
            valid_pts2 = valid_pts2[valid_points]
            
            # Scale reconstruction to have median distance of 1
            scale = 1.0 / median_dist
            points_3d *= scale
            t *= scale
            P2[:, 3] = t.ravel()
            
            # Convert points back to pixel coordinates for storage
            valid_pts1_pixel = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), 
                                            K, None)[0][:, 0, :]
            valid_pts2_pixel = cv2.projectPoints(points_3d, cv2.Rodrigues(R)[0], t, 
                                            K, None)[0][:, 0, :]
            
            if len(points_3d) < self.min_tracked_points:
                raise ValueError(f"Insufficient points after filtering: {len(points_3d)} < {self.min_tracked_points}")
            
            # Initialize reconstruction
            self.reconstruction.update({
                'cameras': [P1, P2],
                'points_3d': points_3d,
                'image_points': [valid_pts1_pixel, valid_pts2_pixel],
                'images': [id1, id2]
            })
            
            # Initialize tracks
            self.point_tracks = defaultdict(dict)
            for i in range(len(points_3d)):
                self.point_tracks[i] = {
                    id1: valid_pts1_pixel[i],
                    id2: valid_pts2_pixel[i]
                }
            
            self.registered_images.update([id1, id2])
            logger.info(f"Initialized reconstruction with {len(points_3d)} points")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
    
    def find_next_images(self, G: nx.Graph, registered: Set[int], num_candidates: int = 3) -> List[Tuple[int, int]]:
        """Find next best images to add to reconstruction"""
        candidates = []
        
        # Find all edges connecting registered to unregistered images
        for reg_id in registered:
            for neighbor_id in G[reg_id]:
                if neighbor_id not in registered:
                    edge_data = G[reg_id][neighbor_id]
                    score = edge_data['num_inliers'] * edge_data['inlier_ratio']
                    
                    # Count strong connections to registered images
                    strong_connections = sum(
                        1 for other_reg in registered
                        if neighbor_id in G[other_reg] and 
                        G[other_reg][neighbor_id]['inlier_ratio'] > self.good_inlier_ratio
                    )
                    
                    # Prefer sequential connections
                    sequential_bonus = any(abs(neighbor_id - reg) == 1 
                                        for reg in registered)
                    
                    # Additional score for well-distributed points
                    distribution_score = (1.2 if edge_data.get('well_distributed', True) 
                                    else 1.0)
                    
                    total_score = (score * 
                                (1.2 if sequential_bonus else 1.0) * 
                                distribution_score)
                    
                    candidates.append({
                        'new_id': neighbor_id,
                        'ref_id': reg_id,
                        'score': total_score,
                        'strong_connections': strong_connections,
                        'inlier_ratio': edge_data['inlier_ratio'],
                        'num_inliers': edge_data['num_inliers']
                    })
        
        if not candidates:
            return []
        
        # Sort candidates by strong connections first, then by score
        candidates.sort(key=lambda x: (-x['strong_connections'], -x['score']))
        
        # Log candidate details
        logger.info(f"Found {len(candidates)} candidates for next view")
        for i, c in enumerate(candidates[:num_candidates]):
            logger.info(f"Candidate {i+1}:")
            logger.info(f"- New ID: {c['new_id']}, Reference ID: {c['ref_id']}")
            logger.info(f"- Strong connections: {c['strong_connections']}")
            logger.info(f"- Inlier ratio: {c['inlier_ratio']:.3f}")
            logger.info(f"- Number of inliers: {c['num_inliers']}")
            logger.info(f"- Total score: {c['score']:.3f}")
        
        return [(c['new_id'], c['ref_id']) for c in candidates[:num_candidates]]
    
    def add_view_robust(self, new_id: int, ref_id: int, G: nx.Graph) -> bool:
        """Add a new view to the reconstruction robustly using PnP-RANSAC"""
        logger.info(f"Attempting to add view {new_id} using reference {ref_id}")
        start_time = time.time()
        
        try:
            # Load pair data
            pair_name = f'pair_{min(new_id, ref_id)}_{max(new_id, ref_id)}'
            pts1, pts2, _, inlier_mask = self.load_pair_data(pair_name)
            logger.info(f"Loaded {len(pts1)} point correspondences")
            
            # Use only inlier points
            if inlier_mask is not None:
                if len(pts1) != len(inlier_mask):
                    valid_size = min(len(pts1), len(inlier_mask))
                    pts1 = pts1[:valid_size]
                    pts2 = pts2[:valid_size]
                    inlier_mask = inlier_mask[:valid_size]
                pts1 = pts1[inlier_mask]
                pts2 = pts2[inlier_mask]
                logger.info(f"Using {len(pts1)} inlier points")
            
            # Get reference camera index and points
            ref_idx = self.reconstruction['images'].index(ref_id)
            ref_points = self.reconstruction['image_points'][ref_idx]
            ref_cam = self.reconstruction['cameras'][ref_idx]
            logger.info(f"Reference view has {len(ref_points)} points")
            
            # Find 2D-3D correspondences using KD-tree
            ref_points_tree = cKDTree(ref_points)
            pt_dists, pt_indices = ref_points_tree.query(pts1, k=2, distance_upper_bound=2.0)
            
            points_3d = []
            points_2d = []
            valid_indices = []
            
            # Use ratio test for matching
            for i, (dists, idxs) in enumerate(zip(pt_dists, pt_indices)):
                if dists[0] != np.inf and idxs[0] < len(self.reconstruction['points_3d']):
                    # Check if match is unique (ratio test)
                    if dists[1] == np.inf or dists[0] < 0.9 * dists[1]:
                        points_3d.append(self.reconstruction['points_3d'][idxs[0]])
                        points_2d.append(pts2[i])
                        valid_indices.append(idxs[0])
            
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)
            
            # Debug matching
            logger.info(f"After KD-tree matching:")
            logger.info(f"- Total points: {len(pts1)}")
            logger.info(f"- Matched points: {len(points_3d)}")
            if len(points_3d) > 0:
                logger.info(f"- Average matching distance: {np.mean(pt_dists[pt_dists != np.inf]):.3f}")
            
            if len(points_3d) < self.min_tracked_points:
                logger.warning(f"Insufficient correspondences: {len(points_3d)} < {self.min_tracked_points}")
                return False
            
            # Normalize points for numerical stability
            pts_mean = np.mean(points_3d, axis=0)
            pts_scale = np.max(np.abs(points_3d - pts_mean))
            points_3d_normalized = (points_3d - pts_mean) / pts_scale
            
            # Predict pose from motion model
            R_pred, t_pred = self.predict_next_pose(self.reconstruction['cameras'])
            use_guess = R_pred is not None and t_pred is not None
            
            # PnP-RANSAC
            try:
                if use_guess:
                    # Use motion model prediction
                    rvec_guess = cv2.Rodrigues(R_pred)[0]
                    tvec_guess = t_pred.reshape(3, 1)
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        points_3d_normalized,
                        points_2d,
                        self.reconstruction['camera_matrix'],
                        None,
                        useExtrinsicGuess=True,
                        rvec=rvec_guess,
                        tvec=tvec_guess,
                        confidence=0.9999,
                        reprojectionError=self.pnp_reproj_threshold,
                        flags=cv2.SOLVEPNP_EPNP,
                        iterationsCount=2000
                    )
                else:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        points_3d_normalized,
                        points_2d,
                        self.reconstruction['camera_matrix'],
                        None,
                        confidence=0.9999,
                        reprojectionError=self.pnp_reproj_threshold,
                        flags=cv2.SOLVEPNP_EPNP,
                        iterationsCount=2000
                    )
                
                if not success or inliers is None:
                    logger.warning(f"PnP failed for view {new_id}")
                    return False
                
                # Denormalize translation
                tvec = tvec * pts_scale + pts_mean.reshape(3, 1)
                
                # Refine pose using all inliers
                success, rvec, tvec = cv2.solvePnP(
                    points_3d[inliers[:, 0]],
                    points_2d[inliers[:, 0]],
                    self.reconstruction['camera_matrix'],
                    None,
                    rvec,
                    tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
            except cv2.error as e:
                logger.error(f"OpenCV error during PnP: {str(e)}")
                return False
            
            # Check inlier ratio
            inlier_ratio = len(inliers) / len(points_3d)
            if inlier_ratio < self.pnp_min_inlier_ratio:
                logger.warning(f"Low inlier ratio: {inlier_ratio:.3f} < {self.pnp_min_inlier_ratio}")
                return False
            
            # Convert rotation vector to matrix and verify pose
            R, _ = cv2.Rodrigues(rvec)
            if not self.verify_camera_pose(R, tvec, points_3d[inliers[:, 0]], points_2d[inliers[:, 0]]):
                logger.warning("Camera pose verification failed")
                return False
            
            new_camera = np.hstack([R, tvec])
            
            # Track existing points
            point_map = {tuple(pt.tolist()): i for i, pt in enumerate(ref_points)}
            for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                pt_key = tuple(pt1.tolist())
                if pt_key in point_map:
                    pt_idx = point_map[pt_key]
                    if pt_idx < len(self.point_tracks):
                        self.point_tracks[pt_idx][new_id] = pt2
            
            # Triangulate new points
            valid_new_points = []
            valid_new_refs = []
            
            # Get camera centers
            ref_center = -ref_cam[:3, :3].T @ ref_cam[:3, 3]
            new_center = -new_camera[:3, :3].T @ new_camera[:3, 3]
            
            for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                pt_key = tuple(pt1.tolist())
                if pt_key not in point_map:
                    pts4D = cv2.triangulatePoints(ref_cam, new_camera,
                                                pt1.reshape(-1, 1, 2),
                                                pt2.reshape(-1, 1, 2))
                    new_point = (pts4D[:3] / pts4D[3]).reshape(3)
                    
                    # Check triangulation angle using camera centers
                    ray1 = new_point - ref_center
                    ray2 = new_point - new_center
                    
                    ray1 = ray1 / np.linalg.norm(ray1)
                    ray2 = ray2 / np.linalg.norm(ray2)
                    
                    cos_angle = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Validate point
                    if (angle > self.min_triangulation_angle and 
                        np.abs(new_point).max() < self.max_point_distance):
                        valid_new_points.append(new_point)
                        valid_new_refs.append((pt1, pt2))
            
            if valid_new_points:
                valid_new_points = np.array(valid_new_points)
                if not self.check_scale_consistency(valid_new_points, self.reconstruction['points_3d']):
                    logger.warning("Scale consistency check failed")
                    return False
                
                # Add new points and update tracks
                start_idx = len(self.reconstruction['points_3d'])
                self.reconstruction['points_3d'] = np.vstack([
                    self.reconstruction['points_3d'],
                    valid_new_points
                ])
                
                for i, (pt1, pt2) in enumerate(valid_new_refs):
                    track_id = start_idx + i
                    self.point_tracks[track_id] = {
                        ref_id: pt1,
                        new_id: pt2
                    }
                
                logger.info(f"Added {len(valid_new_points)} new 3D points")
            
            # Update reconstruction
            self.reconstruction['cameras'].append(new_camera)
            self.reconstruction['images'].append(new_id)
            self.reconstruction['image_points'].append(pts2)
            self.registered_images.add(new_id)
            
            # Update statistics
            process_time = time.time() - start_time
            self.stats['processing_times'].append(process_time)
            self.stats['point_counts'].append(len(self.reconstruction['points_3d']))
            self.stats['added_views'].append(new_id)
            
            logger.info(f"Successfully added view {new_id}")
            logger.info(f"- Inliers: {len(inliers)} ({inlier_ratio:.3f} ratio)")
            logger.info(f"- Processing time: {process_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding view {new_id}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return False

    def find_best_initial_component(self, G: nx.Graph) -> List[int]:
        """Find best connected component to start reconstruction"""
        components = list(nx.connected_components(G))
        best_score = -1
        best_component = None
        
        logger.info(f"Found {len(components)} connected components")
        
        for component in components:
            subgraph = G.subgraph(component)
            
            # Score based on number of strong connections and sequential pairs
            strong_edges = sum(1 for _, _, data in subgraph.edges(data=True)
                             if data['inlier_ratio'] > self.good_inlier_ratio)
            
            # Prefer components with sequential pairs
            sequential_pairs = sum(1 for u, v in subgraph.edges()
                                 if abs(u - v) == 1)
            
            # Component score combines size, strong connections, and sequential pairs
            score = (strong_edges * len(component) + 
                    sequential_pairs * 2 +  # Give extra weight to sequential pairs
                    len(component) * 0.5)   # Small bonus for component size
            
            if score > best_score:
                best_score = score
                best_component = component
                
        if best_component is None:
            raise ValueError("No suitable components found")
            
        logger.info(f"Selected component with {len(best_component)} images and score {best_score:.2f}")
        return sorted(best_component)

    def find_best_initial_pair(self, G: nx.Graph, component: List[int]) -> Tuple[int, int]:
        """Find best pair to start reconstruction"""
        pairs = []
        for id1 in component:
            for id2 in G[id1]:
                if id2 in component:
                    edge_data = G[id1][id2]
                    
                    # Score combines multiple factors
                    feature_score = edge_data['num_inliers'] * edge_data['inlier_ratio']
                    baseline_score = 1.0 / (1.0 + abs(id1 - id2))  # Prefer sequential pairs
                    well_distributed = edge_data.get('well_distributed', True)
                    distribution_score = 1.2 if well_distributed else 1.0
                    
                    total_score = feature_score * baseline_score * distribution_score
                    pairs.append((total_score, (id1, id2)))
        
        if not pairs:
            raise ValueError("No suitable initial pairs found")
        
        best_pair = max(pairs, key=lambda x: x[0])[1]
        logger.info(f"Selected initial pair {best_pair}")
        return best_pair

    def reconstruct(self, matches_csv: str) -> Dict:
        """Run robust incremental SfM"""
        logger.info("Starting reconstruction...")
        start_time = time.time()
        
        try:
            # Load matches and build graph
            matches_df = pd.read_csv(matches_csv)
            G = self.build_covisibility_graph(matches_df)
            
            if len(G.nodes()) == 0:
                raise ValueError("No valid image pairs found")
            
            # Find best starting component and pair
            best_component = self.find_best_initial_component(G)
            id1, id2 = self.find_best_initial_pair(G, best_component)
            
            # Initialize reconstruction
            self.initialize_reconstruction(id1, id2)
            
            # Incremental reconstruction
            n_consecutive_failures = 0
            max_consecutive_failures = 3
            total_images = len(G.nodes())
            
            with tqdm(desc="Adding views", total=total_images-2) as pbar:
                while True:
                    next_images = self.find_next_images(G, self.registered_images)
                    if not next_images:
                        logger.info("No more images to add")
                        break
                    
                    success = False
                    for new_id, ref_id in next_images:
                        if self.add_view_robust(new_id, ref_id, G):
                            success = True
                            n_consecutive_failures = 0
                            pbar.update(1)
                            
                            # Global bundle adjustment every N views
                            if len(self.reconstruction['images']) % self.bundle_adjustment_frequency == 0:
                                logger.info("Running bundle adjustment...")
                                try:
                                    self.reconstruction = run_bundle_adjustment(self.reconstruction)
                                    logger.info("Bundle adjustment completed")
                                except Exception as e:
                                    logger.error(f"Bundle adjustment failed: {e}")
                            break
                    
                    if not success:
                        n_consecutive_failures += 1
                        if n_consecutive_failures >= max_consecutive_failures:
                            logger.warning(f"Stopping after {max_consecutive_failures} consecutive failures")
                            break
            
            # Final global bundle adjustment
            logger.info("Running final bundle adjustment...")
            try:
                self.reconstruction = run_bundle_adjustment(self.reconstruction)
            except Exception as e:
                logger.error(f"Final bundle adjustment failed: {e}")
            
            # Calculate reconstruction statistics
            total_time = time.time() - start_time
            n_cameras = len(self.reconstruction['images'])
            n_points = len(self.reconstruction['points_3d'])
            avg_track_length = np.mean([len(track) for track in self.point_tracks.values()])
            avg_reproj_error = np.mean(self.stats['reprojection_errors'])
            
            logger.info("Reconstruction completed:")
            logger.info(f"- Total time: {total_time:.2f} seconds")
            logger.info(f"- Cameras: {n_cameras}/{total_images} ({n_cameras/total_images*100:.1f}%)")
            logger.info(f"- 3D points: {n_points}")
            logger.info(f"- Average track length: {avg_track_length:.2f}")
            logger.info(f"- Average reprojection error: {avg_reproj_error:.2f}")
            
            return self.reconstruction
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

    def save_reconstruction(self, output_dir: Path):
        """Save reconstruction results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cameras and points
        np.savez(output_dir / 'reconstruction.npz',
                cameras=self.reconstruction['cameras'],
                points_3d=self.reconstruction['points_3d'],
                images=self.reconstruction['images'],
                camera_matrix=self.reconstruction['camera_matrix'])
        
        # Save statistics
        stats_df = pd.DataFrame({
            'processing_times': self.stats['processing_times'],
            'point_counts': self.stats['point_counts'],
            'reprojection_errors': self.stats['reprojection_errors'],
            'track_lengths': self.stats['track_lengths']
        })
        stats_df.to_csv(output_dir / 'reconstruction_stats.csv', index=False)
        
        logger.info(f"Reconstruction saved to {output_dir}")

def main():
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path("data")
    output_dir = Path("results")
    
    try:
        # Initialize reconstruction
        sfm = RobustSfM(data_dir)
        
        # Run reconstruction
        reconstruction = sfm.reconstruct(data_dir / "pair_matches.csv")
        
        # Save results
        sfm.save_reconstruction(output_dir)
        
    except Exception as e:
        logger.error(f"Failed to run reconstruction: {e}")
        raise

if __name__ == "__main__":
    main()