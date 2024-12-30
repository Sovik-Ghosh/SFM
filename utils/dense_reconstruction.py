import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DenseReconstruction:
    def __init__(self, reconstruction, image_dir):
        self.reconstruction = reconstruction
        self.image_dir = Path(image_dir)
        self.depth_maps = {}
        
        # More lenient parameters
        self.min_depth = 0.01  # Reduced minimum depth
        self.max_depth = 1000.0  # Increased maximum depth
        self.min_points_for_reconstruction = 100  # Reduced minimum points
        
        # SGBM parameters tuned for more points
        self.sgbm_params = {
            'minDisparity': 0,
            'numDisparities': 256,  # Increased
            'blockSize': 7,
            'P1': 8 * 3 * 7**2,
            'P2': 32 * 3 * 7**2,
            'disp12MaxDiff': 2,
            'uniquenessRatio': 5,  # Reduced
            'speckleWindowSize': 200,
            'speckleRange': 64
        }
        
        # Point cloud parameters
        self.min_depth = 0.1
        self.max_depth = 100.0
        self.min_points_for_reconstruction = 1000
        
        logger.info(f"Initialized dense reconstruction with {len(reconstruction['images'])} images")
        
    def compute_depth_maps(self):
        """Compute depth maps for all images using Semi-Global Block Matching"""
        logger.info("Computing depth maps...")
        
        for i, img_id in enumerate(self.reconstruction['images']):
            # Construct image path
            img_path = self.image_dir / f"{img_id:06d}.jpg"
            if not img_path.exists():
                logger.warning(f"Image {img_path} not found")
                continue
            
            # Read reference image
            ref_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                logger.warning(f"Failed to read image {img_path}")
                continue
                
            # Find best neighbors for stereo matching
            neighbors = self._find_best_neighbors(i, n_neighbors=2)
            if not neighbors:
                logger.warning(f"No suitable neighbors found for image {img_id}")
                continue
            
            depth_maps = []
            for neighbor_idx in neighbors:
                # Get neighbor image
                neighbor_id = self.reconstruction['images'][neighbor_idx]
                neighbor_path = self.image_dir / f"{neighbor_id:06d}.jpg"
                
                if not neighbor_path.exists():
                    continue
                    
                neighbor_img = cv2.imread(str(neighbor_path), cv2.IMREAD_GRAYSCALE)
                if neighbor_img is None:
                    continue
                    
                # Ensure images are same size
                if neighbor_img.shape != ref_img.shape:
                    neighbor_img = cv2.resize(neighbor_img, ref_img.shape[::-1])
                
                # Compute relative pose
                R_rel, t_rel = self._compute_relative_pose(i, neighbor_idx)
                
                # Rectify images
                try:
                    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                        self.reconstruction['camera_matrix'],
                        None,  # No distortion
                        self.reconstruction['camera_matrix'],
                        None,  # No distortion
                        ref_img.shape[::-1],
                        R_rel,
                        t_rel
                    )
                    
                    # Create stereo matcher
                    stereo = cv2.StereoSGBM_create(**self.sgbm_params)
                    
                    # Compute disparity
                    disparity = stereo.compute(ref_img, neighbor_img).astype(np.float32) / 16.0
                    
                    # Filter invalid disparities
                    disparity[disparity < 0] = 0
                    
                    # Convert disparity to depth
                    depth = cv2.reprojectImageTo3D(disparity, Q)
                    depth_maps.append(depth)
                    
                except cv2.error as e:
                    logger.warning(f"Failed to compute depth map for pair {img_id}-{neighbor_id}: {e}")
                    continue
            
            if depth_maps:
                # Fuse depth maps
                self.depth_maps[i] = self._fuse_depth_maps(depth_maps)
                logger.info(f"Computed depth map for image {img_id}")
        
        logger.info(f"Computed {len(self.depth_maps)} depth maps")
    
    def _find_best_neighbors(self, ref_idx: int, n_neighbors: int = 2) -> list:
        """
        Find best neighboring views for depth map computation
        
        Args:
            ref_idx: Index of reference image
            n_neighbors: Number of neighbors to find
            
        Returns:
            List of neighbor indices
        """
        ref_camera = self.reconstruction['cameras'][ref_idx]
        ref_center = -ref_camera[:3, :3].T @ ref_camera[:3, 3]
        
        distances = []
        for i, camera in enumerate(self.reconstruction['cameras']):
            if i == ref_idx:
                continue
                
            center = -camera[:3, :3].T @ camera[:3, 3]
            dist = np.linalg.norm(center - ref_center)
            
            # Compute viewing angle
            R_rel = camera[:3, :3] @ ref_camera[:3, :3].T
            angle = np.arccos((np.trace(R_rel) - 1) / 2)
            
            # Score based on distance and angle
            score = dist * (1 + angle)
            distances.append((i, score))
        
        # Sort by score and return indices
        neighbors = sorted(distances, key=lambda x: x[1])[:n_neighbors]
        return [idx for idx, _ in neighbors]
    
    def _compute_relative_pose(self, idx1: int, idx2: int) -> tuple:
        """
        Compute relative pose between two cameras
        
        Args:
            idx1: Index of first camera
            idx2: Index of second camera
            
        Returns:
            Tuple of (R_rel, t_rel)
        """
        camera1 = self.reconstruction['cameras'][idx1]
        camera2 = self.reconstruction['cameras'][idx2]
        
        R1 = camera1[:3, :3]
        t1 = camera1[:3, 3]
        R2 = camera2[:3, :3]
        t2 = camera2[:3, 3]
        
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        return R_rel, t_rel.reshape(3, 1)
    
    def _fuse_depth_maps(self, depth_maps: list) -> np.ndarray:
        """
        Fuse multiple depth maps using median filtering
        
        Args:
            depth_maps: List of depth maps
            
        Returns:
            Fused depth map
        """
        depth_maps = np.stack(depth_maps)
        
        # Remove invalid depths
        depth_maps[depth_maps < self.min_depth] = np.nan
        depth_maps[depth_maps > self.max_depth] = np.nan
        
        # Compute median ignoring nans
        return np.nanmedian(depth_maps, axis=0)
    
    def create_dense_point_cloud(self):
        """Create dense point cloud from depth maps"""
        logger.info("Creating dense point cloud...")
        points = []
        colors = []

        # First compute depth maps if not already done
        if not self.depth_maps:
            logger.info("Computing depth maps first...")
            self.compute_depth_maps()

        for i, img_id in enumerate(self.reconstruction['images']):
            if i not in self.depth_maps:
                logger.warning(f"No depth map for image {img_id}")
                continue
                
            depth_map = self.depth_maps[i]
            img_path = self.image_dir / 'images' / f"{img_id:06d}.jpg"  # Updated path
            
            logger.info(f"Processing image {img_id}")
            logger.info(f"Depth map shape: {depth_map.shape}")
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Read image for color
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get camera parameters
            camera = self.reconstruction['cameras'][i]
            R = camera[:3, :3]
            t = camera[:3, 3]
            K = self.reconstruction['camera_matrix']
            
            # Create points from depth map
            rows, cols = depth_map.shape[:2]
            step = 2  # Skip some pixels for efficiency
            points_count = 0
            
            for y in range(0, rows, step):
                for x in range(0, cols, step):
                    depth = depth_map[y, x]
                    if self.min_depth < depth < self.max_depth:
                        # Back-project to 3D using camera matrix
                        point_img = np.array([x, y, 1]) * depth
                        point_cam = np.linalg.inv(K) @ point_img
                        point_world = R.T @ (point_cam - t)
                        
                        # Check if point is within reasonable bounds
                        if np.all(np.abs(point_world) < 100):  # Adjust threshold as needed
                            points.append(point_world)
                            colors.append(img[y, x] / 255.0)
                            points_count += 1
            
            logger.info(f"Generated {points_count} points from image {img_id}")
        
        total_points = len(points)
        logger.info(f"Total points generated: {total_points}")
        
        if total_points < self.min_points_for_reconstruction:
            logger.warning("Low point count, reducing minimum requirement")
            self.min_points_for_reconstruction = min(100, total_points)  # Adjust threshold
        
        if total_points == 0:
            raise ValueError("No valid points generated in dense reconstruction")
        
        points = np.array(points)
        colors = np.array(colors)
        
        # Filter outliers
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        inliers = distances < np.percentile(distances, 95)  # Keep 95% closest points
        
        return points[inliers], colors[inliers]
    
    def estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """
        Estimate normals for point cloud
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Nx3 array of normal vectors
        """
        if len(points) == 0:
            raise ValueError("Empty point cloud")
            
        logger.info("Estimating normals...")
        
        # Convert points to float64 for better precision
        points = np.asarray(points, dtype=np.float64)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.5,  # Search radius
                max_nn=50    # Maximum nearest neighbors
            )
        )
        
        # Get camera center (average of all camera centers)
        centers = []
        for camera in self.reconstruction['cameras']:
            R = camera[:3, :3]
            t = camera[:3, 3]
            center = -R.T @ t
            centers.append(center)
        camera_center = np.mean(centers, axis=0)
        
        # Orient normals towards cameras
        pcd.orient_normals_towards_camera_location(camera_center)
        
        logger.info("Normal estimation completed")
        return np.asarray(pcd.normals)
    
    def create_mesh(self, points: np.ndarray, normals: np.ndarray):
        """
        Create mesh using Poisson reconstruction
        
        Args:
            points: Nx3 array of 3D points
            normals: Nx3 array of normal vectors
            
        Returns:
            Open3D mesh
        """
        if len(points) != len(normals):
            raise ValueError("Number of points and normals must match")
            
        logger.info("Creating mesh...")
        
        # Create point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=9,       # Reconstruction depth
            width=0,       # Default width
            scale=1.1,     # Scale factor
            linear_fit=False  # Use non-linear optimization
        )
        
        logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
        return mesh
        
    def save_results(self, output_dir: Path):
        """
        Save reconstruction results
        
        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save depth maps
        depth_dir = output_dir / 'depth_maps'
        depth_dir.mkdir(exist_ok=True)
        
        for i, depth_map in self.depth_maps.items():
            img_id = self.reconstruction['images'][i]
            np.save(depth_dir / f"depth_{img_id:06d}.npy", depth_map)
            
        logger.info(f"Results saved to {output_dir}")