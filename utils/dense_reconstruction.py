import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

class DenseReconstruction:
    def __init__(self, reconstruction, image_dir):
        self.reconstruction = reconstruction
        self.image_dir = Path(image_dir)
        self.depth_maps = {}
        
    def compute_depth_maps(self):
        """
        Compute depth maps for all images using Semi-Global Block Matching
        """
        for i, image_path in enumerate(self.reconstruction['images']):
            # Read reference image
            ref_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            # Find best neighbor images
            neighbors = self._find_best_neighbors(i, n_neighbors=2)
            
            depth_maps = []
            for neighbor_idx in neighbors:
                # Read neighbor image
                neighbor_path = self.reconstruction['images'][neighbor_idx]
                neighbor_img = cv2.imread(str(neighbor_path), cv2.IMREAD_GRAYSCALE)
                
                # Compute relative pose
                R_rel, t_rel = self._compute_relative_pose(i, neighbor_idx)
                
                # Rectify images
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
                stereo = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=128,
                    blockSize=5,
                    P1=8 * 3 * 5**2,
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )
                
                # Compute disparity
                disparity = stereo.compute(ref_img, neighbor_img)
                
                # Convert disparity to depth
                depth = cv2.reprojectImageTo3D(disparity, Q)
                depth_maps.append(depth)
            
            # Fuse depth maps
            self.depth_maps[i] = self._fuse_depth_maps(depth_maps)
    
    def _find_best_neighbors(self, ref_idx, n_neighbors=2):
        """Find best neighboring views for depth map computation"""
        ref_camera = self.reconstruction['cameras'][ref_idx]
        ref_center = -ref_camera[:3, :3].T @ ref_camera[:3, 3]
        
        distances = []
        for i, camera in enumerate(self.reconstruction['cameras']):
            if i == ref_idx:
                continue
            
            center = -camera[:3, :3].T @ camera[:3, 3]
            dist = np.linalg.norm(center - ref_center)
            distances.append((i, dist))
        
        # Sort by distance and return indices
        neighbors = sorted(distances, key=lambda x: x[1])[:n_neighbors]
        return [idx for idx, _ in neighbors]
    
    def _compute_relative_pose(self, idx1, idx2):
        """Compute relative pose between two cameras"""
        camera1 = self.reconstruction['cameras'][idx1]
        camera2 = self.reconstruction['cameras'][idx2]
        
        R1 = camera1[:3, :3]
        t1 = camera1[:3, 3]
        R2 = camera2[:3, :3]
        t2 = camera2[:3, 3]
        
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        return R_rel, t_rel
    
    def _fuse_depth_maps(self, depth_maps):
        """Fuse multiple depth maps using median filtering"""
        depth_maps = np.stack(depth_maps)
        return np.median(depth_maps, axis=0)
    
    def estimate_normals(self, points):
        """
        Estimate normals for point cloud
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            normals: Nx3 array of normal vectors
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,  # Search radius
                max_nn=30    # Maximum nearest neighbors
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
        
        return np.asarray(pcd.normals)
    
    def create_dense_point_cloud(self):
        """Create dense point cloud from depth maps"""
        points = []
        colors = []
        
        for i, (image_path, depth_map) in enumerate(zip(self.reconstruction['images'], 
                                                       self.depth_maps.values())):
            # Read image for color
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get camera parameters
            camera = self.reconstruction['cameras'][i]
            R = camera[:3, :3]
            t = camera[:3, 3]
            
            # Create points from depth map
            rows, cols = depth_map.shape[:2]
            for y in range(rows):
                for x in range(cols):
                    depth = depth_map[y, x]
                    if (depth > 0).any():  # Valid depth
                        # Back-project to 3D
                        point_cam = np.array([x, y, 1]) * depth
                        point_world = R.T @ (point_cam - t)
                        
                        points.append(point_world)
                        colors.append(img[y, x] / 255.0)
        
        return np.array(points), np.array(colors)
    
    def create_mesh(self, points, normals):
        """Create mesh using Poisson reconstruction"""
        # Create point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=9,  # Reconstruction depth
            width=0,  # Default width
            scale=1.1,  # Scale factor
            linear_fit=False  # Use non-linear optimization
        )
        
        return mesh