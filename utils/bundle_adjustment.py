import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

class BundleAdjustment:
    def __init__(self, camera_params, points_3d, point_2d, camera_indices, point_indices, camera_matrix):
        self.camera_params = camera_params  # Rotation vectors and translations (6 params per camera)
        self.points_3d = points_3d  # 3D points
        self.point_2d = point_2d    # 2D observations
        self.camera_indices = camera_indices  # Camera index for each observation
        self.point_indices = point_indices    # Point index for each observation
        self.camera_matrix = camera_matrix    # Intrinsic camera matrix
        
    def project(self, points, camera_params):
        """Convert 3D points to 2D by projecting them using camera parameters"""
        points_proj = self.rotate_points(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        
        # Perspective division
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        
        # Apply camera intrinsics
        f = self.camera_matrix[0, 0]  # Assuming fx = fy
        c = self.camera_matrix[:2, 2]
        points_proj *= f
        points_proj += c
        
        return points_proj
    
    def rotate_points(self, points, rot_vecs):
        """Convert rotation vectors to matrices and multiply"""
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        return cos_theta * points + sin_theta * np.cross(v, points) + \
               dot * (1 - cos_theta) * v
    
    def compute_residuals(self, params):
        """Compute residuals for bundle adjustment"""
        n_cameras = int(len(self.camera_params))
        n_points = int(len(self.points_3d))
        
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        points_proj = self.project(points_3d[self.point_indices],
                                 camera_params[self.camera_indices])
        
        return (points_proj - self.point_2d).ravel()
    
    def optimize(self, verbose=True):
        """Run bundle adjustment optimization"""
        x0 = np.hstack([self.camera_params.ravel(), self.points_3d.ravel()])
        n_cameras = len(self.camera_params)
        n_points = len(self.points_3d)
        n_obs = len(self.point_2d)
        
        # Create sparse matrix pattern
        A = self.create_sparse_matrix(n_cameras, n_points, n_obs)
        
        # Optimize
        res = least_squares(
            self.compute_residuals, 
            x0, 
            jac_sparsity=A, 
            verbose=2 if verbose else 0,
            method='trf',
            loss='cauchy',
            tr_solver='lsmr',
            max_nfev=100
        )
        
        if res.success:
            # Extract optimized parameters
            camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
            points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
            return camera_params, points_3d
        else:
            raise RuntimeError("Bundle adjustment failed to converge")
    
    def create_sparse_matrix(self, n_cameras, n_points, n_obs):
        """Create the sparse matrix pattern for bundle adjustment"""
        m = n_obs * 2  # Each observation has x and y coordinates
        n = n_cameras * 6 + n_points * 3  # Camera params and 3D points
        A = lil_matrix((m, n), dtype=int)
        
        i = np.arange(n_obs)
        for s in range(6):  # Camera parameters
            A[2 * i, self.camera_indices * 6 + s] = 1
            A[2 * i + 1, self.camera_indices * 6 + s] = 1
            
        for s in range(3):  # 3D points
            A[2 * i, n_cameras * 6 + self.point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + self.point_indices * 3 + s] = 1
            
        return A

def run_bundle_adjustment(reconstruction_data):
    """
    Run bundle adjustment on reconstruction data
    
    Args:
        reconstruction_data (dict): Dictionary containing:
            - cameras: List of camera matrices
            - points_3d: Array of 3D points
            - image_points: List of 2D points for each image
            - camera_matrix: Intrinsic camera matrix
    
    Returns:
        dict: Optimized reconstruction data
    """
    # Convert camera matrices to Rodrigues rotation and translation
    camera_params = []
    for camera in reconstruction_data['cameras']:
        R = camera[:, :3]
        t = camera[:, 3]
        rvec, _ = cv2.Rodrigues(R)
        camera_params.append(np.concatenate([rvec.ravel(), t]))
    camera_params = np.array(camera_params)
    
    # Prepare point indices and camera indices
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for i, pts_2d in enumerate(reconstruction_data['image_points']):
        for j, pt in enumerate(pts_2d):
            camera_indices.append(i)
            point_indices.append(j)
            points_2d.append(pt)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    # Create bundle adjustment object
    ba = BundleAdjustment(
        camera_params=camera_params,
        points_3d=reconstruction_data['points_3d'],
        point_2d=points_2d,
        camera_indices=camera_indices,
        point_indices=point_indices,
        camera_matrix=reconstruction_data['camera_matrix']
    )
    
    # Run optimization
    try:
        print("Running bundle adjustment...")
        optimized_cameras, optimized_points = ba.optimize()
        
        # Convert back to camera matrices
        optimized_camera_matrices = []
        for params in optimized_cameras:
            rvec = params[:3]
            t = params[3:6]
            R, _ = cv2.Rodrigues(rvec)
            optimized_camera_matrices.append(np.hstack([R, t.reshape(3, 1)]))
        
        # Update reconstruction data
        reconstruction_data['cameras'] = optimized_camera_matrices
        reconstruction_data['points_3d'] = optimized_points
        print("Bundle adjustment completed successfully")
        
    except Exception as e:
        print(f"Bundle adjustment failed: {str(e)}")
    
    return reconstruction_data