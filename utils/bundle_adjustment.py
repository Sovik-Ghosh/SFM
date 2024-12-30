import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import logging

logger = logging.getLogger(__name__)

class BundleAdjustment:
    def __init__(self, camera_params, points_3d, point_2d, camera_indices, point_indices, camera_matrix):
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.point_2d = point_2d
        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.camera_matrix = camera_matrix
        
        # Added validation
        if len(self.camera_params) == 0 or len(self.points_3d) == 0:
            raise ValueError("Empty camera parameters or 3D points")
        
        # Store dimensions for later use
        self.n_cameras = len(self.camera_params)
        self.n_points = len(self.points_3d)
        self.n_obs = len(self.point_2d)
        
    def project(self, points, camera_params):
        """Convert 3D points to 2D by projecting them using camera parameters"""
        points_proj = self.rotate_points(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        
        # Handle points at infinity
        valid_points = points_proj[:, 2] > 1e-8
        points_proj[valid_points] = points_proj[valid_points, :2] / points_proj[valid_points, 2, np.newaxis]
        
        # Apply camera intrinsics
        f = self.camera_matrix[0, 0]  # Assuming fx = fy
        c = self.camera_matrix[:2, 2]
        points_proj[valid_points] *= f
        points_proj[valid_points] += c
        
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
        camera_params = params[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        points_3d = params[self.n_cameras * 6:].reshape((self.n_points, 3))
        
        points_proj = self.project(points_3d[self.point_indices],
                                 camera_params[self.camera_indices])
        
        return (points_proj - self.point_2d).ravel()
    
    def optimize(self, verbose=True, max_nfev=100):
        """Run bundle adjustment optimization"""
        x0 = np.hstack([self.camera_params.ravel(), self.points_3d.ravel()])
        
        # Create sparse matrix pattern
        A = self.create_sparse_matrix(self.n_cameras, self.n_points, self.n_obs)
        
        logger.info(f"Starting optimization with {self.n_cameras} cameras, "
                   f"{self.n_points} points, and {self.n_obs} observations")
        
        try:
            res = least_squares(
                self.compute_residuals, 
                x0, 
                jac_sparsity=A, 
                verbose=2 if verbose else 0,
                method='trf',
                loss='cauchy',
                tr_solver='lsmr',
                max_nfev=max_nfev,
                ftol=1e-4,
                xtol=1e-4
            )
            
            if res.success:
                final_cost = np.mean(res.fun**2)
                logger.info(f"Bundle adjustment converged with final cost: {final_cost:.6f}")
                
                # Extract optimized parameters
                camera_params = res.x[:self.n_cameras * 6].reshape((self.n_cameras, 6))
                points_3d = res.x[self.n_cameras * 6:].reshape((self.n_points, 3))
                return camera_params, points_3d
            else:
                raise RuntimeError(f"Bundle adjustment failed to converge: {res.message}")
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise
    
    def create_sparse_matrix(self, n_cameras, n_points, n_obs):
        """Create the sparse matrix pattern for bundle adjustment"""
        m = n_obs * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)
        
        i = np.arange(n_obs)
        for s in range(6):
            A[2 * i, self.camera_indices * 6 + s] = 1
            A[2 * i + 1, self.camera_indices * 6 + s] = 1
            
        for s in range(3):
            A[2 * i, n_cameras * 6 + self.point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + self.point_indices * 3 + s] = 1
            
        return A

def run_bundle_adjustment(reconstruction):
    """Run bundle adjustment on reconstruction data"""
    logger.info("Preparing bundle adjustment...")
    
    try:
        # Convert camera matrices to Rodrigues rotation and translation
        camera_params = []
        for camera in reconstruction['cameras']:
            R = camera[:3, :3]
            t = camera[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            camera_params.append(np.concatenate([rvec.ravel(), t]))
        camera_params = np.array(camera_params)
        
        # Prepare point indices and camera indices
        camera_indices = []
        point_indices = []
        points_2d = []
        
        for i, pts in enumerate(reconstruction['image_points']):
            for j, pt in enumerate(pts):
                if j < len(reconstruction['points_3d']):  # Only use points that have 3D coordinates
                    camera_indices.append(i)
                    point_indices.append(j)
                    points_2d.append(pt[:2])  # Ensure using only 2D coordinates
        
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_2d = np.array(points_2d, dtype=np.float64)
        
        points_3d = reconstruction['points_3d'].astype(np.float64)
        
        logger.info(f"Starting optimization with {len(reconstruction['cameras'])} cameras, "
                   f"{len(points_3d)} points, and {len(points_2d)} observations")
        
        # Create BA object
        ba = BundleAdjustment(
            camera_params=camera_params,
            points_3d=points_3d,
            point_2d=points_2d,
            camera_indices=camera_indices,
            point_indices=point_indices,
            camera_matrix=reconstruction['camera_matrix']
        )
        
        # Run optimization
        optimized_cameras, optimized_points = ba.optimize()
        
        # Convert back to camera matrices
        new_cameras = []
        for params in optimized_cameras:
            rvec = params[:3]
            t = params[3:6]
            R, _ = cv2.Rodrigues(rvec)
            new_cameras.append(np.hstack([R, t.reshape(3, 1)]))
        
        # Update reconstruction
        reconstruction['cameras'] = new_cameras
        reconstruction['points_3d'] = optimized_points
        
        return reconstruction
        
    except Exception as e:
        logger.error(f"Bundle adjustment failed: {str(e)}")
        return reconstruction  # Return original if optimization fails