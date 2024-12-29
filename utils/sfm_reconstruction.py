import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from .bundle_adjustment import run_bundle_adjustment

class SfMReconstruction:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.reconstruction = {
            'cameras': [],
            'points_3d': [],
            'image_points': [],
            'images': [],
            'camera_matrix': np.array([
                                        [1100, 0, 512],
                                        [0, 1100, 384],
                                        [0, 0, 1]
                                    ], dtype=np.float32)

        }

    def load_pair_data(self, pair_name):
        """Load precomputed pair data from files"""
        # Load corresponding points (these are now only inliers)
        pts1 = np.load(self.data_dir / 'correspondences' / f'{pair_name}_pts1.npy')
        pts2 = np.load(self.data_dir / 'correspondences' / f'{pair_name}_pts2.npy')
        
        # Load fundamental matrix data
        F_data = np.load(self.data_dir / 'fundamental' / f'{pair_name}_F.npz')
        F = F_data['F']
        
        # Load matches data (includes inlier mask)
        matches_data = np.load(self.data_dir / 'matches' / f'{pair_name}_matches.npz')
        inlier_mask = matches_data['inlier_mask']
        
        return pts1, pts2, F, inlier_mask

    def reconstruct(self, matches_csv):
        """Main reconstruction function"""
        # Load and sort matches, now includes geometric verification metrics
        df = pd.read_csv(matches_csv)
        df_sorted = df.sort_values(['inlier_ratio', 'num_inliers'], 
                                 ascending=[False, False])
        
        # Filter pairs based on geometric verification
        df_filtered = df_sorted[
            (df_sorted['inlier_ratio'] > 0.3) & 
            (df_sorted['num_inliers'] > 50)  # Increased minimum inliers
        ]
        
        if len(df_filtered) == 0:
            raise ValueError("No image pairs pass geometric verification criteria")
        
        # Initialize from best pair
        first_pair = df_filtered.iloc[0]
        pair_name = first_pair['pair_name']
        img1_path = self.data_dir / 'images' / first_pair['img1']
        img2_path = self.data_dir / 'images' / first_pair['img2']
        
        # Load initial pair data
        pts1, pts2, F, mask = self.load_pair_data(pair_name)
        
        # Ensure points are in correct format
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        

        # Check if we have enough points
        if len(pts1) < 8:  # Need at least 8 points for fundamental matrix
            raise ValueError(f"Not enough points in initial pair: {len(pts1)}")
            
        # Initialize reconstruction from first pair
        E = self.reconstruction['camera_matrix'].T @ F @ self.reconstruction['camera_matrix']
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.reconstruction['camera_matrix'])

        
        # Set initial cameras and points
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera matrix
        P2 = np.hstack((R, t))  # Second camera matrix
        
        # Triangulate initial points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        # Update reconstruction
        self.reconstruction['cameras'] = [P1, P2]
        self.reconstruction['points_3d'] = points_3d.T
        self.reconstruction['image_points'] = [pts1, pts2]
        self.reconstruction['images'] = [img1_path, img2_path]
        
        # Process remaining images
        processed_images = set([img1_path.name, img2_path.name])
        
        # Iterate through remaining pairs
        for _, row in df_filtered.iterrows():
            if row['pair_name'] == pair_name:
                continue
                
            img1, img2 = row['img1'], row['img2']
            
            # Skip if both images already processed
            if img1 in processed_images and img2 in processed_images:
                continue
                
            # Process new image
            new_img = img1 if img1 not in processed_images else img2
            if new_img in processed_images:
                continue
                
            # Load pair data
            pts1, pts2, _, _ = self.load_pair_data(row['pair_name'])
            
            # Ensure points are in correct format
            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)
            
            # Skip if not enough points
            if len(pts1) < 10:  # Need enough points for PnP
                continue
                
            # Swap points if needed
            if row['img2'] == new_img:
                pts1, pts2 = pts2, pts1
            
            try:
                # Convert 3D-2D correspondences to correct format
                obj_points = self.reconstruction['points_3d'][:len(pts1)]
                img_points = pts1
                
                if len(obj_points) < 4:
                    continue
                    
                # Ensure points are in correct format for solvePnP
                obj_points = np.float32(obj_points)
                img_points = np.float32(img_points)

                #print("3D object points shape:", obj_points.shape)
                #print("2D image points shape:", img_points.shape)
                #print("Number of 3D points:", len(obj_points))
                #print("Number of 2D points:", len(img_points))
                
                # PnP to get new camera pose
                try:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                                                        objectPoints=obj_points,
                                                        imagePoints=img_points,
                                                        cameraMatrix=self.reconstruction['camera_matrix'],
                                                        distCoeffs=None,
                                                        iterationsCount=200,
                                                        reprojectionError=8.0,  # Added typical default reprojection error
                                                        confidence=0.99,
                                                        flags=cv2.SOLVEPNP_ITERATIVE
                                                    )
                    if not success:
                        print("PnP RANSAC failed to find a solution")
                except cv2.error as e:
                    print(f"OpenCV error in solvePnPRansac: {e}")
                
                if not success or len(inliers) < 10:
                    continue
                    
                # Convert rotation vector to matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create new camera matrix
                P_new = np.hstack((R, tvec))
                
                # Triangulate new points
                points_4d = cv2.triangulatePoints(
                    self.reconstruction['cameras'][-1],
                    P_new,
                    pts2.T,
                    pts1.T
                )
                new_points_3d = points_4d[:3] / points_4d[3]
                
                # Update reconstruction
                self.reconstruction['cameras'].append(P_new)
                self.reconstruction['points_3d'] = np.vstack([
                    self.reconstruction['points_3d'],
                    new_points_3d.T
                ])
                self.reconstruction['image_points'].append(pts1)
                self.reconstruction['images'].append(self.data_dir / 'images' / new_img)
                processed_images.add(new_img)
                
                print(f"Added image {new_img} to reconstruction")
                
            except Exception as e:
                print(f"Error processing image {new_img}: {str(e)}")
                continue
        
        return self.reconstruction