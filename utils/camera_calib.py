import numpy as np
import cv2
import glob
import os
from typing import Tuple, List

class CameraCalibrator:
    def __init__(self, chessboard_size: Tuple[int, int], square_size: float = 1.0):
        """
        Initialize the camera calibrator.
        
        Args:
            chessboard_size: Tuple of (rows, cols) of internal corners in the chessboard
            square_size: Size of each square in the chessboard (in your preferred unit)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.mean_error = None
        
    def find_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Find chessboard corners in an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (success, corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
        return ret, corners
    
    def add_image(self, image: np.ndarray, visualize: bool = False) -> bool:
        """
        Process a calibration image and add its points if valid.
        
        Args:
            image: Input calibration image
            visualize: If True, shows the detected corners
            
        Returns:
            bool indicating if the image was successfully processed
        """
        ret, corners = self.find_corners(image)
        
        if ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            
            if visualize:
                # Draw and display the corners
                drawn_img = cv2.drawChessboardCorners(image.copy(), 
                                                    self.chessboard_size, 
                                                    corners, 
                                                    ret)
                cv2.imshow('Detected Corners', drawn_img)
                cv2.waitKey(500)
                
        return ret
    
    def calibrate(self, image_shape: Tuple[int, int]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform camera calibration.
        
        Args:
            image_shape: Tuple of (height, width) of the input images
            
        Returns:
            Tuple of (RMS error, camera matrix, distortion coefficients)
        """
        if not self.objpoints:
            raise ValueError("No calibration images have been processed")
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_shape[::-1], None, None)
        
        # Store the calibration results
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Calculate re-projection error
        self.mean_error = self.calculate_reprojection_error()
        
        return ret, mtx, dist
    
    def calculate_reprojection_error(self) -> float:
        """
        Calculate the re-projection error after calibration.
        
        Returns:
            Mean re-projection error
        """
        total_error = 0
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs)
            
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
            
        return total_error/len(self.objpoints)
    
    def save_calibration(self, filename: str):
        """
        Save calibration results to a file.
        
        Args:
            filename: Output filename
        """
        if self.camera_matrix is None:
            raise ValueError("Calibration hasn't been performed yet")
            
        np.savez(filename,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                rvecs=self.rvecs,
                tvecs=self.tvecs,
                mean_error=self.mean_error)

def main():
    # Example usage
    calibrator = CameraCalibrator(chessboard_size=(6,9), square_size=0.025)  # 25mm squares
    
    # Process all images in a directory
    image_files = glob.glob('calibration_images/*.jpg')
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image: {image_file}")
            continue
            
        success = calibrator.add_image(img, visualize=True)
        print(f"Processed {image_file}: {'Success' if success else 'Failed'}")
    
    # Perform calibration
    if calibrator.objpoints:
        image_shape = img.shape[:2]  # Height, Width
        rms, mtx, dist = calibrator.calibrate(image_shape)
        
        print("\nCalibration Results:")
        print(f"RMS: {rms}")
        print("\nCamera Matrix:")
        print(mtx)
        print("\nDistortion Coefficients:")
        print(dist)
        print(f"\nMean Re-projection Error: {calibrator.mean_error}")
        
        # Save the calibration results
        calibrator.save_calibration('camera_calibration.npz')
    else:
        print("No valid calibration images were processed")

if __name__ == "__main__":
    main()