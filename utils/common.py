import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def detect_features(img):
    """
    Detect FAST features and compute ORB descriptors
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        keypoints: List of keypoints
        descriptors: Array of descriptors
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Initialize FAST detector
    fast = cv2.FastFeatureDetector_create(
        threshold=20,           # Intensity threshold
        nonmaxSuppression=True  # Apply non-maximum suppression
    )
    
    # Detect keypoints
    keypoints = fast.detect(gray)
    
    # Initialize ORB descriptor
    orb = cv2.ORB_create(
        nfeatures=10000,        # Maximum number of features
        scaleFactor=1.2,        # Pyramid scale factor
        nlevels=8,              # Number of pyramid levels
        edgeThreshold=31,       # Border size where features are not detected
        firstLevel=0,           # Level of pyramid to put source image
        WTA_K=2,                # Number of random pixels to produce descriptor element
        scoreType=cv2.ORB_HARRIS_SCORE,  # Use Harris score
        patchSize=31,           # Size of patch used for orientation
        fastThreshold=20        # FAST threshold
    )
    
    # Compute descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)
    
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match features using FLANN matcher with ratio test
    
    Args:
        desc1, desc2: Feature descriptors to match
        
    Returns:
        good_matches: List of good matches that pass ratio test
    """
    # Initialize FLANN matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,         # Number of hash tables
        key_size=12,            # Length of the key in the hash tables
        multi_probe_level=1     # Number of neighboring buckets to search
    )
    search_params = dict(
        checks=50               # Number of times to check each bucket
    )
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for group in matches:
        if len(group) >= 2:
            best, second = group
            # If best match is significantly better than second best
            if best.distance < 0.7 * second.distance:
                good_matches.append(best)
    
    return good_matches

def geometric_verification(pts1, pts2, F, threshold=3.0):
    """
    Perform geometric verification of matches using epipolar geometry
    
    Args:
        pts1, pts2: Corresponding points in two images
        F: Fundamental matrix
        threshold: Distance threshold for epipolar lines
        
    Returns:
        dict: Verification results and metrics
    """
    # Compute epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # Compute symmetric epipolar distances
    errors1 = np.abs(np.sum(np.multiply(pts1, lines1[:, :2]), axis=1) + lines1[:, 2]) / \
             np.sqrt(np.sum(np.square(lines1[:, :2]), axis=1))
    errors2 = np.abs(np.sum(np.multiply(pts2, lines2[:, :2]), axis=1) + lines2[:, 2]) / \
             np.sqrt(np.sum(np.square(lines2[:, :2]), axis=1))
    
    symmetric_errors = (errors1 + errors2) / 2
    
    # Identify inliers
    inlier_mask = symmetric_errors < threshold
    
    # Calculate reprojection error
    reproj_error = np.mean(symmetric_errors[inlier_mask]) if np.any(inlier_mask) else float('inf')
    
    # Check point distribution
    if np.any(inlier_mask):
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]
        
        # Compute point spread
        pts1_std = np.std(pts1_inliers, axis=0)
        pts2_std = np.std(pts2_inliers, axis=0)
        min_spread = 20  # minimum required spread in pixels
        well_distributed = np.all(pts1_std > min_spread) and np.all(pts2_std > min_spread)
    else:
        well_distributed = False
    
    return {
        'metrics': {
            'total_matches': len(pts1),
            'inliers': np.sum(inlier_mask),
            'inlier_ratio': np.mean(inlier_mask),
            'reprojection_error': reproj_error,
            'symmetric_error': np.mean(symmetric_errors),
            'well_distributed': well_distributed
        },
        'inlier_mask': inlier_mask,
        'symmetric_errors': symmetric_errors
    }

def verify_match_quality(geometric_results, min_inliers=15, min_ratio=0.3, max_error=2.0):
    """
    Verify if matches pass quality thresholds
    
    Args:
        geometric_results: Results from geometric_verification
        min_inliers: Minimum number of inliers required
        min_ratio: Minimum inlier ratio required
        max_error: Maximum allowed reprojection error
        
    Returns:
        bool: Whether matches pass quality checks
    """
    metrics = geometric_results['metrics']
    
    conditions = [
        metrics['inliers'] >= min_inliers,
        metrics['inlier_ratio'] >= min_ratio,
        metrics['reprojection_error'] <= max_error,
        metrics['well_distributed']
    ]
    
    return all(conditions)

def visualize_matches(img1, img2, kp1, kp2, matches, title="Matches"):
    """
    Visualize matches between two images
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints in images
        matches: List of matches
        title: Plot title
    """
    # Draw matches
    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Display
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} - {len(matches)} matches')
    plt.axis('off')
    plt.show()

def visualize_geometric_verification(img1, img2, pts1, pts2, geometric_results, save_path=None):
    """
    Visualize geometric verification results
    
    Args:
        img1, img2: Input images
        pts1, pts2: Corresponding points
        geometric_results: Results from geometric verification
        save_path: Optional path to save visualization
    """
    inlier_mask = geometric_results['inlier_mask']
    metrics = geometric_results['metrics']
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot first image
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.scatter(pts1[inlier_mask, 0], pts1[inlier_mask, 1], 
               c='g', s=20, label='Inliers')
    plt.scatter(pts1[~inlier_mask, 0], pts1[~inlier_mask, 1], 
               c='r', s=20, label='Outliers')
    plt.legend()
    plt.title(f'Image 1 - {metrics["inliers"]} inliers')
    
    # Plot second image
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2[inlier_mask, 0], pts2[inlier_mask, 1], 
               c='g', s=20, label='Inliers')
    plt.scatter(pts2[~inlier_mask, 0], pts2[~inlier_mask, 1], 
               c='r', s=20, label='Outliers')
    plt.legend()
    plt.title(f'Image 2 - {metrics["inlier_ratio"]:.2%} inlier ratio')
    
    # Add metrics text
    plt.figtext(0.02, 0.02, 
                f'Reprojection Error: {metrics["reprojection_error"]:.2f}\n' +
                f'Well Distributed: {metrics["well_distributed"]}',
                fontsize=10)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def load_image_pair(img1_path, img2_path):
    """
    Load and check image pair
    
    Args:
        img1_path, img2_path: Paths to images
        
    Returns:
        tuple: (img1, img2) if successful, (None, None) if failed
    """
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        print(f"Error loading images: {img1_path}, {img2_path}")
        return None, None
        
    return img1, img2