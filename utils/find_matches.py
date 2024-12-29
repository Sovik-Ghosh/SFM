import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_two_images(img1, img2, plot=False):
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize FAST detector
    fast = cv2.FastFeatureDetector_create(
        threshold=20,           # Detect keypoints with higher contrast
        nonmaxSuppression=True  # Remove nearby points
    )
    
    # Detect keypoints
    kp1 = fast.detect(gray1)
    kp2 = fast.detect(gray2)
    
    # Compute descriptors
    orb = cv2.ORB_create(
        nfeatures=10000,        # Increase max features
        scaleFactor=1.2,        # Pyramid decimation ratio
        nlevels=8,              # Number of pyramid levels
        edgeThreshold=31,       # Size of border where features are not detected
        firstLevel=0,           # The level of pyramid to put source image to
        WTA_K=2,                # Number of points that produce each element of the descriptor
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31            # Size of the patch used by the oriented BRIEF descriptor
    )
    
    # Compute descriptors
    kp1, des1 = orb.compute(gray1, kp1)
    kp2, des2 = orb.compute(gray2, kp2)
    
    print(f"Features detected in image 1: {len(kp1)}")
    print(f"Features detected in image 2: {len(kp2)}")
    
    # Initialize FLANN matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using kNN
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Use RANSAC to find inliers
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    
    print(f"Number of inlier matches: {len(inlier_matches)}")
    
    # Draw matches
    if plot:
        match_img = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Inlier matches: {len(inlier_matches)}')
        plt.axis('off')
        plt.show()
    
    return kp1, kp2, inlier_matches

def visualize_standard_matches(img1, img2, kp1, kp2, matches):
    """Standard OpenCV match visualization"""
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Standard Match Visualization - {len(matches)} matches')
    plt.axis('off')
    plt.show()

def visualize_colored_matches(img1, img2, kp1, kp2, matches):
    """Detailed visualization with random colored lines"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create combined image
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # Draw matches with random colors
    np.random.seed(42)
    for match in matches:
        color = np.random.randint(0, 255, 3).tolist()
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])
        
        cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Colored Match Visualization - {len(matches)} matches')
    plt.axis('off')
    plt.show()

def visualize_flow(img1, img2, kp1, kp2, matches):
    """Flow-like visualization"""
    vis = img1.copy()
    
    for match in matches:
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        
        # Calculate flow direction and magnitude
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        magnitude = np.sqrt(dx*dx + dy*dy)
        
        # Color based on direction and magnitude
        angle = np.arctan2(dy, dx)
        color = (
            int(128 + 127*np.cos(angle)),
            int(128 + 127*np.sin(angle)),
            int(255 * (magnitude / max(img1.shape)))
        )
        
        cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Flow Visualization - {len(matches)} matches')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage
    img1_path = '/Users/sovikghosh/Desktop/doc/3D/SFM/data/images/DSC01161.JPG'
    img2_path = '/Users/sovikghosh/Desktop/doc/3D/SFM/data/images/DSC01162.JPG'
    

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Call visualization functions
    kp1, kp2, matches = match_two_images(img1, img2)
    visualize_standard_matches(img1, img2, kp1, kp2, matches)
    visualize_colored_matches(img1, img2, kp1, kp2, matches)
    visualize_flow(img1, img2, kp1, kp2, matches)