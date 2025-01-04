import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging



class ImageMatcher:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.silhouette_dir = self.data_dir / 'silhouettes'
        self.matches_dir = self.data_dir / 'matches'
        self.fund_dir = self.data_dir / 'fundamental'
        self.corr_dir = self.data_dir / 'correspondences'
        self.viz_dir = self.data_dir / 'visualizations'
        
        # Create output directories
        for dir_path in [self.matches_dir, self.fund_dir, self.corr_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def visualize_features_with_mask(self, img, mask, keypoints, filename):
        """Debug visualization of features and mask"""
        debug_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw mask boundary in green
        if mask is not None:
            # Print unique values in mask
            print(f"Unique values in mask: {np.unique(mask)}")
            # Draw mask overlay
            mask_overlay = np.zeros_like(debug_img)
            mask_overlay[mask > 0] = [0, 255, 0]  # Green for mask area
            debug_img = cv2.addWeighted(debug_img, 1.0, mask_overlay, 0.3, 0)
        
        # Draw keypoints in red
        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)
            # Print mask value at keypoint
            if mask is not None:
                print(f"Mask value at keypoint ({x}, {y}): {mask[y, x]}")
        
        cv2.imwrite(filename, debug_img)

    def load_mask(self, idx):
        """Load PGM silhouette mask"""
        mask_path = self.silhouette_dir / f"{idx:05d}.pgm"
        
        if not mask_path.exists():
            logging.warning(f"Mask not found: {mask_path}")
            return None
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to load mask: {mask_path}")
            return None
        
        # Ensure binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # IMPORTANT: Invert the mask - foreground should be white (255)
        mask = cv2.bitwise_not(mask)
        
        # Optional: Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def detect_features(self, img, mask=None):
        """
        Detect FAST features and compute ORB descriptors with strict mask enforcement.
        FAST provides rapid keypoint detection while ORB adds rotation invariance
        and distinctive descriptors.
        """
        # Convert to grayscale if needed - both FAST and ORB require grayscale input
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Handle masking if provided
        if mask is not None:
            # Ensure mask is binary (0 or 255)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Apply mask to create our working image
            masked_img = cv2.bitwise_and(gray, gray, mask=mask)
        else:
            masked_img = gray
        
        # Initialize FAST detector
        # threshold: Intensity difference threshold (higher = fewer but stronger features)
        # nonmaxSuppression: True to ensure better feature distribution
        fast = cv2.FastFeatureDetector_create(
            threshold=20,
            nonmaxSuppression=True
        )
        
        # Detect FAST keypoints
        keypoints = fast.detect(masked_img, mask)
        
        # Initialize ORB for descriptor computation
        # We create ORB separately because we only want its descriptor computation
        orb = cv2.ORB_create(
            nfeatures=10000,  # Maximum number of features to retain
            scaleFactor=1.2,  # Pyramid decimation ratio
            nlevels=8,        # Number of pyramid levels
            edgeThreshold=31  # Size of border where features are not detected
        )
        
        # Compute ORB descriptors for the FAST keypoints
        # Note: ORB will only compute descriptors for the keypoints we pass it
        keypoints, descriptors = orb.compute(masked_img, keypoints)
        
        # Additional mask verification - ensures strict adherence to mask
        if mask is not None and keypoints:
            valid_keypoints = []
            valid_descriptors = []
            
            # Check each keypoint against the mask
            for idx, kp in enumerate(keypoints):
                x, y = map(int, kp.pt)
                # Verify point is within image bounds and in masked region
                if (0 <= y < mask.shape[0] and 
                    0 <= x < mask.shape[1] and 
                    mask[y, x] == 255):  # Check if point is in foreground
                    valid_keypoints.append(kp)
                    if descriptors is not None:
                        valid_descriptors.append(descriptors[idx])
            
            keypoints = valid_keypoints
            descriptors = np.array(valid_descriptors) if valid_descriptors else None
        
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match binary descriptors using Brute Force matcher"""
        # Create BFMatcher for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return good_matches

    def geometric_verification(self, pts1, pts2, F, threshold=3.0):
        """Perform geometric verification using epipolar geometry"""
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
        
        # Calculate metrics
        reproj_error = np.mean(symmetric_errors[inlier_mask]) if np.any(inlier_mask) else float('inf')
        
        # Check point distribution
        if np.any(inlier_mask):
            pts1_inliers = pts1[inlier_mask]
            pts2_inliers = pts2[inlier_mask]
            pts1_std = np.std(pts1_inliers, axis=0)
            pts2_std = np.std(pts2_inliers, axis=0)
            min_spread = 20
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

    def verify_match_quality(self, geometric_results, min_inliers=15, min_ratio=0.3, max_error=2.0):
        """Verify if matches pass quality thresholds"""
        metrics = geometric_results['metrics']
        
        conditions = [
            metrics['inliers'] >= min_inliers,
            metrics['inlier_ratio'] >= min_ratio,
            metrics['reprojection_error'] <= max_error,
            metrics['well_distributed']
        ]
        
        return all(conditions)

    def visualize_matches(self, img1, img2, kp1, kp2, matches, geometric_results, save_path):
        """Visualize matches with inlier/outlier classification"""
        inlier_mask = geometric_results['inlier_mask']
        metrics = geometric_results['metrics']
        
        # Create visualization image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        viz_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        viz_img[:h1, :w1] = img1
        viz_img[:h2, w1:w1+w2] = img2
        
        # Draw matches
        for idx, m in enumerate(matches):
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2 = tuple(map(int, (kp2[m.trainIdx].pt[0] + w1, kp2[m.trainIdx].pt[1])))
            
            color = (0, 255, 0) if inlier_mask[idx] else (0, 0, 255)
            cv2.line(viz_img, pt1, pt2, color, 1)
            cv2.circle(viz_img, pt1, 3, color, -1)
            cv2.circle(viz_img, pt2, 3, color, -1)
        
        # Add text with metrics
        text = f"Inliers: {metrics['inliers']}/{len(matches)} ({metrics['inlier_ratio']:.1%})"
        cv2.putText(viz_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(str(save_path), viz_img)

    def process_image_pair(self, img1_path, img2_path, pair_name, mask = False):
        """Process a single image pair with masks"""
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            raise ValueError(f"Could not read images: {img1_path}, {img2_path}")
        
        mask1 = mask2 = None
        
        # Load masks
        if mask:
            idx1 = int(img1_path.stem)
            idx2 = int(img2_path.stem)
            mask1 = self.load_mask(idx1)
            mask2 = self.load_mask(idx2)
        
        # Detect features with masks
        kp1, desc1 = self.detect_features(img1, mask1)
        kp2, desc2 = self.detect_features(img2, mask2)
        
        if len(kp1) < 1 or len(kp2) < 1:
            return None
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 5:
            return None
        
        # Get matching points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Compute fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        
        if F is None or F.shape != (3, 3):
            return None
        
        # Geometric verification
        geometric_results = self.geometric_verification(pts1, pts2, F)
        
        if not self.verify_match_quality(geometric_results):
            return None
        
        # Save visualizations
        viz_path = self.viz_dir / f'{pair_name}_matches.png'
        self.visualize_matches(img1, img2, kp1, kp2, matches, geometric_results, viz_path)
        
        # Save data
        inlier_mask = geometric_results['inlier_mask']
        self.save_pair_data(pair_name, pts1, pts2, F, inlier_mask, matches)
        
        return {
            'pair_name': pair_name,
            'img1': img1_path.name,
            'img2': img2_path.name,
            'num_matches': len(matches),
            'num_inliers': np.sum(inlier_mask),
            'inlier_ratio': geometric_results['metrics']['inlier_ratio'],
            'reprojection_error': geometric_results['metrics']['reprojection_error'],
            'well_distributed': geometric_results['metrics']['well_distributed']
        }

    def save_pair_data(self, pair_name, pts1, pts2, F, inlier_mask, matches):
        """Save matching data and results"""
        # Save corresponding points (only inliers)
        np.save(self.corr_dir / f'{pair_name}_pts1.npy', pts1[inlier_mask])
        np.save(self.corr_dir / f'{pair_name}_pts2.npy', pts2[inlier_mask])
        
        # Save fundamental matrix and mask
        np.savez(self.fund_dir / f'{pair_name}_F.npz',
                F=F, mask=inlier_mask, pts1=pts1, pts2=pts2)
        
        # Save matches
        np.savez(self.matches_dir / f'{pair_name}_matches.npz',
                queryIdx=np.array([m.queryIdx for m in matches]),
                trainIdx=np.array([m.trainIdx for m in matches]),
                distance=np.array([m.distance for m in matches]),
                inlier_mask=inlier_mask)

    def process_image_range(self, start_idx, end_idx, mask = False):
        """Process all image pairs in range"""
        pairs = []
        for i in range(start_idx, end_idx):
            for j in range(i+1, end_idx+1):
                img1_path = self.image_dir / f"{i:05d}.jpg"
                img2_path = self.image_dir / f"{j:05d}.jpg"
                
                if not (img1_path.exists() and img2_path.exists()):
                    continue
                    
                pair_name = f'pair_{i}_{j}'
                pairs.append((img1_path, img2_path, pair_name))
        
        for img1_path, img2_path, pair_name in tqdm(pairs, desc="Processing pairs"):
            try:
                result = self.process_image_pair(img1_path, img2_path, pair_name, mask)
                if result is not None:
                    self.results.append(result)
            except Exception as e:
                logging.error(f"Error processing {pair_name}: {str(e)}")
                continue

    def save_results(self, output_csv):
        """Save matching results summary"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        
        print("\nMatching Summary:")
        print(f"Total pairs processed: {len(self.results)}")
        print(f"Average matches per pair: {df['num_matches'].mean():.1f}")
        print(f"Average inliers per pair: {df['num_inliers'].mean():.1f}")
        print(f"Average inlier ratio: {df['inlier_ratio'].mean():.3f}")
        print(f"Average reprojection error: {df['reprojection_error'].mean():.3f}")
        print(f"Pairs with well-distributed points: {df['well_distributed'].sum()}")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize matcher
    matcher = ImageMatcher("/teamspace/studios/this_studio/SFM/data")
    
    # Process images
    matcher.process_image_range(1, 151, False)  # Process images 0-35
    
    # Save results
    matcher.save_results("/teamspace/studios/this_studio/SFM/data/matching_results.csv")

if __name__ == "__main__":
    main()