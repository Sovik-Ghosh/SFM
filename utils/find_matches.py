import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from .common import (detect_features, match_features, geometric_verification, 
                   verify_match_quality, visualize_geometric_verification)

class ImageMatcher:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.matches_dir = self.data_dir / 'matches'
        self.fund_dir = self.data_dir / 'fundamental'
        self.corr_dir = self.data_dir / 'correspondences'
        self.viz_dir = self.data_dir / 'visualizations'
        
        # Create directories
        for dir_path in [self.matches_dir, self.fund_dir, self.corr_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def process_image_pair(self, img1_path, img2_path, pair_name):
        """
        Process a single image pair with geometric verification
        """
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            raise ValueError(f"Could not read images: {img1_path}, {img2_path}")
        
        # Detect features
        kp1, desc1 = detect_features(img1)
        kp2, desc2 = detect_features(img2)
        
        if len(kp1) < 100 or len(kp2) < 100:  # Minimum features threshold
            return None
        
        # Match features
        matches = match_features(desc1, desc2)
        
        if len(matches) < 50:  # Minimum matches threshold
            return None
        
        # Get matching points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Compute fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        
        if F is None or F.shape != (3, 3):
            return None
        
        # Perform geometric verification
        geometric_results = geometric_verification(pts1, pts2, F)
        
        # Check if matches pass quality thresholds
        if not verify_match_quality(geometric_results):
            return None
        
        # Visualize results
        viz_path = self.viz_dir / f'{pair_name}_verification.png'
        visualize_geometric_verification(img1, img2, pts1, pts2, geometric_results, viz_path)
        
        # Get inlier matches using mask from geometric verification
        inlier_mask = geometric_results['inlier_mask']
        inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
        
        # Save results
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
        """Save pair matching data with geometric verification results"""
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
    
    def process_image_range(self, start_idx, end_idx):
        """Process all image pairs in range"""
        # Generate all possible pairs
        pairs = []
        for i in range(start_idx, end_idx):
            for j in range(i+1, end_idx+1):
                img1_path = self.image_dir / f'DSC0{i}.JPG'
                img2_path = self.image_dir / f'DSC0{j}.JPG'
                pair_name = f'pair_{i}_{j}'
                pairs.append((img1_path, img2_path, pair_name))
        
        # Process all pairs with progress bar
        for img1_path, img2_path, pair_name in tqdm(pairs, desc="Processing pairs"):
            try:
                result = self.process_image_pair(img1_path, img2_path, pair_name)
                if result is not None:
                    self.results.append(result)
            except Exception as e:
                logging.error(f"Error processing {pair_name}: {str(e)}")
                continue
    
    def save_results(self, output_csv):
        """Save matching results to CSV with geometric verification metrics"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        
        # Print summary
        print("\nMatching Summary:")
        print(f"Total pairs processed: {len(self.results)}")
        print(f"Average matches per pair: {df['num_matches'].mean():.1f}")
        print(f"Average inliers per pair: {df['num_inliers'].mean():.1f}")
        print(f"Average inlier ratio: {df['inlier_ratio'].mean():.3f}")
        print(f"Average reprojection error: {df['reprojection_error'].mean():.3f}")
        print(f"Pairs with well-distributed points: {df['well_distributed'].sum()}")

# Example usage:
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize matcher
    matcher = ImageMatcher("data")
    
    # Process images
    matcher.process_image_range(1161, 1262)
    
    # Save results
    matcher.save_results("data/pair_matches.csv")