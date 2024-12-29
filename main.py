from utils import matching
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

CURR_DIR = Path(__file__).resolve().parent

def save_pair_data(kp1, kp2, matches, pair_name, base_path="data"):
    """
    Save feature matching data for an image pair
    """
    # Create directory structure
    data_dir = CURR_DIR / Path(base_path)
    correspondences_dir = data_dir / "correspondences"
    matches_dir = data_dir / "matches"
    fundamental_dir = data_dir / "fundamental"
    
    # Create directories if they don't exist
    for dir_path in [correspondences_dir, matches_dir, fundamental_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if len(matches) > 0:
        # Get corresponding points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Compute Fundamental Matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        
        # Save all data
        np.save(correspondences_dir / f"{pair_name}_pts1.npy", pts1)
        np.save(correspondences_dir / f"{pair_name}_pts2.npy", pts2)
        np.savez(matches_dir / f"{pair_name}_matches.npz", 
                queryIdx=np.array([m.queryIdx for m in matches]),
                trainIdx=np.array([m.trainIdx for m in matches]),
                distance=np.array([m.distance for m in matches]))
        np.savez(fundamental_dir / f"{pair_name}_F.npz", 
                F=F, mask=mask, pts1=pts1, pts2=pts2)
        
        return F, mask, len(matches), np.sum(mask)
    return None, None, 0, 0

def generate_and_save_all_pairs(start=1161, end=1262, base_path="data"):
    """
    Generate all possible image pairs
    """
    pairs = []
    total_pairs = ((end - start + 1) * (end - start)) // 2
    
    with tqdm(total=total_pairs, desc="Generating pairs") as pbar:
        for i in range(start, end):
            for j in range(i+1, end+1):
                img1_path = CURR_DIR / Path(f"data/images/DSC0{i}.JPG")
                img2_path = CURR_DIR / Path(f"data/images/DSC0{j}.JPG")
                pair_name = f"pair_{i}_{j}"
                pairs.append((img1_path, img2_path, pair_name))
                pbar.update(1)
    return pairs

def process_all_pairs(pairs, csv_path='data/pair_matches.csv'):
    """
    Process all pairs using matching function and save to CSV
    """
    # Prepare CSV file
    csv_file_path = CURR_DIR / csv_path
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Pair', 'Image1', 'Image2', 'Total Matches', 'Inlier Matches'])
    
    results = {}
    
    for img1_path, img2_path, pair_name in tqdm(pairs, desc="Processing pairs"):
        try:
            # Read images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                tqdm.write(f"Error: Could not read images for {pair_name}")
                continue
                
            # Get matches
            kp1, kp2, matches = matching(img1, img2)
            
            # Save data
            F, mask, num_matches, num_inliers = save_pair_data(kp1, kp2, matches, pair_name)
            
            if F is not None:
                # Append to CSV
                with open(csv_file_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([
                        pair_name, 
                        img1_path.name, 
                        img2_path.name, 
                        num_matches, 
                        num_inliers
                    ])
                
                results[pair_name] = {
                    'F': F,
                    'mask': mask,
                    'num_matches': num_matches,
                    'num_inliers': num_inliers
                }
                tqdm.write(f"{pair_name}: {num_matches} matches, {num_inliers} inliers")
                
        except Exception as e:
            tqdm.write(f"Error processing {pair_name}: {e}")
            continue
    
    return results

# Usage
if __name__ == "__main__":
    pairs = generate_and_save_all_pairs(1161, 1262)
    results = process_all_pairs(pairs)