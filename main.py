import argparse
import logging
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    ImageMatcher,
    RobustSfM,
    run_bundle_adjustment,
    visualize_cameras_and_points,
    visualize_point_cloud,
    DenseReconstruction,
    save_reconstruction,
    save_dense_reconstruction,
    export_colmap_format
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Structure from Motion Pipeline')
    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform')
    
    # Preprocessing parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing')
    preprocess_parser.add_argument('--data_dir', type=str, required=True,
                                help='Path to data directory containing images')
    preprocess_parser.add_argument('--start_idx', type=int, default=0,
                                help='Starting image index')
    preprocess_parser.add_argument('--end_idx', type=int, default=40,
                                help='Ending image index')
    preprocess_parser.add_argument('--min_matches', type=int, default=150,
                                help='Minimum number of matches')
    preprocess_parser.add_argument('--visualize', action='store_true',
                                help='Show preprocessing visualizations')
    
    # Reconstruction parser
    reconstruct_parser = subparsers.add_parser('reconstruct', help='Run reconstruction')
    reconstruct_parser.add_argument('--data_dir', type=str, required=True,
                                help='Path to data directory containing preprocessed data')
    reconstruct_parser.add_argument('--output_dir', type=str, required=True,
                                help='Path to output directory')
    reconstruct_parser.add_argument('--min_inliers', type=int, default=100,
                                help='Minimum number of inliers for a valid match')
    reconstruct_parser.add_argument('--min_inlier_ratio', type=float, default=0.5,
                                help='Minimum inlier ratio for a valid match')
    reconstruct_parser.add_argument('--good_inlier_ratio', type=float, default=0.8,
                                help='Threshold for very good pairs')
    reconstruct_parser.add_argument('--skip_dense', action='store_true',
                                help='Skip dense reconstruction')
    reconstruct_parser.add_argument('--visualize', action='store_true',
                                help='Show reconstruction visualizations')
    
    # Full pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--data_dir', type=str, required=True,
                              help='Path to data directory containing images')
    pipeline_parser.add_argument('--output_dir', type=str, required=True,
                              help='Path to output directory')
    pipeline_parser.add_argument('--start_idx', type=int, default=0,
                              help='Starting image index')
    pipeline_parser.add_argument('--end_idx', type=int, default=40,
                              help='Ending image index')
    pipeline_parser.add_argument('--min_inliers', type=int, default=100,
                              help='Minimum number of inliers for a valid match')
    pipeline_parser.add_argument('--min_inlier_ratio', type=float, default=0.5,
                              help='Minimum inlier ratio for a valid match')
    pipeline_parser.add_argument('--good_inlier_ratio', type=float, default=0.8,
                              help='Threshold for very good pairs')
    pipeline_parser.add_argument('--skip_dense', action='store_true',
                              help='Skip dense reconstruction')
    pipeline_parser.add_argument('--visualize', action='store_true',
                              help='Show visualizations')
    
    return parser.parse_args()

class SfMPipeline:
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else None
        
        # Create directory structure
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'sparse').mkdir(exist_ok=True)
            if not args.skip_dense:
                (self.output_dir / 'dense').mkdir(exist_ok=True)
                
        # Ensure required directories exist in data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'images').mkdir(exist_ok=True)
        (self.data_dir / 'matches').mkdir(exist_ok=True)
        (self.data_dir / 'fundamental').mkdir(exist_ok=True)
        (self.data_dir / 'correspondences').mkdir(exist_ok=True)
    
    def run_preprocessing(self):
        """Run feature matching and geometric verification"""
        logger.info("Starting preprocessing...")
        start_time = time.time()
        
        try:
            # Initialize matcher with parameters
            matcher = ImageMatcher(self.data_dir)
            matcher.min_matches = self.args.min_matches if hasattr(self.args, 'min_matches') else 150
            
            # Process image pairs
            matcher.process_image_range(self.args.start_idx, self.args.end_idx)
            
            # Save results
            matcher.save_results(self.data_dir / 'pair_matches.csv')
            
            elapsed_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds!")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def run_reconstruction(self):
        """Run SfM reconstruction using RobustSfM"""
        logger.info("Starting reconstruction...")
        start_time = time.time()
        
        try:
            # Initialize RobustSfM with parameters
            sfm = RobustSfM(self.data_dir)
            
            # Set parameters if provided
            if hasattr(self.args, 'min_inliers'):
                sfm.min_inliers = self.args.min_inliers
            if hasattr(self.args, 'min_inlier_ratio'):
                sfm.min_inlier_ratio = self.args.min_inlier_ratio
            if hasattr(self.args, 'good_inlier_ratio'):
                sfm.good_inlier_ratio = self.args.good_inlier_ratio
            
            # Run reconstruction
            reconstruction = sfm.reconstruct(str(self.data_dir / 'pair_matches.csv'))
            
            # Visualize if requested
            if self.args.visualize:
                logger.info("Visualizing reconstruction...")
                visualize_cameras_and_points(reconstruction)
            
            # Dense reconstruction if not skipped
            if not self.args.skip_dense:
                logger.info("Starting dense reconstruction...")
                dense = DenseReconstruction(reconstruction, self.data_dir)
                points, colors = dense.create_dense_point_cloud()
                mesh = dense.create_mesh(points, dense.estimate_normals(points))
                
                if self.args.visualize:
                    visualize_point_cloud(points, colors)
            
            # Save results
            logger.info("Saving reconstruction results...")
            save_reconstruction(reconstruction, self.output_dir / 'sparse')
            
            if not self.args.skip_dense:
                save_dense_reconstruction(points, colors, mesh, self.output_dir / 'dense')
            
            # Export to COLMAP format
            export_colmap_format(reconstruction, self.output_dir / 'colmap')
            
            elapsed_time = time.time() - start_time
            logger.info(f"Reconstruction completed in {elapsed_time:.2f} seconds!")
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            raise
    
    def run_full_pipeline(self):
        """Run complete pipeline including preprocessing and reconstruction"""
        try:
            # Run preprocessing
            self.run_preprocessing()
            
            # Run reconstruction
            self.run_reconstruction()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    args = parse_args()
    
    try:
        pipeline = SfMPipeline(args)
        
        if args.operation == 'preprocess':
            pipeline.run_preprocessing()
        elif args.operation == 'reconstruct':
            pipeline.run_reconstruction()
        elif args.operation == 'pipeline':
            pipeline.run_full_pipeline()
        else:
            logger.error("Invalid operation specified")
            
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()