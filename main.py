import argparse
import logging
import time
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from utils import (
    SfMExporter,
    StructureFromMotion,
    ImageMatcher
)

def validate_directory(path: str, should_exist: bool = True) -> Path:
    """Validate directory path and create if necessary"""
    path = Path(path)
    if should_exist and not path.exists():
        raise ValueError(f"Directory does not exist: {path}")
    if not should_exist:
        path.mkdir(parents=True, exist_ok=True)
    return path

def validate_numeric_range(value: int, min_val: int, max_val: int, name: str) -> None:
    """Validate numeric arguments are within acceptable ranges"""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

def parse_args():
    parser = argparse.ArgumentParser(description='Structure from Motion Pipeline')
    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform')
    
    # Add global arguments
    parser.add_argument('--log_level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    # Preprocessing parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing')
    preprocess_parser.add_argument('--data_dir', type=str, required=True,
                                help='Path to data directory containing images')
    preprocess_parser.add_argument('--start_idx', type=int, default=0,
                                help='Starting image index (0-999)')
    preprocess_parser.add_argument('--end_idx', type=int, default=35,
                                help='Ending image index (0-999)')
    preprocess_parser.add_argument('--min_matches', type=int, default=150,
                                help='Minimum number of matches (20-1000)')
    preprocess_parser.add_argument('--visualize', action='store_true',
                                help='Show preprocessing visualizations')
    preprocess_parser.add_argument('--mask', type=str,
                                help='Path to mask image')
    
    # Reconstruction parser
    reconstruct_parser = subparsers.add_parser('reconstruct', help='Run reconstruction')
    reconstruct_parser.add_argument('--data_dir', type=str, required=True,
                                help='Path to data directory containing preprocessed data')
    reconstruct_parser.add_argument('--output_dir', type=str, required=True,
                                help='Path to output directory')
    reconstruct_parser.add_argument('--num_images', type=int, default=36,
                                help='Number of images to process (2-1000)')
    reconstruct_parser.add_argument('--export_colmap', action='store_true',
                                help='Export to COLMAP format')
    reconstruct_parser.add_argument('--export_meshlab', action='store_true',
                                help='Export to MeshLab format')
    
    # Full pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--data_dir', type=str, required=True,
                              help='Path to data directory containing images')
    pipeline_parser.add_argument('--output_dir', type=str, required=True,
                              help='Path to output directory')
    pipeline_parser.add_argument('--start_idx', type=int, default=0,
                              help='Starting image index (0-999)')
    pipeline_parser.add_argument('--end_idx', type=int, default=35,
                              help='Ending image index (0-999)')
    pipeline_parser.add_argument('--num_images', type=int, default=36,
                              help='Number of images to process (2-1000)')
    pipeline_parser.add_argument('--export_colmap', action='store_true',
                              help='Export to COLMAP format')
    pipeline_parser.add_argument('--export_meshlab', action='store_true',
                              help='Export to MeshLab format')
    pipeline_parser.add_argument('--mask', type=str,
                              help='Path to mask image')
    
    args = parser.parse_args()
    
    if not args.operation:
        parser.error("Operation required: choose 'preprocess', 'reconstruct', or 'pipeline'")
    
    return args

class SfMPipeline:
    def __init__(self, args):
        self.args = args
        
        # Setup directory structure with error handling
        try:
            self._setup_directories()
            self._validate_inputs()
        except PermissionError as e:
            raise RuntimeError(f"Permission denied while creating directories: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to setup pipeline: {e}")
            
    def _setup_directories(self):
        """Setup directory structure with validation"""
        # Validate data directory
        self.data_dir = validate_directory(self.args.data_dir, should_exist=True)
        
        # Create and validate output directory if needed
        if hasattr(self.args, 'output_dir'):
            self.output_dir = validate_directory(self.args.output_dir, should_exist=False)
            
            # Ensure we have write permissions
            test_file = self.output_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise PermissionError(f"Cannot write to output directory: {e}")
        else:
            self.output_dir = None
            
        # Create subdirectories
        for subdir in ['images', 'matches', 'fundamental', 'correspondences']:
            (self.data_dir / subdir).mkdir(exist_ok=True)
            
        if self.output_dir:
            for subdir in ['reconstruction', 'exports']:
                (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _validate_inputs(self):
        """Validate all input parameters"""
        # Validate numeric ranges
        ranges = {
            'start_idx': (0, 999),
            'end_idx': (0, 999),
            'num_images': (2, 1000),
            'min_matches': (20, 1000)
        }
        
        for param, (min_val, max_val) in ranges.items():
            if hasattr(self.args, param):
                validate_numeric_range(
                    getattr(self.args, param),
                    min_val,
                    max_val,
                    param
                )
        
        # Validate mask file
        if hasattr(self.args, 'mask') and self.args.mask:
            mask_path = Path(self.args.mask)
            if not mask_path.exists():
                raise ValueError(f"Mask file does not exist: {mask_path}")
            try:
                # Verify mask is readable
                cv2.imread(str(mask_path))
            except Exception as e:
                raise ValueError(f"Invalid mask file: {e}")
    
    def run_preprocessing(self):
        """Run feature matching and geometric verification"""
        logger.info("Starting preprocessing...")
        start_time = time.time()
        
        try:
            # Initialize matcher with parameters
            matcher = ImageMatcher(self.data_dir)
            
            # Process image pairs
            matcher.process_image_range(self.args.start_idx, self.args.end_idx, self.args.mask)
            
            # Save results
            matcher.save_results(self.data_dir / 'matching_results.csv')
            
            elapsed_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds!")
            return True
            
        except cv2.error as e:
            logger.error(f"OpenCV error during preprocessing: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def run_reconstruction(self):
        """Run SfM reconstruction and export results"""
        logger.info("Starting reconstruction pipeline...")
        start_time = time.time()
        
        try:
            # Initialize SfM with data directory
            sfm = StructureFromMotion(self.data_dir)
            
            # Run reconstruction
            logger.info("Running Structure from Motion...")
            sfm.run_reconstruction(self.args.num_images)
            
            # Save reconstruction results
            recon_dir = self.output_dir / 'reconstruction'
            logger.info(f"Saving reconstruction to {recon_dir}")
            sfm.save_reconstruction(recon_dir)
            
            # Export results if requested
            if self.args.export_colmap or self.args.export_meshlab:
                logger.info("Initializing exporter...")
                exporter = SfMExporter(recon_dir)
                
                export_dir = self.output_dir / 'exports'
                
                if self.args.export_colmap:
                    logger.info("Exporting to COLMAP format...")
                    colmap_dir = export_dir / 'colmap'
                    colmap_dir.mkdir(exist_ok=True)
                    exporter.export_colmap(colmap_dir)
                    
                if self.args.export_meshlab:
                    logger.info("Exporting to MeshLab format...")
                    meshlab_path = export_dir / 'reconstruction.ply'
                    exporter.export_meshlab(meshlab_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds!")
            return True
            
        except MemoryError:
            logger.error("Insufficient memory for reconstruction")
            return False
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False
    
    def run_full_pipeline(self):
        """Run complete pipeline including preprocessing and reconstruction"""
        try:
            # Run preprocessing
            if not self.run_preprocessing():
                return False
            
            # Run reconstruction
            if not self.run_reconstruction():
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Configure logging with rotation
        log_file = Path('logs') / f'sfm_pipeline_{time.strftime("%Y%m%d_%H%M%S")}.log'
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        
        # Initialize and run pipeline
        pipeline = SfMPipeline(args)
        
        success = False
        if args.operation == 'preprocess':
            success = pipeline.run_preprocessing()
        elif args.operation == 'reconstruct':
            success = pipeline.run_reconstruction()
        elif args.operation == 'pipeline':
            success = pipeline.run_full_pipeline()
            
        return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    finally:
        logging.shutdown()

if __name__ == "__main__":
    sys.exit(main())