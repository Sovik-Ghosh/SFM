import numpy as np
import json
from pathlib import Path
import sqlite3
import logging
from typing import Dict, List, Tuple, Union, Any

class SfMExporter:
    def __init__(self, reconstruction_dir: str):
        self.recon_dir = Path(reconstruction_dir)
        self._load_reconstruction()

    def _load_reconstruction(self):
        try:
            # Load reconstruction data
            with open(self.recon_dir / 'poses.json', 'r') as f:
                self.poses = json.load(f)
                logging.info(f"Loaded poses for {len(self.poses)} images")

            with open(self.recon_dir / 'points3D.json', 'r') as f:
                points_data = json.load(f)
                self.points3D = points_data['points3D']
                self.tracks = points_data['tracks']
                
                # Debug information
                logging.info(f"Initially loaded {len(self.points3D)} points")
                track_lengths = [len(track) for track in self.tracks]
                logging.info(f"Track length stats - Min: {min(track_lengths)}, Max: {max(track_lengths)}, Average: {np.mean(track_lengths):.2f}")

            # Filter out points with less than 2 observations
            valid_points = []
            valid_tracks = []
            for point, track in zip(self.points3D, self.tracks):
                if len(track) >= 2:
                    valid_points.append(point)
                    valid_tracks.append(track)

            self.points3D = valid_points
            self.tracks = valid_tracks
            logging.info(f"After filtering: {len(self.points3D)} valid points")

            # Debug: Print sample point and track
            if len(self.points3D) > 0:
                logging.info(f"Sample point: {self.points3D[0]}")
                logging.info(f"Sample track: {self.tracks[0]}")

        except FileNotFoundError as e:
            raise ValueError(f"Failed to load reconstruction data: {e}")

    def export_colmap(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export cameras.txt
        cameras_path = output_dir / 'cameras.txt'
        with open(cameras_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("1 PINHOLE 1024 768 2393.95 2398.12 932.38 628.26\n")
            logging.info(f"Wrote camera parameters to {cameras_path}")

        # Export images.txt
        images_path = output_dir / 'images.txt'
        with open(images_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            total_point_refs = 0
            for img_id, pose_data in self.poses.items():
                R = np.array(pose_data['R'])
                t = np.array(pose_data['t']).reshape(3)

                # Convert to quaternion
                q = self._rotation_matrix_to_quaternion(R)
                qw, qx, qy, qz = q
                
                # Write first line - camera pose
                image_line = f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {int(img_id):08d}.jpg"
                f.write(f"{image_line}\n")
                
                # Write second line - point observations
                points2D = []
                point_id_refs = []
                for point_idx, track in enumerate(self.tracks):
                    if str(img_id) in track:
                        x, y = track[str(img_id)]
                        points2D.append(f"{x} {y} {point_idx + 1}")
                        point_id_refs.append(point_idx + 1)
                f.write(f"{' '.join(points2D)}\n")
                
                total_point_refs += len(points2D)
                if len(points2D) > 0:
                    logging.info(f"Image {img_id}: Added {len(points2D)} point references")
            
            logging.info(f"Total point references in images.txt: {total_point_refs}")

        # Export points3D.txt
        points_path = output_dir / 'points3D.txt'
        with open(points_path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            points_written = 0
            for idx, (point, track) in enumerate(zip(self.points3D, self.tracks)):
                point_id = idx + 1
                x, y, z = point
                r, g, b = 255, 255, 255
                error = 1.0

                # Build track information
                track_elements = []
                for image_id in sorted(track.keys()):
                    track_elements.append(f"{image_id} 0")

                if len(track_elements) >= 2:  # Only write points visible in at least 2 images
                    track_str = ' '.join(track_elements)
                    f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {error} {track_str}\n")
                    points_written += 1

            logging.info(f"Wrote {points_written} points to points3D.txt")

    def _rotation_matrix_to_quaternion(self, R):
        """More stable rotation matrix to quaternion conversion"""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
        return qw, qx, qy, qz

    def _create_colmap_database(self, db_path: Path):
        """Create empty COLMAP database with just camera parameters"""
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Create tables
        c.execute('''CREATE TABLE cameras
                    (camera_id INTEGER PRIMARY KEY, model INTEGER,
                     width INTEGER, height INTEGER, params BLOB)''')
                     
        c.execute('''CREATE TABLE images
                    (image_id INTEGER PRIMARY KEY, name TEXT,
                     camera_id INTEGER, prior_qw REAL, prior_qx REAL,
                     prior_qy REAL, prior_qz REAL, prior_tx REAL,
                     prior_ty REAL, prior_tz REAL)''')

        try:
            # Insert camera (PINHOLE model = 1)
            camera_params = np.array([2393.95, 2398.12, 932.38, 628.26], dtype=np.float64)
            c.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?)",
                     (1, 1, 1024, 768, camera_params.tobytes()))
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def export_all(self, output_dir: str):
        """Export all formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export COLMAP formats
        colmap_dir = output_dir / 'colmap'
        colmap_dir.mkdir(exist_ok=True)
        
        self._create_colmap_database(colmap_dir / 'database.db')
        self.export_colmap(colmap_dir)
        
        logging.info(f"Exported all formats to {output_dir}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        base_dir = Path("/teamspace/studios/this_studio/SFM/bunny_data")
        reconstruction_dir = base_dir / "reconstruction"
        export_dir = base_dir / "exports"

        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing reconstruction from {reconstruction_dir}")
        exporter = SfMExporter(reconstruction_dir)
        exporter.export_all(export_dir)
        
        logging.info("Export completed successfully")

    except Exception as e:
        logging.error(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    main()