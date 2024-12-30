import numpy as np
import json
import open3d as o3d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_reconstruction(reconstruction: dict, output_dir: Path) -> None:
    """
    Save reconstruction data to files
    
    Args:
        reconstruction: Dictionary containing reconstruction data
        output_dir: Output directory path
    
    Raises:
        ValueError: If required reconstruction data is missing
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate reconstruction data
        required_keys = ['cameras', 'points_3d', 'images', 'image_points', 'camera_matrix']
        for key in required_keys:
            if key not in reconstruction:
                raise ValueError(f"Missing required key in reconstruction: {key}")
        
        # Save camera parameters
        cameras = []
        for i, camera in enumerate(reconstruction['cameras']):
            camera_data = {
                'id': i,
                'matrix': camera.tolist(),
                'image_id': str(reconstruction['images'][i])
            }
            cameras.append(camera_data)
        
        with open(output_dir / 'cameras.json', 'w') as f:
            json.dump(cameras, f, indent=2)
        
        # Save sparse point cloud with better error handling
        if len(reconstruction['points_3d']) > 0:
            sparse_pcd = o3d.geometry.PointCloud()
            sparse_pcd.points = o3d.utility.Vector3dVector(reconstruction['points_3d'])
            if not o3d.io.write_point_cloud(str(output_dir / 'sparse.ply'), sparse_pcd):
                logger.warning("Failed to save sparse point cloud")
        
        # Save camera matrix
        np.save(output_dir / 'camera_matrix.npy', reconstruction['camera_matrix'])
        
        # Save image points
        points_dir = output_dir / 'image_points'
        points_dir.mkdir(exist_ok=True)
        for i, points in enumerate(reconstruction['image_points']):
            np.save(points_dir / f'image_{i:06d}.npy', points)
        
        logger.info(f"Saved reconstruction to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save reconstruction: {e}")
        raise

def save_dense_reconstruction(points: np.ndarray, colors: np.ndarray, 
                            mesh: o3d.geometry.TriangleMesh, 
                            output_dir: Path) -> None:
    """
    Save dense reconstruction results
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (optional)
        mesh: Open3D triangle mesh
        output_dir: Output directory path
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure dense subdirectory exists
        dense_dir = output_dir / 'dense'
        dense_dir.mkdir(exist_ok=True)
        
        # Save dense point cloud
        if points is not None and len(points) > 0:
            dense_pcd = o3d.geometry.PointCloud()
            dense_pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None and len(colors) == len(points):
                dense_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if not o3d.io.write_point_cloud(str(dense_dir / 'dense.ply'), dense_pcd):
                logger.warning("Failed to save dense point cloud")
        
        # Save mesh if available
        if mesh is not None and len(mesh.vertices) > 0:
            # Save full resolution mesh
            if not o3d.io.write_triangle_mesh(str(dense_dir / 'mesh.ply'), mesh):
                logger.warning("Failed to save full resolution mesh")
            
            try:
                # Save decimated mesh for visualization
                decimated_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=min(len(mesh.triangles) // 4, 100000)
                )
                o3d.io.write_triangle_mesh(str(dense_dir / 'mesh_simplified.ply'), 
                                         decimated_mesh)
            except Exception as e:
                logger.warning(f"Failed to create simplified mesh: {e}")
        
        logger.info(f"Saved dense reconstruction to {dense_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save dense reconstruction: {e}")
        raise

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
    
    Returns:
        np.ndarray: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])

def export_colmap_format(reconstruction: dict, output_dir: Path) -> None:
    """
    Export reconstruction to COLMAP format
    
    Args:
        reconstruction (dict): Dictionary containing reconstruction data
        output_dir (Path): Output directory path
    
    Raises:
        ValueError: If required reconstruction data is missing
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate required keys
        required_keys = ['cameras', 'images', 'image_points', 'points_3d', 'camera_matrix']
        for key in required_keys:
            if key not in reconstruction:
                raise ValueError(f"Missing required key in reconstruction: {key}")
        
        # Export cameras
        with open(output_dir / 'cameras.txt', 'w') as f:
            # Write header
            f.write('# Camera list with one line of data per camera:\n')
            f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            
            # Write camera parameters
            K = reconstruction['camera_matrix']
            # Estimate image width and height based on camera matrix
            width = int(K[0,2] * 2)
            height = int(K[1,2] * 2)
            f.write(f'1 SIMPLE_PINHOLE {width} {height} {K[0,0]} {K[0,2]} {K[1,2]}\n')
        
        # Export images
        with open(output_dir / 'images.txt', 'w') as f:
            # Write header
            f.write('# Image list with two lines of data per image:\n')
            f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
            f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
            
            for i, image_id in enumerate(reconstruction['images']):
                # Get camera pose
                R = reconstruction['cameras'][i][:3, :3]
                t = reconstruction['cameras'][i][:3, 3]
                
                # Convert rotation matrix to quaternion
                q = rotation_matrix_to_quaternion(R)
                
                # Construct image name from ID
                image_name = f"{image_id:06d}.jpg"
                
                # Write camera pose
                f.write(f'{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {image_name}\n')
                
                # Write image points
                points = reconstruction['image_points'][i]
                point_str = ' '.join(f'{x} {y} {j+1}' for j, (x, y) in enumerate(points))
                f.write(f'{point_str}\n')
        
        # Export points
        with open(output_dir / 'points3D.txt', 'w') as f:
            # Write header
            f.write('# 3D point list with one line of data per point:\n')
            f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
            
            for i, point in enumerate(reconstruction['points_3d']):
                # Use default color (white) if no color information
                f.write(f'{i+1} {point[0]} {point[1]} {point[2]} 255 255 255 1.0')
                
                # Write track information (which images see this point)
                track_str = ''
                for j, image_points in enumerate(reconstruction['image_points']):
                    for k, pt in enumerate(image_points):
                        if np.allclose(pt, points[i]):
                            track_str += f' {j+1} {k}'
                
                f.write(f'{track_str}\n')
        
        logger.info(f"Exported reconstruction to COLMAP format in {output_dir}")
    
    except Exception as e:
        logger.error(f"Failed to export COLMAP format: {e}")
        raise

def save_camera_trajectory(reconstruction: dict, output_path: Path) -> None:
    """
    Save camera trajectory for visualization
    
    Args:
        reconstruction (dict): Dictionary containing camera matrices
        output_path (Path): Path to save the trajectory numpy file
    
    Raises:
        ValueError: If no cameras are found in the reconstruction
    """
    if 'cameras' not in reconstruction or len(reconstruction['cameras']) == 0:
        raise ValueError("No cameras found in reconstruction")
    
    trajectory = []
    for camera in reconstruction['cameras']:
        # Extract camera center
        # Camera center is calculated as -R.T @ t
        # Where R is the rotation matrix and t is the translation vector
        R = camera[:3, :3]
        t = camera[:3, 3]
        C = -R.T @ t
        
        # Add to trajectory
        trajectory.append(C)
    
    trajectory = np.array(trajectory)
    np.save(output_path, trajectory)

def export_meshlab_project(reconstruction_dir: Path, output_path: Path = None) -> None:
    """
    Export project file for MeshLab
    
    Args:
        reconstruction_dir (Path): Directory containing reconstruction outputs
        output_path (Path, optional): Path to save the MeshLab project file. 
                                     If None, saves in the reconstruction directory.
    
    Raises:
        ValueError: If reconstruction directory does not exist
    """
    reconstruction_dir = Path(reconstruction_dir)
    if not reconstruction_dir.is_dir():
        raise ValueError(f"Reconstruction directory does not exist: {reconstruction_dir}")
    
    # Use reconstruction_dir if no specific output path provided
    if output_path is None:
        output_path = reconstruction_dir / 'project.mlp'
    else:
        output_path = Path(output_path)
    
    # Prepare potential mesh and point cloud file paths
    sparse_ply = reconstruction_dir / 'sparse.ply'
    dense_ply = reconstruction_dir / 'dense.ply'
    mesh_ply = reconstruction_dir / 'mesh.ply'
    
    # Attempt to use dense subdirectory if exists
    dense_subdir = reconstruction_dir / 'dense'
    if dense_subdir.is_dir():
        dense_ply = dense_subdir / 'dense.ply'
        mesh_ply = dense_subdir / 'mesh.ply'
    
    # Write MeshLab project file
    with open(output_path, 'w') as f:
        f.write('<!DOCTYPE MeshLabProject>\n')
        f.write('<MeshLabProject>\n')
        
        # Start mesh group
        f.write(' <MeshGroup>\n')
        
        # Add sparse point cloud
        if sparse_ply.exists():
            f.write(f'  <MLMesh filename="{sparse_ply.relative_to(reconstruction_dir)}" label="Sparse Points">\n')
            f.write('   <MLMatrix44>\n')
            f.write('    1 0 0 0 \n')
            f.write('    0 1 0 0 \n')
            f.write('    0 0 1 0 \n')
            f.write('    0 0 0 1 \n')
            f.write('   </MLMatrix44>\n')
            f.write('  </MLMesh>\n')
        
        # Add dense point cloud if exists
        if dense_ply.exists():
            f.write(f'  <MLMesh filename="{dense_ply.relative_to(reconstruction_dir)}" label="Dense Points">\n')
            f.write('   <MLMatrix44>\n')
            f.write('    1 0 0 0 \n')
            f.write('    0 1 0 0 \n')
            f.write('    0 0 1 0 \n')
            f.write('    0 0 0 1 \n')
            f.write('   </MLMatrix44>\n')
            f.write('  </MLMesh>\n')
        
        # Add mesh if exists
        mesh_ply = reconstruction_dir / 'dense' / 'mesh.ply'
        if mesh_ply.exists():
            f.write(f'  <MLMesh filename="dense/mesh.ply" label="Surface Mesh">\n')
            f.write('   <MLMatrix44>\n')
            f.write('    1 0 0 0 \n')
            f.write('    0 1 0 0 \n')
            f.write('    0 0 1 0 \n')
            f.write('    0 0 0 1 \n')
            f.write('   </MLMatrix44>\n')
            f.write('  </MLMesh>\n')
        
        f.write(' </MeshGroup>\n')
        f.write('</MeshLabProject>\n')