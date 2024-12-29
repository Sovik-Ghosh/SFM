import numpy as np
import json
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation

def save_reconstruction(reconstruction, output_dir):
    """
    Save reconstruction data to files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save camera parameters
    cameras = []
    for i, camera in enumerate(reconstruction['cameras']):
        camera_data = {
            'id': i,
            'matrix': camera.tolist(),
            'image': str(reconstruction['images'][i])
        }
        cameras.append(camera_data)
    
    with open(output_dir / 'cameras.json', 'w') as f:
        json.dump(cameras, f, indent=2)
    
    # Save sparse point cloud
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(reconstruction['points_3d'])
    o3d.io.write_point_cloud(str(output_dir / 'sparse.ply'), sparse_pcd)
    
    # Save camera matrix
    np.save(output_dir / 'camera_matrix.npy', reconstruction['camera_matrix'])
    
    # Save image points
    for i, points in enumerate(reconstruction['image_points']):
        np.save(output_dir / f'image_points_{i}.npy', points)

def save_dense_reconstruction(points, colors, mesh, output_dir):
    """
    Save dense reconstruction results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dense point cloud with colors
    dense_pcd = o3d.geometry.PointCloud()
    dense_pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        dense_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_dir / 'dense.ply'), dense_pcd)
    
    # Save mesh if available
    if mesh is not None:
        # Save full resolution mesh
        o3d.io.write_triangle_mesh(str(output_dir / 'mesh.ply'), mesh)
        
        # Save decimated mesh for visualization
        decimated_mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=len(mesh.triangles) // 4
        )
        o3d.io.write_triangle_mesh(str(output_dir / 'mesh_simplified.ply'), 
                                 decimated_mesh)

def export_colmap_format(reconstruction, output_dir):
    """
    Export reconstruction in COLMAP format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write cameras file
    with open(output_dir / 'cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        # Assuming all images have same size and camera matrix
        f.write('1 SIMPLE_PINHOLE 1000 1000 %f %f %f\n' % 
                (reconstruction['camera_matrix'][0,0],
                 reconstruction['camera_matrix'][0,2],
                 reconstruction['camera_matrix'][1,2]))
    
    # Write images file
    with open(output_dir / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        
        for i, camera in enumerate(reconstruction['cameras']):
            # Extract rotation matrix and translation
            R = camera[:3, :3]
            t = camera[:3, 3]
            
            # Convert rotation matrix to quaternion
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # [x, y, z, w]
            # Reorder to COLMAP format [w, x, y, z]
            quat = np.roll(quat, 1)
            
            # Write camera pose
            image_name = Path(reconstruction['images'][i]).name
            f.write('%d %f %f %f %f %f %f %f 1 %s\n' % 
                   (i+1, quat[0], quat[1], quat[2], quat[3], t[0], t[1], t[2], 
                    image_name))
            
            # Write point observations
            points_2d = reconstruction['image_points'][i]
            points_line = []
            for j, pt in enumerate(points_2d):
                points_line.append(f'{pt[0]} {pt[1]} {j+1}')
            f.write(' '.join(points_line) + '\n')
    
    # Write points3D file
    with open(output_dir / 'points3D.txt', 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        for i, point in enumerate(reconstruction['points_3d']):
            # Each 3D point needs at least one observation
            track_string = '1 0'  # Image_id and point2D_idx
            f.write('%d %f %f %f 255 255 255 0 %s\n' % 
                   (i+1, point[0], point[1], point[2], track_string))

def save_camera_trajectory(reconstruction, output_path):
    """
    Save camera trajectory for visualization
    """
    trajectory = []
    for camera in reconstruction['cameras']:
        # Extract camera center
        R = camera[:3, :3]
        t = camera[:3, 3]
        C = -R.T @ t
        
        # Add to trajectory
        trajectory.append(C)
    
    trajectory = np.array(trajectory)
    np.save(output_path, trajectory)

def export_meshlab_project(reconstruction, output_dir):
    """
    Export project file for MeshLab
    """
    output_dir = Path(output_dir)
    meshlab_file = output_dir / 'project.mlp'
    
    with open(meshlab_file, 'w') as f:
        f.write('<!DOCTYPE MeshLabProject>\n')
        f.write('<MeshLabProject>\n')
        
        # Add sparse point cloud
        f.write(' <MeshGroup>\n')
        f.write(f'  <MLMesh filename="sparse.ply" label="Sparse Points">\n')
        f.write('   <MLMatrix44>\n')
        f.write('    1 0 0 0 \n')
        f.write('    0 1 0 0 \n')
        f.write('    0 0 1 0 \n')
        f.write('    0 0 0 1 \n')
        f.write('   </MLMatrix44>\n')
        f.write('  </MLMesh>\n')
        
        # Add dense point cloud if exists
        dense_ply = output_dir / 'dense' / 'dense.ply'
        if dense_ply.exists():
            f.write(f'  <MLMesh filename="dense/dense.ply" label="Dense Points">\n')
            f.write('   <MLMatrix44>\n')
            f.write('    1 0 0 0 \n')
            f.write('    0 1 0 0 \n')
            f.write('    0 0 1 0 \n')
            f.write('    0 0 0 1 \n')
            f.write('   </MLMatrix44>\n')
            f.write('  </MLMesh>\n')
        
        # Add mesh if exists
        mesh_ply = output_dir / 'dense' / 'mesh.ply'
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