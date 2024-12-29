import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def visualize_cameras_and_points(reconstruction):
    """
    Visualize cameras and sparse point cloud
    """
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    points = reconstruction['points_3d']
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', s=1)
    
    # Plot camera positions and orientations
    for camera in reconstruction['cameras']:
        R = camera[:, :3]
        t = camera[:, 3]
        
        # Camera center
        center = -R.T @ t
        ax.scatter(center[0], center[1], center[2], c='r', marker='o')
        
        # Camera axes
        axis_length = 0.5
        for i, color in enumerate(['r', 'g', 'b']):
            axis = R.T[:, i] * axis_length
            ax.quiver(center[0], center[1], center[2],
                     axis[0], axis[1], axis[2],
                     color=color)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SfM Reconstruction')
    
    plt.show()

def create_open3d_point_cloud(points, colors=None):
    """
    Create Open3D point cloud from numpy array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_point_cloud(points, colors=None):
    """
    Visualize point cloud using Open3D
    """
    pcd = create_open3d_point_cloud(points, colors)
    o3d.visualization.draw_geometries([pcd])

def filter_point_cloud(points, voxel_size=0.05):
    """
    Filter and downsample point cloud
    """
    pcd = create_open3d_point_cloud(points)
    
    # Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Statistical outlier removal
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20,
                                                std_ratio=2.0)
    
    # Extract filtered points
    filtered_points = np.asarray(cl.points)
    
    return filtered_points

def estimate_normals(points, camera_centers):
    """
    Estimate point cloud normals using camera positions
    """
    pcd = create_open3d_point_cloud(points)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Orient normals towards cameras
    camera_center = np.mean(camera_centers, axis=0)
    pcd.orient_normals_towards_camera_location(camera_center)
    
    return np.asarray(pcd.normals)