"bunny" data set -- 36 views sampled on a hemisphere

This directory contains images, corresponding silhouettes and camera calibration parameters. Note that the object may be partially outside the field of view in some images.

All images are stored in the subdirectory ./images in PPM file format.

Corresponding silhouettes are stored in the subdirectory ./silhouettes in PGM file format. Pixels with intensity values 0 (black) denote object occupancy, whereas all other pixels represent the background. 

Camera parameter files are stored in the subdirectory ./calib in txt file format. Each camera parameter file has the following format:

-------------------------------------------
CONTOUR
P[0][0] P[0][1] P[0][2] P[0][3]
P[1][0] P[1][1] P[1][2] P[1][3]
P[2][0] P[2][1] P[2][2] P[2][3]
-------------------------------------------

"CONTOUR" is just a header and should be ignored. P[3][4] denotes the full 3x4 projection matrix. If X denotes a homogeneous 3D coordinate of a point, and u denotes a homogeneous 2D coordinate of its image projection, then X and u are related by the following equation:

P*X = d*u,

where d is the depth of the point with respect to the camera.

An approximate bounding box for the model is:

x: [-7.5, 7.5]
y: [-10, 10]
z: [-7.5, 15]

Created by Kalin Kolev and Daniel Cremers, University of Bonn, Germany.
