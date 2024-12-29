sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran

cd
mkdir opencv
cd opencv
wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip -O opencv.zip
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.zip -O opencv_contrib.zip
unzip opencv.zip
unzip opencv_contrib.zip
rm -f *.zip

cd ~/opencv/opencv-4.10.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
      -D PYTHON_EXECUTABLE=$(which python3) \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON_INCLUDE_DIRS=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
      -D PYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.so')") \
      -D PYTHON_LIBRARIES=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.so')") \
      -D PYTHON_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_dnn=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      ..
make -j$(nproc)