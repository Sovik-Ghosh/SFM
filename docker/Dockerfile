# Runtime image: stage 1 builds with the CUDA toolkit; stage 2 ships the CUDA runtime libraries
# (cudart, cuBLAS, cuSOLVER) and the installed programs.
#   docker build -f docker/Dockerfile -t sfm .
#   docker run --rm --gpus all -v "$PWD:/work" -w /work sfm sfm_run configs/bunny_sift_tuned2.json
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN echo "Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
# The runtime stage has the CUDA libraries in /usr/local/cuda/lib64 (same path in both stages).
RUN cmake --preset release -G Ninja -DSFM_BUILD_TESTS=OFF -DSFM_BUILD_VIEWER=OFF \
      -DCMAKE_INSTALL_RPATH=/usr/local/cuda/lib64 \
    && cmake --build --preset release \
    && cmake --install build/release --prefix /opt/sfm

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04 AS runtime
COPY --from=build /opt/sfm /opt/sfm
ENV PATH=/opt/sfm/bin:$PATH
WORKDIR /work
CMD ["sfm_version"]
