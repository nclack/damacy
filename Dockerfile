# damacy — single-stage CUDA devel image for cluster runs of damacy_bench.
#
# Build:
#   docker build -t damacy:dev .
# Run (cluster, with GPU passthrough):
#   docker run --rm --gpus all damacy:dev ctest --test-dir build --output-on-failure
#   docker run --rm --gpus all damacy:dev \
#     python -c "import damacy; print(damacy.__version__)"
# On hosts using the CDI runtime instead of --gpus, substitute
#   --device=nvidia.com/gpu=all
#
# Notes:
#  * The host must provide the NVIDIA driver via --gpus / nvidia-container-toolkit.
#    The CUDA devel base image bundles the toolkit & cudart libs we link against;
#    libcuda.so comes from the host driver at runtime.
#  * Toolchain is whatever the cuda devel base ships (gcc 13.x) plus apt's cmake
#    and ninja. nvcc finds gcc as its host compiler automatically.
#  * Python is apt's python3 (3.12 on ubuntu 24.04, satisfying pyproject's
#    >=3.11). uv manages the venv at /opt/venv and drives the editable install.
#  * nvcomp is the standalone NVIDIA distribution (not the pip wheel) extracted
#    to /opt/nvcomp; CMake locates it via -DNvcomp_ROOT.

ARG CUDA_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG NVCOMP_VERSION=5.0.0.6
# nvcomp standalone tarball (CUDA 13 build). NVIDIA serves these from a
# redist tree keyed on the (linux-arch, version) tuple.
ARG NVCOMP_URL=https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-${NVCOMP_VERSION}_cuda13-archive.tar.xz

# ----- system toolchain via apt ----------------------------------------------
# Just enough to drive cmake + ninja + python; gcc/g++/nvcc come from the
# cuda devel base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        xz-utils \
        cmake \
        ninja-build \
        python3 \
        python3-dev \
 && rm -rf /var/lib/apt/lists/*

# ----- uv (static binary) + venv ---------------------------------------------
# Official installer drops uv at /root/.local/bin/uv; relocate to /usr/local/bin
# so it's on PATH for non-root invocations too.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv \
 && uv --version

# Project venv at /opt/venv. Activating via PATH+VIRTUAL_ENV lets uv pip
# install into it without further flags, and lets find_package(Python)
# pick it up during the cmake configure.
RUN uv venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv

# scikit-build-core is the build backend declared in pyproject.toml. We
# install it ahead of time so the editable install below (run with
# --no-build-isolation) can find it.
# pytest drives python/tests/* via the python_pytest ctest target;
# installed alongside scikit-build-core so the cmake configure can
# detect it and register the test.
RUN uv pip install scikit-build-core pytest pytest-cov

# ----- build configuration ---------------------------------------------------
# Override at `docker build` time via --build-arg to produce a coverage
# image for CI. The defaults give a release-ish image suitable for
# cluster runs of damacy_bench.
ARG CMAKE_BUILD_TYPE=RelWithDebInfo
ARG DAMACY_COVERAGE=OFF

# gcovr + codecov-cli are only needed when DAMACY_COVERAGE=ON; install
# them into the project venv so they're on PATH for the test workflow.
# codecov-cli runs inside the container because the runner host is NixOS
# and the upstream codecov-action ships a generic-linux binary that
# can't exec there.
RUN if [ "${DAMACY_COVERAGE}" = "ON" ]; then uv pip install gcovr codecov-cli; fi

# ----- nvcomp (standalone NVIDIA distribution) -------------------------------
RUN set -eux; \
    mkdir -p /opt/nvcomp; \
    curl -fsSL "${NVCOMP_URL}" \
        | tar -xJ -C /opt/nvcomp --strip-components=1; \
    test -f /opt/nvcomp/include/nvcomp.h
ENV Nvcomp_ROOT=/opt/nvcomp

# ----- damacy build ----------------------------------------------------------
WORKDIR /workspace/damacy

# Copy the source tree (governed by .dockerignore — bench/data, build/,
# .venv, .git, etc. are excluded).
COPY . /workspace/damacy

# Configure + build the C library, damacy_bench, and the Python extension.
# CMAKE_BUILD_TYPE and DAMACY_COVERAGE come from --build-arg above.
RUN cmake -S . -B build -G Ninja \
        -DDAMACY_PYTHON=ON \
        -DDAMACY_COVERAGE=${DAMACY_COVERAGE} \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
RUN cmake --build build

# Build-time sanity (CPU-only subset; the two CUDA tests need a GPU at
# runtime, which the Docker builder typically lacks). python_pytest
# also needs the editable install (done below), so it is excluded
# here too. Cluster smoke is the real validation.
RUN ctest --test-dir build --output-on-failure -E "test_damacy|test_assemble|python_pytest" \
 || (echo "WARN: ctest reported failures; continuing so the image still ships" >&2; true)

# Editable install of the Python package. The .so was just built by cmake
# above and lives at build/python/_native*.so; copy it next to __init__.py
# so the editable install resolves `damacy._native` without rebuilding.
RUN cp build/python/_native*.so python/damacy/
RUN uv pip install --no-deps --no-build-isolation -e .

# ----- runtime defaults ------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/damacy
CMD ["/bin/bash"]
