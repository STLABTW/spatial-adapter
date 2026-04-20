#!/bin/bash
#
# Helper functions for building a conda environment
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"


initialize_conda() {
  local CONDA_BASE=$(conda info --base)
  local CONDA_DIRS=(
    "$CONDA_BASE"
    "/opt/conda"
    "/opt/miniconda"
    "/opt/miniconda2"
    "/opt/miniconda3"
  )

  local IS_CONDA_FOUND=false
  for dir in "${CONDA_DIRS[@]}"; do
    if [ -f "$dir/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="$dir"
      IS_CONDA_FOUND=true
      break
    fi
  done

  if ! $IS_CONDA_FOUND; then
    echo -e "${FG_RED}No Conda environment found matching${FG_RESET}"  >&2
    return ${ERROR_EXITCODE}
  fi

  echo -e "${FG_YELLOW}Intializing conda${FG_RESET}"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}


find_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  initialize_conda
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"

  if [ ${#CONDA_INFO[@]} -eq 0 ]; then
    echo -e "${FG_RED}No Conda environment found matching '${ENV_NAME}'${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  AVAILABLE_ENV_NAME="${CONDA_INFO[0]}"

  if [[ "x${AVAILABLE_ENV_NAME}x" != "x${ENV_NAME}x" ]]; then
    echo -e "${FG_RED}Conda Env '${ENV_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  if [ "x${CONDA_INFO[1]}x" == "x*x" ]; then
    CONDA_ENV_DIR="${CONDA_INFO[2]}"
  else
    CONDA_ENV_DIR="${CONDA_INFO[1]}"
  fi
  # DO NOT REMOVE
  echo "${CONDA_ENV_DIR}"
}


initialize_conda_env() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  conda create -c conda-forge -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  conda activate "${CONDA_ENV}"
  pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  pip install --no-cache-dir virtualenv
  conda deactivate
}

retry_to_find_conda_env_path() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  if [ "x${PYTHON_VERSION}x" == "xx" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
  fi

  initialize_conda_env "${CONDA_ENV}" "${PYTHON_VERSION}"

  find_conda_env_path "${CONDA_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Unknown exception occurs from the side of Conda infra${FG_RESET}"
  fi

}

install_python_package() {
  local TARGET_PROJECT_DIR=$1

  pushd "${TARGET_PROJECT_DIR}" || exit

  # Editable install: source tree == import location, so the .so compiled by
  # `make build-cpp` into spatial_adapter/cpp_extensions/ is picked up directly.
  # (Previously this did `poetry build && pip install dist/*.tar.gz`, which
  # produced a non-editable site-packages copy and broke C++ extension loading.)
  echo -e "${FG_YELLOW}Installing python package (editable mode)${FG_RESET}"

  # Refresh poetry.lock for reproducibility if Poetry is configured/available.
  if grep -q "\[tool\.poetry\]" pyproject.toml && command -v poetry >/dev/null 2>&1; then
    echo -e "${FG_YELLOW}Refreshing poetry.lock${FG_RESET}"
    poetry lock || echo -e "${FG_YELLOW}poetry lock failed; continuing with pip${FG_RESET}"
  fi

  pip install -e .

  popd || exit
}


activate_conda_environment() {
  local CONDA_ENV=$1
  initialize_conda
  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Activating Conda Env: ${CONDA_ENV}${FG_RESET}"
    conda activate ${CONDA_ENV}
  else
    echo -e "${FG_RED}Activation Failed. Conda is not installed.${FG_RESET}"
  fi
}

update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local CONDA_ENV=$2

  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Updating Conda environment - ${CONDA_ENV}${FG_RESET}"
    bash "${CONDA_DIR}/build_conda_env.sh" --conda_env ${CONDA_ENV}
  else
    echo -e "${FG_RED}Update Failed. Conda is not installed.${FG_RESET}"
  fi
}

install_mac_packages() {
    echo "Installing system packages for macOS..."
    brew update
    # Only install X11 libraries (if needed)
    brew install libxrender libsm libxext || {
        echo -e "${FG_RED}Failed to install X11 libraries via Homebrew.${FG_RESET}"
        exit 1
    }
}

install_linux_packages() {
    echo -e "${FG_YELLOW}Installing system packages for Linux...${FG_RESET}"

    if [[ -f /etc/debian_version ]]; then
        sudo apt update
        sudo apt install -y libxrender1 libsm6 libxext6 || { \
            echo -e "${FG_RED}Failed to install X11 libraries. Ensure you have sudo privileges.${FG_RESET}"
            exit 1
        }
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y armadillo-devel pybind11-devel cmake gcc gcc-c++ \
                            libXrender libSM libXext || { \
            echo -e "${FG_RED}Failed to install system packages. Ensure you have sudo privileges.${FG_RESET}"
            exit 1
        }
    else
        echo -e "${FG_RED}Unsupported Linux distribution. Please install X11 libraries manually.${FG_RESET}"
    fi
}

# Function to install dependencies using Conda
install_conda_packages() {
    initialize_conda
    echo -e "${FG_YELLOW}🚀 Installing dependencies via Conda (with acceleration)...${FG_RESET}"

    if ! command -v conda &> /dev/null; then
        echo -e "${FG_RED}❌ Conda is not installed. Please install Miniconda or Anaconda first.${FG_RESET}"
        exit 1
    fi

    # Ensure conda-forge has strict priority
    conda config --add channels conda-forge
    conda config --set channel_priority strict

    # Install mamba if available
    if ! command -v mamba &> /dev/null; then
        echo -e "${FG_YELLOW}Installing mamba for faster dependency resolution...${FG_RESET}"
        conda install -n base -c conda-forge mamba -y || {
            echo -e "${FG_RED}⚠️ Failed to install mamba. Falling back to conda...${FG_RESET}"
        }
    fi

    INSTALL_CMD="conda install"
    if command -v mamba &> /dev/null; then
        INSTALL_CMD="mamba install"
    fi

    # Packages to install
    DEPS=(
        python=${DEFAULT_PYTHON_VERSION}
        cmake
        pybind11
        armadillo
        openblas
        eigen
        "libblas=*=*openblas"
        "liblapack=*=*openblas"
        virtualenv
    )

    # Only add gfortran_linux-64 and libgfortran5 on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        DEPS+=(gfortran_linux-64 libgfortran5)
    fi

    echo -e "${FG_YELLOW}🔧 Installing: ${DEPS[*]}${FG_RESET}"
    $INSTALL_CMD -y -c conda-forge "${DEPS[@]}" || {
        echo -e "${FG_RED}❌ Failed to install packages with ${INSTALL_CMD}.${FG_RESET}"
        exit 1
    }

    # Environment variable setup
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export OpenBLAS_ROOT="$CONDA_PREFIX"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    echo -e "${FG_GREEN}✅ Conda environment and dependencies installed successfully using ${INSTALL_CMD}.${FG_RESET}"

    # Pre-install Cython and NumPy for Pystan build compatibility
    echo -e "${FG_YELLOW}🔧 Pre-installing Cython and NumPy for build compatibility...${FG_RESET}"
    $INSTALL_CMD -y -c conda-forge cython numpy || {
        echo -e "${FG_RED}❌ Failed to install Cython and NumPy with ${INSTALL_CMD}.${FG_RESET}"
        exit 1
    }
    echo -e "${FG_GREEN}✅ Cython and NumPy pre-installed successfully.${FG_RESET}"

    # Install PyTorch + torchvision with CUDA 12.8 support (required for Blackwell / sm_120 GPUs)
    echo -e "${FG_YELLOW}🔧 Installing PyTorch + torchvision with CUDA 12.8 support...${FG_RESET}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || {
        echo -e "${FG_RED}⚠️ Failed to install PyTorch with CUDA 12.8. Falling back to default...${FG_RESET}"
        pip install torch torchvision
    }
    echo -e "${FG_GREEN}✅ PyTorch + torchvision installed.${FG_RESET}"
}


build_cpp() {
    echo -e "${FG_YELLOW}🔹 Building C++ extensions in the env $CONDA_PREFIX ...${FG_RESET}"

    # Ensure OpenBLAS is correctly exposed to CMake and linker
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export OpenBLAS_ROOT="$CONDA_PREFIX"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    # Set OpenBLAS library path based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OPENBLAS_LIB="$CONDA_PREFIX/lib/libopenblas.dylib"
    else
        OPENBLAS_LIB="$CONDA_PREFIX/lib/libopenblas.so"
    fi

    # Navigate to the C++ extension directory
    cd ${CONDA_DIR}/../../spatial_adapter/cpp_extensions || {
        echo -e "${FG_RED}❌ Error: C++ extensions directory not found.${FG_RESET}"
        exit 1
    }

    # Remove old build files
    echo "🧹 Cleaning old build files..."
    rm -rf build CMakeCache.txt CMakeFiles spatial_utils.so cmake_install.cmake

    # Create a new build directory
    mkdir -p build && cd build

    # Run CMake with Conda's Fortran libraries
    cmake \
      -DCMAKE_INCLUDE_PATH=$CONDA_PREFIX/include \
      -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib \
      -DBLAS_LIBRARIES=$OPENBLAS_LIB \
      -DLAPACK_LIBRARIES=$OPENBLAS_LIB \
      -DCMAKE_BUILD_TYPE=Release ..

    # Compile using all available cores
    echo "⚙️  Compiling C++ extensions..."
    make -j$(nproc)

    # Verify shared library dependencies
    echo "🔍 Verifying dependencies of compiled .so file..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        otool -L ../spatial_utils.so | grep -E "blas|lapack|armadillo"
    else
        ldd ../spatial_utils.so | grep -E "blas|lapack|armadillo"
    fi

    # Return to the original working directory
    cd - || exit

    # Install the package using Poetry and copy C++ extensions
    echo "📦 Installing package and copying C++ extensions..."
    poetry install

    # Get the package directory and copy C++ extensions if needed
    PACKAGE_DIR=$(python -c "import spatial_adapter; print(spatial_adapter.__file__)" | sed 's/__init__.py//')
    echo "Package directory: $PACKAGE_DIR"

    # Check if it's a development install (editable mode) by comparing realpaths
    DEV_INSTALL_DIR="$(realpath "${PACKAGE_BASE_PATH}/spatial_adapter" 2>/dev/null)"
    INSTALLED_DIR="$(realpath "${PACKAGE_DIR%/}" 2>/dev/null)"
    if [[ -n "${DEV_INSTALL_DIR}" && "${INSTALLED_DIR}" == "${DEV_INSTALL_DIR}" ]]; then
        echo -e "${FG_GREEN}✅ Development install detected - C++ extensions already in place${FG_RESET}"
    else
        echo "Installing C++ extensions to: $PACKAGE_DIR"
        cp spatial_adapter/cpp_extensions/spatial_utils.so "$PACKAGE_DIR/cpp_extensions/"
        echo -e "${FG_GREEN}✅ C++ extensions installed successfully${FG_RESET}"
    fi

    # Verify the C++ extensions are accessible
    echo "🔍 Verifying C++ extensions are accessible..."
    python -c "from spatial_adapter.cpp_extensions import spatial_utils; print('✅ C++ extensions imported successfully')"

    echo -e "${FG_GREEN}🎉 Build and installation complete!${FG_RESET}"
}

fix_lapack_blas_linking() {
    echo -e "${FG_YELLOW}Verifying BLAS/LAPACK linking...${FG_RESET}"

    if ! ldconfig -p | grep -q "liblapack.so"; then
        echo -e "${FG_RED}❌ LAPACK is missing! Installing...${FG_RESET}"
        sudo apt install -y liblapack-dev libopenblas-dev || {
            echo -e "${FG_RED}Failed to install LAPACK. Please check manually.${FG_RESET}"
            exit 1
        }
    fi

    if ! ldconfig -p | grep -q "libgfortran.so.3"; then
        if [ ! -e /lib/x86_64-linux-gnu/libgfortran.so.3 ]; then
            echo -e "${FG_YELLOW}Creating a symlink for libgfortran.so.3 -> libgfortran.so.5...${FG_RESET}"
            sudo ln -s /lib/x86_64-linux-gnu/libgfortran.so.5 /lib/x86_64-linux-gnu/libgfortran.so.3
        else
            echo -e "${FG_GREEN}✔ libgfortran.so.3 already exists.${FG_RESET}"
        fi
    else
        echo -e "${FG_GREEN}✔ libgfortran.so.3 is already registered by ldconfig.${FG_RESET}"
    fi

    echo -e "${FG_GREEN}✅ BLAS, LAPACK, and Fortran libraries are properly linked.${FG_RESET}"
}

fix_poetry_dependencies() {
    echo -e "${FG_YELLOW}🔧 Fixing Poetry dependencies...${FG_RESET}"

    # Install Poetry if not present
    if ! command -v poetry &> /dev/null; then
        echo -e "${FG_YELLOW}Installing Poetry...${FG_RESET}"
        pip install --no-cache-dir "poetry==${POETRY_VERSION}"
    fi

    # Install virtualenv if not present
    if ! python -c "import virtualenv" 2>/dev/null; then
        echo -e "${FG_YELLOW}Installing virtualenv...${FG_RESET}"
        pip install --no-cache-dir virtualenv
    fi

    # Check Poetry version and install compatible poetry-plugin-export
    POETRY_VERSION_CURRENT=$(poetry --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo -e "${FG_YELLOW}Poetry version: ${POETRY_VERSION_CURRENT}${FG_RESET}"

    if [[ "${POETRY_VERSION_CURRENT}" == 1.* ]]; then
        # For Poetry 1.x, install compatible version
        if ! poetry show poetry-plugin-export 2>/dev/null; then
            echo -e "${FG_YELLOW}Installing poetry-plugin-export for Poetry 1.x...${FG_RESET}"
            poetry self add "poetry-plugin-export<1.9.0"
        fi
    else
        # For Poetry 2.x, install latest version
        if ! poetry show poetry-plugin-export 2>/dev/null; then
            echo -e "${FG_YELLOW}Installing poetry-plugin-export for Poetry 2.x...${FG_RESET}"
            poetry self add poetry-plugin-export
        fi
    fi

    echo -e "${FG_GREEN}✅ Poetry dependencies fixed.${FG_RESET}"
}
