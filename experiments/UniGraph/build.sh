#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"
VSAG_BUILD_DIR="${REPO_DIR}/build-release"
BASELINE_BUILD_DIR="${REPO_DIR}/build-paper-baselines"
BIN_DIR="${EXPERIMENT_DIR}/bin"

if [[ -z "${JOBS:-}" ]]; then
    JOBS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')"
    if (( JOBS > 8 )); then
        JOBS=8
    fi
fi
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: JOBS must be a positive integer (got %q)\n' "${JOBS}" >&2
    exit 2
fi

for tool in cmake c++ make install; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        printf 'error: required build tool is missing: %s\n' "${tool}" >&2
        exit 1
    fi
done

printf 'Configuring VSAG in %s\n' "${VSAG_BUILD_DIR}"
cmake -S "${REPO_DIR}" -B "${VSAG_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_EXAMPLES=ON \
    -DENABLE_TOOLS=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_PYBINDS=OFF \
    -DENABLE_INTEL_MKL=OFF \
    -DNUM_BUILDING_JOBS="${JOBS}"

# The experiment programs read HDF5 inputs, while the independent baselines
# also link OpenBLAS. Build these external projects before configuring the
# baseline subproject so CMake can locate their installed headers and libraries.
cmake --build "${VSAG_BUILD_DIR}" \
    --target hdf5 openblas \
    --parallel "${JOBS}"

cmake --build "${VSAG_BUILD_DIR}" \
    --target 701_uhg_exp1 703_uhg_exp3 704_uhg_exp4 706_uhg_exp6 \
             708_uhg_exp8 709_uhg_exp9 \
    --parallel "${JOBS}"

printf 'Configuring paper baselines in %s\n' "${BASELINE_BUILD_DIR}"
cmake -S "${REPO_DIR}/examples/cpp/uhg_paper_baselines" \
    -B "${BASELINE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVSAG_BUILD_ROOT="${VSAG_BUILD_DIR}" \
    -DHDF5_ROOT="${VSAG_BUILD_DIR}/hdf5/install" \
    -DOPENBLAS_ROOT="${VSAG_BUILD_DIR}/openblas/install"

cmake --build "${BASELINE_BUILD_DIR}" \
    --target uhg_paper_baselines uhg_extra_baselines \
    --parallel "${JOBS}"

mkdir -p "${BIN_DIR}"
for program in 701_uhg_exp1 703_uhg_exp3 704_uhg_exp4 706_uhg_exp6 \
               708_uhg_exp8 709_uhg_exp9; do
    install -m 0755 \
        "${VSAG_BUILD_DIR}/examples/cpp/${program}" \
        "${BIN_DIR}/${program}"
done
for program in uhg_paper_baselines uhg_extra_baselines; do
    install -m 0755 \
        "${BASELINE_BUILD_DIR}/${program}" \
        "${BIN_DIR}/${program}"
done

printf 'Built all UniGraph experiment programs in %s\n' "${BIN_DIR}"
