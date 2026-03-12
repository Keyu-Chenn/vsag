# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Debug build (with tests)
make debug

# Release build
make release

# Run unit tests
make test

# Run tests with AddressSanitizer
make test_asan

# Run tests with ThreadSanitizer
make test_tsan

# Run specific test case
make test CASE=TestCaseName

# Code formatting
make fmt

# Code coverage
make cov

# Clean build directory
make clean

# Clean release build directory
make clean-release
```

### CMake Configuration Variables

Key environment variables to control build:
- `VSAG_ENABLE_TESTS=ON/OFF` - Enable/disable unit tests
- `VSAG_ENABLE_PYBINDS=ON/OFF` - Enable/disable Python bindings
- `VSAG_ENABLE_TOOLS=ON/OFF` - Enable/disable tools
- `VSAG_ENABLE_EXAMPLES=ON/OFF` - Enable/disable examples
- `VSAG_ENABLE_INTEL_MKL=ON/OFF` - Enable/disable Intel MKL
- `COMPILE_JOBS=N` - Number of parallel build jobs (default: 6)

### Python Wheel Build

```bash
# Build for specific Python version
make pyvsag PY_VERSION=3.10

# Build for all supported versions
make pyvsag-all
```

## Architecture Overview

VSAG is a vector indexing library for similarity search, written in C++ with Python bindings (pyvsag).

### Core Components

**Public API** (`include/vsag/`):
- `index.h` - Main Index interface with KnnSearch, RangeSearch, Build, Serialize/Deserialize
- `factory.h` - Index factory for creating index instances
- `dataset.h` - Dataset structure for vectors and metadata
- `search_param.h` - Search parameter configuration

**Index Types** (`src/algorithm/`):
- **HNSW** - Graph-based index for dense vectors (supports incremental Add/Remove)
- **HGRAPH** - Enhanced graph index with datacell abstraction
- **IVF** - Inverted File index with partition strategies (GNO-IMI, Nearest)
- **Diskann** - SSD-based index for large-scale vectors
- **Sparse Index** - Index for sparse vectors
- **SINDI** - Sparse index for maximum inner product search
- **Pyramid** - High-dimensional index
- **BruteForce** - Flat index for exact search
- **Hybrid Index** - Combines multiple index types

**DataCell Layer** (`src/datacell/`):
- Abstraction for storage components (flatten, graph, bucket, extra_info)
- Supports pluggable quantization (PQ, RaBitQ)
- Provides interface-based design for index composition

**Quantization** (`src/quantization/`):
- Scalar quantization (INT8)
- Product Quantization (PQ)
- RaBitQ quantization

**Storage** (`src/storage/`):
- Persistent storage backends (KV, streaming)
- Used by Diskann and other disk-based indexes

**Impl Layer** (`src/impl/`):
- Allocator - Memory allocation strategies
- BitSet - Filtering support
- ConjugateGraph - Graph enhancement for learned indexes
- Filter - Pre-filtering implementations
- Heap - Result collection
- Searcher - Search execution engine
- ThreadPool - Parallel execution

### Index Factory Pattern

```cpp
// Create index via factory
auto index = vsag::Factory::CreateIndex("hnsw", build_parameters_json).value();
```

Build parameters are JSON strings specifying:
- `dtype`: float32, int8, sparse
- `metric_type`: l2, ip, cosine
- `dim`: vector dimension
- Algorithm-specific params (e.g., `hnsw.max_degree`, `hnsw.ef_construction`)

### Testing Structure

**Unit Tests** (`tests/`):
- `test_hnsw.cpp`, `test_hgraph.cpp`, `test_ivf.cpp` - Per-index tests
- `test_factory.cpp` - Factory and parameter validation
- `test_index.cpp` - Common index interface tests
- `fixtures/` - Test utilities and dataset pools

Test executables:
- `build/tests/unittests` - Unit tests
- `build/tests/functests` - Functional tests
- `build/mockimpl/tests_mockimpl` - Mock implementation tests

Run specific test: `./build/tests/unittests -d yes "TestCaseName"`

### Examples

Located in `examples/cpp/`:
- `101_index_hnsw.cpp` - Basic HNSW usage (recommended starting point)
- `103_index_hgraph.cpp` - HGRAPH index usage
- `106_index_ivf.cpp` - IVF index with partition strategies
- `109_index_sindi.cpp` - Sparse vector search
- `110_index_hybrid.cpp` - Hybrid index composition
- `3xx_*.cpp` - Feature examples (filter, update, remove, clone, etc.)

## Coding Style

- **Style Guide**: Google C++ Style Guide with modifications
- **Indentation**: 4 spaces
- **Line Length**: 100 characters
- **File Extension**: `.cpp` (not `.cc`)
- **Naming Conventions** (enforced by clang-tidy):
  - Classes: CamelCase
  - Namespaces: lower_case
  - Functions (public): CamelCase
  - Functions (private): lower_case
  - Variables: lower_case
  - Private members: suffix with `_`
  - Constants/Macros: UPPER_CASE

Format code with: `make fmt`

## Key Design Patterns

1. **Interface + Implementation**: Public interface in `include/`, implementation in `src/` with private suffix
2. **Parameter Objects**: Each index has `<index>_parameter.h` for configuration
3. **Expected/Result Types**: Uses `tl::expected<T, Error>` for error handling
4. **Factory Pattern**: All indexes created via `vsag::Factory::CreateIndex()`
5. **DataCell Composition**: Indexes compose datacell components for storage

## Development Notes

- Uses Git submodules for dependencies (managed by CMake ExternalProject)
- Docker development image available: `vsaglib/vsag:ubuntu`
- DCO sign-off required for commits: `git commit -s -m "message"`
- Code coverage requirement: >= 90% for new contributions
