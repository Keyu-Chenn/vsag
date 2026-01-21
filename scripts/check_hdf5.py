#!/usr/bin/env python3
import h5py
import sys
import numpy as np

def print_hdf5_structure(filename):
    """Print detailed structure of HDF5 file"""
    try:
        with h5py.File(filename, 'r') as f:
            print(f"🔍 Analyzing: {filename}")
            print("=" * 60)

            # Print all datasets
            print("📊 DATASETS:")
            for name in f.keys():
                dataset = f[name]
                if isinstance(dataset, h5py.Dataset):
                    print(f"  • {name}")
                    print(f"    Shape: {dataset.shape}")
                    print(f"    Dtype: {dataset.dtype}")
                    print(f"    Size: {dataset.size} elements")
                    if dataset.size > 0:
                        # Show first few values for small datasets
                        if dataset.size <= 10:
                            print(f"    Sample: {dataset[()]}")
                        elif len(dataset.shape) == 1 and dataset.shape[0] > 0:
                            print(f"    First 5: {dataset[:5]}")
                        elif len(dataset.shape) == 2:
                            print(f"    First row: {dataset[0][:5] if dataset.shape[1] > 5 else dataset[0]}")
                    print()

            # Print attributes
            if len(f.attrs) > 0:
                print("🏷️  ATTRIBUTES:")
                for key, value in f.attrs.items():
                    print(f"  • {key}: {value}")

            print("=" * 60)

            # Critical validation for VSAG hybrid format
            print("✅ VSAG HYBRID VALIDATION:")
            required_datasets = ['train', 'test', 'train_sparse', 'test_sparse', 'neighbors', 'distances']
            required_attrs = ['type', 'distance']

            # Check required datasets
            for ds in required_datasets:
                if ds in f:
                    print(f"  ✓ {ds}: OK")
                else:
                    print(f"  ✗ {ds}: MISSING!")

            # Check required attributes
            for attr in required_attrs:
                if attr in f.attrs:
                    print(f"  ✓ {attr}: {f.attrs[attr]}")
                else:
                    print(f"  ✗ {attr}: MISSING!")

            # Special check for sparse vectors
            if 'train_sparse' in f and 'test_sparse' in f:
                train_sparse = f['train_sparse']
                test_sparse = f['test_sparse']

                if train_sparse.dtype == np.uint8 and len(train_sparse.shape) == 1:
                    print(f"  ✓ train_sparse: uint8 1D array (size: {train_sparse.size})")
                else:
                    print(f"  ✗ train_sparse: WRONG FORMAT! dtype={train_sparse.dtype}, shape={train_sparse.shape}")

                if test_sparse.dtype == np.uint8 and len(test_sparse.shape) == 1:
                    print(f"  ✓ test_sparse: uint8 1D array (size: {test_sparse.size})")
                else:
                    print(f"  ✗ test_sparse: WRONG FORMAT! dtype={test_sparse.dtype}, shape={test_sparse.shape}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_hdf5.py <your_file.hdf5>")
        print("Example: python check_hdf5.py data/hybrid_test.hdf5")
        sys.exit(1)

    print_hdf5_structure(sys.argv[1])

    import numpy as np

# 查看各种 dtype 的字节大小
print(f"uint8: {np.uint8().itemsize} bytes")      # 1
print(f"uint32: {np.uint32().itemsize} bytes")    # 4
print(f"float32: {np.float32().itemsize} bytes")  # 4
print(f"int64: {np.int64().itemsize} bytes")      # 8

# 或者直接用 dtype 类型
print(f"np.uint8 itemsize: {np.dtype('uint8').itemsize}")      # 1
print(f"np.uint32 itemsize: {np.dtype('uint32').itemsize}")    # 4
print(f"np.float32 itemsize: {np.dtype('float32').itemsize}")  # 4

