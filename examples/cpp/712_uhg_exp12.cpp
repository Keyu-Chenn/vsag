// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 712_uhg_exp12 -- Dedicated SINDI construction-time benchmark.
//
// Loads ONLY train_sparse + train_labels from an HDF5 file, builds a sparse
// SINDI index in memory, and reports build timing and peak RSS.  No
// serialization, no search, no test/dense/ground-truth loading -- the minimal
// memory footprint for measuring pure SINDI build cost on large datasets.
//
// Output is tee'd to stdout and <out_root>/<dataset>/logs/<dataset>_sindi.log,
// and a machine-parseable summary.tsv is written to <out_root>/<dataset>/.
//
// Default build parameters match 709_uhg_exp9:
//   {"dtype":"sparse","metric_type":"ip","index_param":{"use_reorder":true}}

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <sys/resource.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Tee {
    std::ofstream file_;

public:
    bool
    open(const std::string& path) {
        file_.open(path, std::ios::out | std::ios::trunc);
        return file_.is_open();
    }
    void
    flush() {
        std::cout.flush();
        file_.flush();
    }
    template <typename T>
    Tee& operator<<(const T& val) {
        std::cout << val;
        if (file_.is_open()) file_ << val;
        return *this;
    }
    Tee& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cout << manip;
        if (file_.is_open()) file_ << manip;
        return *this;
    }
} tout;

std::string
GetDatasetName(const std::string& h5_file) {
    std::string filename = h5_file;
    size_t ls = filename.find_last_of('/');
    if (ls != std::string::npos) filename = filename.substr(ls + 1);
    size_t ld = filename.find_last_of('.');
    return (ld != std::string::npos) ? filename.substr(0, ld) : filename;
}

int64_t
PeakRSSBytes() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#ifdef __APPLE__
    return ru.ru_maxrss;
#else
    return ru.ru_maxrss * 1024L;
#endif
}

std::string
HumanSize(int64_t bytes) {
    if (bytes < 0) return "NA";
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double b = static_cast<double>(bytes);
    int i = 0;
    while (b >= 1024.0 && i < 4) {
        b /= 1024.0;
        ++i;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f %s", b, units[i]);
    return std::string(buf);
}

void
MkdirP(const std::string& path) {
    std::string dir;
    size_t pos = 0;
    while (pos < path.size()) {
        size_t next = path.find('/', pos + 1);
        if (next == std::string::npos) next = path.size();
        dir = path.substr(0, next);
        mkdir(dir.c_str(), 0755);
        pos = next;
    }
}

std::vector<vsag::SparseVector>
ParseSparseVectors(const std::vector<uint8_t>& blob) {
    std::vector<vsag::SparseVector> result;
    size_t offset = 0;
    while (offset < blob.size()) {
        if (offset + sizeof(uint32_t) > blob.size()) break;
        uint32_t len;
        std::memcpy(&len, &blob[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);
        vsag::SparseVector vec;
        vec.len_ = len;
        if (len > 0) {
            size_t ids_bytes = len * sizeof(uint32_t);
            if (offset + ids_bytes > blob.size()) break;
            vec.ids_ = new uint32_t[len];
            std::memcpy(vec.ids_, &blob[offset], ids_bytes);
            offset += ids_bytes;
            size_t vals_bytes = len * sizeof(float);
            if (offset + vals_bytes > blob.size()) {
                delete[] vec.ids_;
                break;
            }
            vec.vals_ = new float[len];
            std::memcpy(vec.vals_, &blob[offset], vals_bytes);
            offset += vals_bytes;
        } else {
            vec.ids_ = nullptr;
            vec.vals_ = nullptr;
        }
        result.push_back(vec);
    }
    return result;
}

void
FreeSparseVectors(std::vector<vsag::SparseVector>& sparse_vectors) {
    for (auto& vec : sparse_vectors) {
        delete[] vec.ids_;
        delete[] vec.vals_;
        vec.ids_ = nullptr;
        vec.vals_ = nullptr;
    }
}

struct Params {
    std::string h5_file;
    std::string out_root =
        "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/construction/results/sindi";
    bool use_reorder = true;
};

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nDedicated SINDI construction-time benchmark (in-memory, no serialization).\n"
              << "Loads only train_sparse + train_labels, builds SINDI, reports timing/RSS.\n"
              << "\nOptions:\n"
              << "  --out_root <path>          Output root directory\n"
              << "                             (default: .../construction/results/sindi)\n"
              << "  --no-reorder               Disable SINDI use_reorder (default: enabled)\n"
              << "  --help, -h                 Show this help message\n"
              << "\nWrites:\n"
              << "  <out_root>/<dataset>/logs/<dataset>_sindi.log\n"
              << "  <out_root>/<dataset>/summary.tsv\n"
              << std::endl;
}

Params
ParseCommandLine(int argc, char** argv) {
    Params p;
    if (argc < 2) {
        PrintUsage(argv[0]);
        exit(1);
    }
    std::string arg1 = argv[1];
    if (arg1 == "--help" || arg1 == "-h") {
        PrintUsage(argv[0]);
        exit(0);
    }
    p.h5_file = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "--out_root" && i + 1 < argc)
            p.out_root = argv[++i];
        else if (arg == "--no-reorder")
            p.use_reorder = false;
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }
    return p;
}

}  // namespace

int
main(int argc, char** argv) {
    vsag::init();
    Params params = ParseCommandLine(argc, argv);

    try {
        std::string dataset_name = GetDatasetName(params.h5_file);
        std::string out_dir = params.out_root + "/" + dataset_name;
        std::string log_dir = out_dir + "/logs";
        std::string log_path = log_dir + "/" + dataset_name + "_sindi.log";
        std::string summary_path = out_dir + "/summary.tsv";

        MkdirP(log_dir);
        if (!tout.open(log_path)) {
            std::cerr << "Warning: cannot open log file " << log_path
                      << ", output to stdout only" << std::endl;
        }

        // --- Load only what build needs: train_sparse + train_labels ---
        std::vector<vsag::SparseVector> train_sparse;
        std::vector<int64_t> train_labels;
        int64_t num_train = 0;

        tout << "Loading train sparse data from " << params.h5_file << std::endl;
        {
            H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

            H5::DataSet labels_ds = file.openDataSet("train_labels");
            hsize_t label_dims[1];
            labels_ds.getSpace().getSimpleExtentDims(label_dims);
            num_train = static_cast<int64_t>(label_dims[0]);
            train_labels.resize(num_train);
            labels_ds.read(train_labels.data(), H5::PredType::NATIVE_INT64);

            H5::DataSet sparse_ds = file.openDataSet("train_sparse");
            hsize_t sparse_size = sparse_ds.getSpace().getSimpleExtentNpoints();
            std::vector<uint8_t> sparse_blob(sparse_size);
            sparse_ds.read(sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            train_sparse = ParseSparseVectors(sparse_blob);
            if (static_cast<int64_t>(train_sparse.size()) != num_train) {
                throw std::runtime_error("train_sparse count does not match train_labels count");
            }

            file.close();
        }
        tout << "Train sparse loaded: n=" << num_train
             << ", peak_rss=" << HumanSize(PeakRSSBytes()) << std::endl;

        // --- Build ---
        std::string build_params =
            R"({"dtype":"sparse","metric_type":"ip","index_param":{"use_reorder":)" +
            std::string(params.use_reorder ? "true" : "false") + "}}";

        auto index = vsag::Factory::CreateIndex("sindi", build_params).value();

        tout << "Building SINDI (use_reorder=" << (params.use_reorder ? "true" : "false")
             << ")..." << std::endl;
        auto base = vsag::Dataset::Make();
        base->NumElements(num_train)->Ids(train_labels.data())
            ->SparseVectors(train_sparse.data())->Owner(false);

        auto t0 = std::chrono::high_resolution_clock::now();
        index->Build(base);
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(t1 - t0).count();
        int64_t peak_rss = PeakRSSBytes();

        tout << "SINDI built in " << build_s << "s" << std::endl;
        tout << "Peak RSS after build: " << HumanSize(peak_rss) << std::endl;

        // --- Summary block (machine-parseable) ---
        tout << "\n========== Construction Summary ==========\n"
             << "dataset: " << dataset_name << "\n"
             << "method: sindi\n"
             << "num_train: " << num_train << "\n"
             << "use_reorder: " << (params.use_reorder ? "true" : "false") << "\n"
             << "build_seconds: " << build_s << "\n"
             << "peak_rss_bytes: " << peak_rss << "\n"
             << "==========================================\n";
        tout.flush();

        // --- Write summary.tsv ---
        std::ofstream sum(summary_path, std::ios::out | std::ios::trunc);
        if (sum.is_open()) {
            sum << "dataset\tmethod\tmetric\tvalue\tlog\n";
            sum << dataset_name << "\tsindi\tbuild_seconds\t" << build_s << "\t" << log_path
                << "\n";
            sum << dataset_name << "\tsindi\tpeak_rss_bytes\t" << peak_rss << "\t" << log_path
                << "\n";
            sum.close();
            tout << "Summary written to " << summary_path << std::endl;
        } else {
            tout << "Warning: cannot write summary to " << summary_path << std::endl;
        }
        tout << "Log written to " << log_path << std::endl;

        index.reset();
        FreeSparseVectors(train_sparse);

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
