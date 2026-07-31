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

// 711_uhg_exp11 -- Dedicated HNSW construction-time benchmark.
//
// Loads ONLY train_dense + train_labels from an HDF5 file, builds a dense HNSW
// index in memory, and reports build timing and peak RSS.  No serialization,
// no search, no test/sparse/ground-truth loading -- the minimal memory
// footprint for measuring pure HNSW build cost on large datasets.
//
// Output is tee'd to stdout and <out_root>/<dataset>/logs/<dataset>_hnsw.log,
// and a machine-parseable summary.tsv is written to <out_root>/<dataset>/.
//
// Default build parameters match 708_uhg_exp8:
//   {"dtype":"float32","metric_type":"ip","dim":D,
//    "hnsw":{"max_degree":64,"ef_construction":200}}

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
    // Create each component; ignore existing.
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

struct Params {
    std::string h5_file;
    std::string out_root =
        "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/construction/results/hnsw";
    int max_degree = 64;
    int ef_construction = 200;
    std::string metric_type = "ip";
};

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nDedicated HNSW construction-time benchmark (in-memory, no serialization).\n"
              << "Loads only train_dense + train_labels, builds HNSW, reports timing/RSS.\n"
              << "\nOptions:\n"
              << "  --out_root <path>          Output root directory\n"
              << "                             (default: .../construction/results/hnsw)\n"
              << "  --max_degree <int>         HNSW max_degree (default: 64)\n"
              << "  --ef_construction <int>    HNSW ef_construction (default: 200)\n"
              << "  --metric_type <str>        Metric: ip or l2 (default: ip)\n"
              << "  --help, -h                 Show this help message\n"
              << "\nWrites:\n"
              << "  <out_root>/<dataset>/logs/<dataset>_hnsw.log\n"
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
        else if (arg == "--max_degree" && i + 1 < argc)
            p.max_degree = std::atoi(argv[++i]);
        else if (arg == "--ef_construction" && i + 1 < argc)
            p.ef_construction = std::atoi(argv[++i]);
        else if (arg == "--metric_type" && i + 1 < argc)
            p.metric_type = argv[++i];
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }
    if (p.max_degree <= 0) {
        std::cerr << "Error: max_degree must be positive\n";
        exit(1);
    }
    if (p.ef_construction <= 0) {
        std::cerr << "Error: ef_construction must be positive\n";
        exit(1);
    }
    if (p.metric_type != "ip" && p.metric_type != "l2") {
        std::cerr << "Error: metric_type must be ip or l2\n";
        exit(1);
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
        std::string log_path = log_dir + "/" + dataset_name + "_hnsw.log";
        std::string summary_path = out_dir + "/summary.tsv";

        MkdirP(log_dir);
        if (!tout.open(log_path)) {
            std::cerr << "Warning: cannot open log file " << log_path
                      << ", output to stdout only" << std::endl;
        }

        // --- Load only what build needs: train_dense + train_labels ---
        std::vector<float> train_dense;
        std::vector<int64_t> train_labels;
        int64_t num_train = 0;
        int64_t dense_dim = 0;

        tout << "Loading train data from " << params.h5_file << std::endl;
        {
            H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

            H5::DataSet train_ds = file.openDataSet("train");
            H5::DataSpace train_space = train_ds.getSpace();
            hsize_t train_dims[2];
            train_space.getSimpleExtentDims(train_dims);
            num_train = static_cast<int64_t>(train_dims[0]);
            dense_dim = static_cast<int64_t>(train_dims[1]);
            train_dense.resize(num_train * dense_dim);
            train_ds.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

            H5::DataSet labels_ds = file.openDataSet("train_labels");
            hsize_t label_dims[1];
            labels_ds.getSpace().getSimpleExtentDims(label_dims);
            if (static_cast<int64_t>(label_dims[0]) != num_train) {
                throw std::runtime_error("train_labels count does not match train count");
            }
            train_labels.resize(num_train);
            labels_ds.read(train_labels.data(), H5::PredType::NATIVE_INT64);

            file.close();
        }
        tout << "Train loaded: n=" << num_train << ", dim=" << dense_dim
             << ", raw_size=" << HumanSize(num_train * dense_dim * sizeof(float))
             << ", peak_rss=" << HumanSize(PeakRSSBytes()) << std::endl;

        // --- Build ---
        std::string build_params = R"({"dtype":"float32","metric_type":")" +
            params.metric_type + R"(","dim":)" + std::to_string(dense_dim) +
            R"(,"hnsw":{"max_degree":)" + std::to_string(params.max_degree) +
            R"(,"ef_construction":)" + std::to_string(params.ef_construction) + "}}";

        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto index = engine.CreateIndex("hnsw", build_params).value();

        tout << "Building HNSW (max_degree=" << params.max_degree
             << ", ef_construction=" << params.ef_construction
             << ", metric=" << params.metric_type << ")..." << std::endl;
        auto base = vsag::Dataset::Make();
        base->NumElements(num_train)->Dim(dense_dim)->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())->Owner(false);

        auto t0 = std::chrono::high_resolution_clock::now();
        index->Build(base);
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(t1 - t0).count();
        int64_t peak_rss = PeakRSSBytes();

        tout << "HNSW built in " << build_s << "s" << std::endl;
        tout << "Peak RSS after build: " << HumanSize(peak_rss) << std::endl;

        // --- Summary block (machine-parseable) ---
        tout << "\n========== Construction Summary ==========\n"
             << "dataset: " << dataset_name << "\n"
             << "method: hnsw\n"
             << "num_train: " << num_train << "\n"
             << "dim: " << dense_dim << "\n"
             << "max_degree: " << params.max_degree << "\n"
             << "ef_construction: " << params.ef_construction << "\n"
             << "metric_type: " << params.metric_type << "\n"
             << "build_seconds: " << build_s << "\n"
             << "peak_rss_bytes: " << peak_rss << "\n"
             << "==========================================\n";
        tout.flush();

        // --- Write summary.tsv ---
        std::ofstream sum(summary_path, std::ios::out | std::ios::trunc);
        if (sum.is_open()) {
            sum << "dataset\tmethod\tmetric\tvalue\tlog\n";
            sum << dataset_name << "\thnsw\tbuild_seconds\t" << build_s << "\t" << log_path << "\n";
            sum << dataset_name << "\thnsw\tpeak_rss_bytes\t" << peak_rss << "\t" << log_path
                << "\n";
            sum.close();
            tout << "Summary written to " << summary_path << std::endl;
        } else {
            tout << "Warning: cannot write summary to " << summary_path << std::endl;
        }
        tout << "Log written to " << log_path << std::endl;

        index.reset();
        engine.Shutdown();

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
