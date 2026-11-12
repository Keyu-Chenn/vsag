// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#include "hnsw.h"
#include "ivf.h"
#include "io.h"
#include "public_api.h"
#include "../uhg_mixed_alpha_utils.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sys/resource.h>
#include <unordered_set>

using namespace paper_baselines;
using Clock = std::chrono::steady_clock;

struct Arguments {
    std::map<std::string, std::string> values;
    Arguments(int argc, char** argv) {
        const std::set<std::string> allowed = {"--data", "--method", "--index", "--gt", "--alphas",
            "--alpha", "--k", "--points", "--m", "--ef-construction", "--ef-hybrid", "--seed",
            "--alpha-build", "--construction-sparse-keep", "--sparse-keep",
            "--base-sparse-keep", "--query-sparse-keep", "--tau-dense", "--tau-hybrid", "--clusters",
            "--sketch-dim", "--iterations", "--predictions", "--build-threads",
            "--parallel-warmup"};
        for (int i = 2; i < argc; i += 2) {
            Require(i + 1 < argc, "missing value for " + std::string(argv[i]));
            Require(allowed.count(argv[i]), "unknown option " + std::string(argv[i]));
            Require(values.emplace(argv[i], argv[i+1]).second, "duplicate option " + std::string(argv[i]));
        }
    }
    std::string get(std::string key, std::string fallback = "") const {
        auto it = values.find(key); return it == values.end() ? fallback : it->second;
    }
    size_t integer(std::string key, size_t fallback) const {
        auto value = get(key); if (value.empty()) return fallback;
        Require(std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }), "invalid integer");
        return std::stoull(value);
    }
    float real(std::string key, float fallback) const {
        auto value = get(key); if (value.empty()) return fallback;
        size_t used = 0; float result = std::stof(value, &used);
        Require(used == value.size() && std::isfinite(result), "invalid float"); return result;
    }
};

template <class Search> void Measure(Search search, const Arguments& args, const Vectors& base,
                                    const Vectors& queries) {
    size_t k = args.integer("--k", 100);
    std::vector<size_t> shape;
    auto gt = LoadNpy<int64_t>(args.get("--gt"), "<i8", shape);
    Require(shape.size() == 2 && shape[0] == queries.size() && shape[1] >= k, "GT shape mismatch");
    size_t gt_stride = shape[1];
    Require(args.get("--alpha").empty() || args.get("--alphas").empty(), "choose --alpha or --alphas");
    std::vector<float> weights(queries.size(), args.real("--alpha", 0.5));
    if (!args.get("--alphas").empty()) {
        weights = LoadNpy<float>(args.get("--alphas"), "<f4", shape);
        Require(shape.size() == 1 && weights.size() == queries.size(), "alpha shape mismatch");
    }
    for (float a : weights) Require(std::isfinite(a) && a >= 0 && a <= 1, "invalid query alpha");
    std::string grid = args.get("--points", std::to_string(k));
    std::replace(grid.begin(), grid.end(), ',', ' ');
    std::istringstream points(grid); size_t point; bool measured = false;
    std::vector<std::vector<int64_t>> truth(queries.size());
    for (size_t q = 0; q < queries.size(); ++q) {
        truth[q].assign(gt.begin() + q*gt_stride, gt.begin() + q*gt_stride + k);
        std::sort(truth[q].begin(), truth[q].end());
    }
    while (points >> point) {
        Require(point >= k, "search point below k");
        std::vector<std::vector<Hit>> predictions(queries.size());
        Stats stats;
        size_t hits = 0, short_results = 0;
        auto begin = Clock::now();
        for (Id q = 0; q < queries.size(); ++q) {
            predictions[q] = search(q, weights[q], k, point, stats);
        }
        double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        for (size_t q = 0; q < queries.size(); ++q) {
            if (predictions[q].size() != k) ++short_results;
            std::set<Id> seen;
            std::unordered_set<int64_t> expected(truth[q].begin(), truth[q].end());
            for (auto hit : predictions[q]) {
                Require(hit.second < base.size() && seen.insert(hit.second).second, "invalid/duplicate result");
                Require(std::isfinite(hit.first), "nonfinite result distance");
                hits += expected.count(base.labels[hit.second]);
            }
        }
        Require(short_results == 0, "graph returned fewer than k results");
        double count = queries.size();
        std::cout << std::setprecision(12)
                  << "{\"method\":\"" << args.get("--method") << "\",\"point\":" << point
                  << ",\"k\":" << k << ",\"queries\":" << queries.size()
                  << ",\"recall\":" << hits / (count * k) << ",\"qps\":" << count / seconds
                  << ",\"search_seconds\":" << seconds
                  << ",\"qps_scope\":\"query_to_topk\",\"metric_schema\":2"
                  << ",\"dense_per_query\":" << stats.dense / count
                  << ",\"sparse_per_query\":" << stats.sparse / count
                  << ",\"dense_expansions\":" << stats.dense_expansions / count
                  << ",\"hybrid_expansions\":" << stats.hybrid_expansions / count
                  << ",\"transition_nodes\":" << stats.transition / count
                  << ",\"scanned\":" << stats.scanned / count << ",\"lists\":" << stats.lists / count << "}\n";
        if (!args.get("--predictions").empty()) {
            std::ofstream out(args.get("--predictions") + "." + std::to_string(point) + ".tsv");
            for (size_t q = 0; q < queries.size(); ++q) {
                out << q; for (auto hit : predictions[q]) out << '\t' << base.labels[hit.second]; out << '\n';
            }
            Require(out.good(), "prediction write failed");
        }
        measured = true;
    }
    Require(measured && points.eof(), "invalid search point list");
}

void MeasurePublic(PublicHybridHNSW& index, const Arguments& args, const Vectors& queries) {
    size_t k = args.integer("--k", 100);
    std::vector<size_t> shape;
    auto gt = LoadNpy<int64_t>(args.get("--gt"), "<i8", shape);
    Require(shape.size() == 2 && shape[0] == queries.size() && shape[1] >= k,
            "GT shape mismatch");
    size_t gt_stride = shape[1];
    Require(args.get("--alpha").empty() || args.get("--alphas").empty(),
            "choose --alpha or --alphas");
    std::vector<float> weights(queries.size(), args.real("--alpha", 0.5));
    if (!args.get("--alphas").empty()) {
        weights = LoadNpy<float>(args.get("--alphas"), "<f4", shape);
        Require(shape.size() == 1 && weights.size() == queries.size(), "alpha shape mismatch");
    }
    for (float alpha : weights)
        Require(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid query alpha");

    std::vector<std::vector<int64_t>> truth(queries.size());
    for (size_t q = 0; q < queries.size(); ++q) {
        truth[q].assign(gt.begin() + q * gt_stride, gt.begin() + q * gt_stride + k);
        std::sort(truth[q].begin(), truth[q].end());
    }
    PublicSparseQueries public_sparse(queries.sparse);
    const float tau_dense = args.real("--tau-dense", .8);
    const float tau_hybrid = args.real("--tau-hybrid", 1);
    const float legacy_sparse_keep = args.real("--sparse-keep", 1);
    const float base_sparse_keep = args.real("--base-sparse-keep", legacy_sparse_keep);
    const float query_sparse_keep = args.real("--query-sparse-keep", legacy_sparse_keep);
    std::string grid = args.get("--points", std::to_string(k));
    std::replace(grid.begin(), grid.end(), ',', ' ');
    std::istringstream points(grid);
    size_t point;
    bool measured = false;
    while (points >> point) {
        Require(point >= k, "search point below k");
        std::vector<std::vector<int64_t>> predictions(queries.size());
        auto begin = Clock::now();
#pragma omp parallel for num_threads(1) schedule(dynamic)
        for (int64_t raw_q = 0; raw_q < static_cast<int64_t>(queries.size()); ++raw_q) {
            const size_t q = static_cast<size_t>(raw_q);
            nlohmann::json parameters = {{"method", args.get("--method")},
                                         {"alpha", weights[q]},
                                         {"ef_search", point},
                                         {"tau_dense", tau_dense},
                                         {"tau_hybrid", tau_hybrid}};
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(static_cast<int64_t>(queries.dim))
                ->Float32Vectors(queries.at(static_cast<Id>(q)))
                ->SparseVectors(public_sparse.views.data() + q)
                ->Owner(false);
            auto result = index.KnnSearch(query, static_cast<int64_t>(k), parameters.dump());
            predictions[q].assign(result->GetIds(), result->GetIds() + result->GetDim());
        }
        double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        Stats stats = index.TakeStats();
        uhg_mixed_alpha::WriteReturnedCounts(std::cout, predictions);

        size_t hits = 0;
        size_t short_results = 0;
        for (size_t q = 0; q < queries.size(); ++q) {
            if (predictions[q].size() != k) ++short_results;
            std::set<int64_t> seen;
            std::unordered_set<int64_t> expected(truth[q].begin(), truth[q].end());
            for (int64_t label : predictions[q]) {
                Require(seen.insert(label).second, "duplicate result label");
                hits += expected.count(label);
            }
        }
        Require(short_results == 0, "graph returned fewer than k results");
        const double count = queries.size();
        std::cout << std::setprecision(12)
                  << "{\"method\":\"" << args.get("--method") << "\",\"point\":" << point
                  << ",\"k\":" << k << ",\"queries\":" << queries.size()
                  << ",\"recall\":" << hits / (count * k) << ",\"qps\":" << count / seconds
                  << ",\"search_seconds\":" << seconds
                  << ",\"qps_scope\":\"query_to_topk\",\"metric_schema\":2"
                  << ",\"api_scope\":\"dataset_json_knnsearch\""
                  << ",\"base_sparse_keep\":" << base_sparse_keep
                  << ",\"base_sparse_prune_ratio\":" << 1 - base_sparse_keep
                  << ",\"query_sparse_keep\":" << query_sparse_keep
                  << ",\"query_sparse_prune_ratio\":" << 1 - query_sparse_keep
                  << ",\"exact_rerank\":false"
                  << ",\"dense_per_query\":" << stats.dense / count
                  << ",\"sparse_per_query\":" << stats.sparse / count
                  << ",\"dense_expansions\":" << stats.dense_expansions / count
                  << ",\"hybrid_expansions\":" << stats.hybrid_expansions / count
                  << ",\"transition_nodes\":" << stats.transition / count
                  << ",\"scanned\":" << stats.scanned / count
                  << ",\"lists\":" << stats.lists / count << "}\n";
        if (!args.get("--predictions").empty()) {
            std::ofstream out(args.get("--predictions") + "." + std::to_string(point) + ".tsv");
            for (size_t q = 0; q < queries.size(); ++q) {
                out << q;
                for (int64_t label : predictions[q]) out << '\t' << label;
                out << '\n';
            }
            Require(out.good(), "prediction write failed");
        }
        measured = true;
    }
    Require(measured && points.eof(), "invalid search point list");
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: uhg_paper_baselines build|search --data HDF5 --method naive_hybrid_hnsw|optimized_hybrid_hnsw|unified_ivf_mips --index FILE [--build-threads N] [--construction-sparse-keep R] [--gt NPY --k N --points LIST --alpha A|--alphas NPY] [--base-sparse-keep R --query-sparse-keep R]\n";
            return 2;
        }
        std::string action(argv[1]); Require(action == "build" || action == "search", "expected build or search");
        Arguments args(argc, argv);
        std::string method = args.get("--method"), path = args.get("--index");
        Require(!path.empty(), "missing --index");
        Require(method == "naive_hybrid_hnsw" || method == "optimized_hybrid_hnsw" || method == "unified_ivf_mips", "unknown method");
        size_t build_threads = args.integer("--build-threads", 1);
        Require(build_threads >= 1 && build_threads <= 1024, "build threads must be in [1,1024]");
        omp_set_dynamic(0);
        openblas_set_num_threads(action == "build" && method == "unified_ivf_mips"
                                     ? static_cast<int>(build_threads) : 1);
        if (action == "build") {
            Require(args.get("--sparse-keep").empty() &&
                    args.get("--base-sparse-keep").empty() &&
                    args.get("--query-sparse-keep").empty(),
                    "search sparse retention options do not apply during construction");
        } else {
            Require(args.get("--construction-sparse-keep").empty(),
                    "--construction-sparse-keep is build-only");
        }
        if (method != "optimized_hybrid_hnsw") {
            Require(args.get("--construction-sparse-keep").empty() &&
                    args.get("--base-sparse-keep").empty() &&
                    args.get("--query-sparse-keep").empty(),
                    "independent sparse pruning only applies to optimized_hybrid_hnsw");
        }
        auto base = LoadVectors(args.get("--data"), "train");
        if (action == "build") {
            Require(!std::filesystem::exists(path), "refusing to overwrite an existing index");
            auto report_build = [&](double build_seconds, double serialize_seconds,
                                    uint64_t edges, int layers) {
                rusage usage{};
                getrusage(RUSAGE_SELF, &usage);
                std::cout << std::setprecision(12) << "{\"action\":\"build\",\"method\":\"" << method
                          << "\",\"documents\":" << base.size() << ",\"build_seconds\":" << build_seconds
                          << ",\"serialize_seconds\":" << serialize_seconds
                          << ",\"build_time_scope\":\"loaded_input_to_built_index\",\"metric_schema\":2"
                          << ",\"build_threads\":" << build_threads
                          << ",\"construction_sparse_keep\":"
                          << args.real("--construction-sparse-keep", 1)
                          << ",\"construction_sparse_prune_ratio\":"
                          << 1 - args.real("--construction-sparse-keep", 1)
                          << ",\"edges\":" << edges << ",\"layers\":" << layers
                          << ",\"index_bytes\":" << std::filesystem::file_size(path)
                          << ",\"max_rss_kib\":" << usage.ru_maxrss << "}\n";
            };
            auto build_begin = Clock::now();
            if (method == "unified_ivf_mips") {
                UnifiedIVF::Config c;
                c.clusters = args.integer("--clusters", 100); c.sketch_dim = args.integer("--sketch-dim", 1024);
                c.iterations = args.integer("--iterations", 20); c.seed = args.integer("--seed", 42);
                UnifiedIVF index(base, c);
                index.Build();
                double build_seconds = std::chrono::duration<double>(Clock::now() - build_begin).count();
                auto serialize_begin = Clock::now();
                index.Save(path);
                double serialize_seconds = std::chrono::duration<double>(Clock::now() - serialize_begin).count();
                report_build(build_seconds, serialize_seconds, 0, 0);
            } else {
                HybridHNSW::Config c; c.optimized = method == "optimized_hybrid_hnsw";
                c.m = args.integer("--m", 64); c.ef_construction = args.integer("--ef-construction", 200);
                c.ef_hybrid = args.integer("--ef-hybrid", 200); c.seed = args.integer("--seed", 42);
                c.build_threads = build_threads;
                c.parallel_warmup = args.integer("--parallel-warmup", 10000);
                c.alpha_build = args.real("--alpha-build", .5);
                c.construction_sparse_keep = args.real("--construction-sparse-keep", 1);
                HybridHNSW index(base, c);
                index.Build();
                double build_seconds = std::chrono::duration<double>(Clock::now() - build_begin).count();
                auto serialize_begin = Clock::now();
                index.Save(path);
                double serialize_seconds = std::chrono::duration<double>(Clock::now() - serialize_begin).count();
                report_build(build_seconds, serialize_seconds, index.edges(), index.max_level() + 1);
            }
            return 0;
        }
        auto queries = LoadVectors(args.get("--data"), "test");
        Require(queries.dim == base.dim, "base/query dense dimension mismatch");
        if (method == "unified_ivf_mips") {
            Require(args.get("--sparse-keep").empty() &&
                    args.get("--base-sparse-keep").empty() &&
                    args.get("--query-sparse-keep").empty(),
                    "--sparse-keep only applies to optimized_hybrid_hnsw search");
            auto index = UnifiedIVF::Load(base, path);
            index.PrepareQueries(queries.sparse);
            Measure([&](Id q, float a, size_t k, size_t point, Stats& stats) {
                return index.Search(queries.at(q), queries.sparse[q], a, k, point, stats);
            }, args, base, queries);
        } else {
            auto index = HybridHNSW::Load(base, path);
            Require(index.config().optimized == (method == "optimized_hybrid_hnsw"), "index/method mismatch");
            const bool legacy_sparse_keep = !args.get("--sparse-keep").empty();
            Require(!legacy_sparse_keep ||
                    (args.get("--base-sparse-keep").empty() &&
                     args.get("--query-sparse-keep").empty()),
                    "choose --sparse-keep or the independent base/query retention options");
            const float sparse_keep = args.real("--sparse-keep", 1);
            const float base_sparse_keep = legacy_sparse_keep
                ? sparse_keep : args.real("--base-sparse-keep", 1);
            const float query_sparse_keep = legacy_sparse_keep
                ? sparse_keep : args.real("--query-sparse-keep", 1);
            index.ConfigureSearchSparsePruning(base_sparse_keep, query_sparse_keep);
            PublicHybridHNSW public_index(index, base);
            MeasurePublic(public_index, args, queries);
        }
        return 0;
    } catch (const H5::Exception& error) { std::cerr << "ERROR: " << error.getDetailMsg() << '\n'; }
      catch (const std::exception& error) { std::cerr << "ERROR: " << error.what() << '\n'; }
    return 1;
}
