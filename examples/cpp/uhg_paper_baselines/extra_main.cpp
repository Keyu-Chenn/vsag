// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#include "io.h"
#include "ladr.h"
#include "sinnamon.h"
#include "public_api.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sys/resource.h>
#include <vsag/vsag.h>

using namespace paper_baselines;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;

struct ExtraArguments {
    std::map<std::string, std::string> values;
    ExtraArguments(int argc, char** argv) {
        const std::set<std::string> allowed = {
            "--data", "--method", "--index", "--gt", "--alphas", "--alpha", "--k", "--points",
            "--clusters", "--sketch-dim", "--iterations", "--seed", "--build-threads",
            "--hnsw-index", "--sindi-index", "--seeds", "--neighbors", "--query-prune-ratio",
            "--num-queries", "--predictions"};
        for (int i = 2; i < argc; i += 2) {
            Require(i + 1 < argc, "option has no value");
            Require(allowed.count(argv[i]), "unknown option " + std::string(argv[i]));
            Require(values.emplace(argv[i], argv[i+1]).second, "duplicate option");
        }
    }
    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto i = values.find(key); return i == values.end() ? fallback : i->second;
    }
    size_t integer(const std::string& key, size_t fallback) const {
        auto value = get(key); if (value.empty()) return fallback;
        Require(std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
                "invalid integer for " + key);
        return std::stoull(value);
    }
    float real(const std::string& key, float fallback) const {
        auto value = get(key); if (value.empty()) return fallback;
        size_t used = 0; float number = std::stof(value, &used);
        Require(used == value.size() && std::isfinite(number), "invalid float for " + key);
        return number;
    }
};

class LexicalSeeds {
 public:
    LexicalSeeds(const Vectors& base, const Vectors& queries, const std::string& path, float pruning)
        : base_(base), queries_(queries.sparse) {
        Require(pruning >= 0 && pruning <= .9F, "invalid lexical query pruning");
        auto made = vsag::Factory::CreateIndex("sindi",
            R"({"dtype":"sparse","metric_type":"ip","index_param":{"use_reorder":true}})");
        Require(made.has_value(), "could not create LADR seed index");
        index_ = made.value();
        std::ifstream input(path, std::ios::binary);
        Require(input.good(), "cannot open LADR SINDI index");
        auto loaded = index_->Deserialize(input);
        Require(loaded.has_value(), "cannot deserialize LADR SINDI index");
        Require(index_->GetNumElements() == static_cast<int64_t>(base.size()), "seed corpus mismatch");
        parameters_ = nlohmann::json({{"sindi", {{"query_prune_ratio", pruning},
                                                 {"term_prune_ratio", 0.0}}}}).dump();
        identity_ = true;
        for (Id id = 0; id < base.size(); ++id) identity_ &= base.labels[id] == id;
        if (!identity_) {
            rows_.reserve(base.size());
            for (Id id = 0; id < base.size(); ++id) rows_.emplace(base.labels[id], id);
        }
    }
    std::vector<Id> Search(Id query_id, size_t count) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->SparseVectors(queries_.views.data() + query_id)->Owner(false);
        auto result = index_->KnnSearch(query, static_cast<int64_t>(count), parameters_);
        Require(result.has_value(), "LADR lexical retrieval failed");
        std::vector<Id> ids;
        ids.reserve(result.value()->GetDim());
        for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
            const int64_t label = result.value()->GetIds()[j];
            if (identity_) {
                Require(label >= 0 && static_cast<uint64_t>(label) < base_.size(), "unknown seed label");
                ids.push_back(static_cast<Id>(label));
            } else {
                auto row = rows_.find(label);
                Require(row != rows_.end(), "unknown seed label");
                ids.push_back(row->second);
            }
        }
        return ids;
    }
 private:
    const Vectors& base_;
    PublicSparseQueries queries_;
    std::shared_ptr<vsag::Index> index_;
    std::string parameters_;
    bool identity_ = false;
    std::unordered_map<int64_t, Id> rows_;
};

template<class Search>
void MeasureExtra(Search search, const ExtraArguments& args, const Vectors& base, const Vectors& queries) {
    const size_t k = args.integer("--k", 100);
    const size_t query_count = std::min(queries.size(), args.integer("--num-queries", queries.size()));
    Require(k > 0 && k <= base.size() && query_count > 0, "invalid query count/k");
    std::vector<size_t> shape;
    auto gt = LoadNpy<int64_t>(args.get("--gt"), "<i8", shape);
    Require(shape.size() == 2 && shape[0] >= query_count && shape[1] >= k, "GT shape mismatch");
    const size_t stride = shape[1];
    Require(args.get("--alpha").empty() || args.get("--alphas").empty(), "choose alpha or alphas");
    std::vector<float> alpha(queries.size(), args.real("--alpha", .5F));
    if (!args.get("--alphas").empty()) {
        alpha = LoadNpy<float>(args.get("--alphas"), "<f4", shape);
        Require(shape.size() == 1 && alpha.size() >= query_count, "alpha count mismatch");
    }
    for (float value : alpha) Require(value >= 0 && value <= 1 && std::isfinite(value), "invalid alpha");
    std::vector<std::set<int64_t>> truth(query_count);
    for (size_t q = 0; q < query_count; ++q)
        truth[q].insert(gt.begin() + q * stride, gt.begin() + q * stride + k);
    std::string grid = args.get("--points", std::to_string(k));
    std::replace(grid.begin(), grid.end(), ',', ' ');
    std::istringstream points(grid);
    size_t point = 0, pass = 0;
    while (points >> point) {
        Require(point > 0, "search point must be positive");
        std::vector<std::vector<Hit>> predictions(query_count);
        Stats stats;
        const auto start = Clock::now();
        for (Id q = 0; q < query_count; ++q)
            predictions[q] = search(q, alpha[q], k, point, stats);
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        size_t hits = 0, short_queries = 0, missing = 0, returned = 0;
        std::vector<size_t> returned_counts;
        returned_counts.reserve(query_count);
        size_t smallest = k, largest = 0;
        for (size_t q = 0; q < query_count; ++q) {
            Require(predictions[q].size() <= k, "too many results");
            const size_t n = predictions[q].size();
            returned_counts.push_back(n);
            short_queries += n < k; missing += k - n; returned += n;
            smallest = std::min(smallest, n); largest = std::max(largest, n);
            std::set<Id> seen;
            for (auto [distance, id] : predictions[q]) {
                Require(id < base.size() && seen.insert(id).second, "invalid/duplicate result");
                Require(std::isfinite(distance), "nonfinite result");
                hits += truth[q].count(base.labels[id]);
            }
        }
        const double count = static_cast<double>(query_count);
        Json metric = {{"method", args.get("--method")}, {"point", point}, {"pass", pass},
            {"k", k}, {"queries", query_count}, {"recall", hits / (count * k)},
            {"qps", count / seconds}, {"search_seconds", seconds},
            {"qps_scope", "query_to_topk"}, {"metric_schema", 2},
            {"dense_per_query", stats.dense / count}, {"sparse_per_query", stats.sparse / count},
            {"dense_expansions", stats.dense_expansions / count},
            {"scanned", stats.scanned / count}, {"lists", stats.lists / count},
            {"short_query_count", short_queries}, {"total_missing", missing},
            {"duplicate_query_count", 0}, {"returned_min", smallest}, {"returned_max", largest},
            {"returned_mean", returned / count}};
        metric["returned_counts"] = returned_counts;
        std::cout << metric.dump() << std::endl;
        if (!args.get("--predictions").empty()) {
            std::ofstream output(args.get("--predictions") + "." + std::to_string(pass) + ".tsv");
            output << "query_index\tresult_count\tresult_ids\tresult_distance_bits\n";
            for (size_t q = 0; q < query_count; ++q) {
                output << q << '\t' << predictions[q].size() << '\t';
                for (size_t j = 0; j < predictions[q].size(); ++j)
                    output << (j ? "," : "") << base.labels[predictions[q][j].second];
                output << '\t';
                for (size_t j = 0; j < predictions[q].size(); ++j) {
                    uint32_t bits; std::memcpy(&bits, &predictions[q][j].first, sizeof(bits));
                    output << (j ? "," : "") << bits;
                }
                output << '\n';
            }
            Require(output.good(), "prediction write failed");
        }
        ++pass;
    }
    Require(pass > 0 && points.eof(), "invalid search point list");
}

int main(int argc, char** argv) {
    try {
        Require(argc >= 2, "usage: uhg_extra_baselines build|search --method METHOD --data HDF5 --index FILE ...");
        ExtraArguments args(argc, argv);
        const std::string action = argv[1], method = args.get("--method"), path = args.get("--index");
        Require(action == "build" || action == "search", "expected build or search");
        Require(!path.empty(), "missing index path");
        const bool sinnamon = method == "sinnamon";
        const bool adaptive = method == "ladr_adaptive" || method == "ladr_adaptive_hybrid";
        const bool proactive = method == "ladr_proactive" || method == "ladr_proactive_hybrid";
        Require(sinnamon || adaptive || proactive, "unknown method");
        const bool hybrid_rerank = method.find("_hybrid") != std::string::npos;
        const size_t threads = args.integer("--build-threads", 32);
        Require(threads > 0 && threads <= 96, "invalid construction thread budget");
        omp_set_dynamic(0);
        omp_set_num_threads(action == "build" ? threads : 1);
        openblas_set_num_threads(action == "build" && sinnamon ? threads : 1);
        auto base = LoadVectors(args.get("--data"), "train");
        if (action == "build") {
            Require(!std::filesystem::exists(path), "refusing to overwrite index");
            double build_seconds = 0, save_seconds = 0;
            uint64_t edges = 0;
            if (sinnamon) {
                SinnamonIVF::Config config;
                config.clusters = args.integer("--clusters", 1000);
                config.sketch_dim = args.integer("--sketch-dim", 1024);
                config.iterations = args.integer("--iterations", 20);
                config.seed = args.integer("--seed", 42);
                auto start = Clock::now();
                SinnamonIVF index(base, config); index.Build();
                build_seconds = std::chrono::duration<double>(Clock::now() - start).count();
                start = Clock::now(); index.Save(path);
                save_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            } else {
                std::ifstream input(args.get("--hnsw-index"), std::ios::binary | std::ios::ate);
                Require(input.good() && input.tellg() > 0, "cannot open native HNSW input");
                std::vector<char> data(static_cast<size_t>(input.tellg()));
                input.seekg(0); input.read(data.data(), data.size());
                Require(input.good(), "native HNSW input read failed");
                auto start = Clock::now();
                auto graph = LadrGraph::FromNativeHNSW(base, data);
                build_seconds = std::chrono::duration<double>(Clock::now() - start).count();
                edges = graph.neighbors.size();
                start = Clock::now(); graph.Save(base, path);
                save_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            }
            rusage usage{}; getrusage(RUSAGE_SELF, &usage);
            Json metric = {{"action", "build"}, {"method", method}, {"documents", base.size()},
                {"build_seconds", build_seconds}, {"serialize_seconds", save_seconds},
                {"build_time_scope", "loaded_input_to_built_index"}, {"metric_schema", 2},
                {"build_threads", threads}, {"edges", edges},
                {"index_bytes", std::filesystem::file_size(path)}, {"max_rss_kib", usage.ru_maxrss}};
            if (!sinnamon) metric["required_build_dependencies"] = {"dense_hnsw", "sindi_reorder"};
            std::cout << metric.dump() << std::endl;
            return 0;
        }
        auto queries = LoadVectors(args.get("--data"), "test");
        Require(queries.dim == base.dim, "query dimension mismatch");
        if (sinnamon) {
            auto index = SinnamonIVF::Load(base, path);
            index.PrepareQueries(queries.sparse);
            MeasureExtra([&](Id q, float a, size_t k, size_t point, Stats& stats) {
                return index.Search(queries.at(q), queries.sparse[q], a, k, point, stats);
            }, args, base, queries);
        } else {
            auto graph = LadrGraph::Load(base, path);
            LexicalSeeds seed_index(base, queries, args.get("--sindi-index"),
                                    args.real("--query-prune-ratio", .5F));
            Ladr index(base, graph); Ladr::Workspace work;
            const size_t seeds = args.integer("--seeds", 200), neighbors = args.integer("--neighbors", 128);
            Require(seeds > 0 && neighbors > 0 && neighbors <= 128, "invalid LADR settings");
            MeasureExtra([&](Id q, float a, size_t k, size_t point, Stats& stats) {
                auto ids = seed_index.Search(q, adaptive ? seeds : point);
                return index.Search(queries.at(q), queries.sparse[q], a, k, ids,
                                    adaptive ? point : 1, neighbors, adaptive, hybrid_rerank, work, stats);
            }, args, base, queries);
        }
        return 0;
    } catch (const H5::Exception& error) {
        std::cerr << "ERROR: " << error.getDetailMsg() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
    }
    return 1;
}
