#include "morphserve/control/simulator.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string output_dir;
    std::string real_report;
    std::string results_dir = "results";
    std::string bench_root = "data/interleaved-bench-v1";
    std::string sources_dir = "data/bench_sources";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--real-report" && i + 1 < argc) {
            real_report = argv[++i];
        } else if (arg == "--results-dir" && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (arg == "--bench-root" && i + 1 < argc) {
            bench_root = argv[++i];
        } else if (arg == "--sources-dir" && i + 1 < argc) {
            sources_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: morphserve-sim --output-dir <DIR> [--real-report <JSON>] "
                      << "[--results-dir <DIR>] [--bench-root <DIR>] [--sources-dir <DIR>]\n";
            return 0;
        }
    }

    if (output_dir.empty()) {
        std::cerr << "error: --output-dir is required\n";
        return 1;
    }

    try {
        const auto rep = morphserve::control::ClusterSimulator::run_scale_and_interconnect_sweep(
            output_dir, real_report, results_dir, bench_root, sources_dir);
        return (rep.value("status", "") == "ok") ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "morphserve-sim fatal: " << e.what() << "\n";
        return 1;
    }
}
