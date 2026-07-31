// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uhg_mixed_alpha {

inline std::vector<float>
ReadNPYFloat32(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mixed alpha npy file: " + path);
    }

    char magic[8];
    file.read(magic, 8);
    if (!file || magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Invalid mixed alpha npy file: " + path);
    }

    const uint8_t major = static_cast<uint8_t>(magic[6]);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hlen = 0;
        file.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
        header_len = hlen;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("Unsupported npy version in mixed alpha file: " + path);
    }

    std::string header(header_len, '\0');
    file.read(header.data(), header_len);
    if (!file || (header.find("'<f4'") == std::string::npos &&
                  header.find("'|f4'") == std::string::npos &&
                  header.find("\"<f4\"") == std::string::npos &&
                  header.find("\"|f4\"") == std::string::npos)) {
        throw std::runtime_error("Mixed alpha npy file must have dtype float32: " + path);
    }
    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": True") != std::string::npos) {
        throw std::runtime_error("Fortran-order mixed alpha npy file is not supported: " + path);
    }

    shape.clear();
    const size_t shape_pos = header.find("shape");
    const size_t paren_start = header.find('(', shape_pos);
    const size_t paren_end = header.find(')', paren_start);
    if (shape_pos == std::string::npos || paren_start == std::string::npos ||
        paren_end == std::string::npos) {
        throw std::runtime_error("Failed to parse mixed alpha npy shape: " + path);
    }
    std::stringstream shape_stream(
        header.substr(paren_start + 1, paren_end - paren_start - 1));
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        const auto first = token.find_first_not_of(" \t");
        if (first == std::string::npos) {
            continue;
        }
        const auto last = token.find_last_not_of(" \t");
        shape.push_back(std::stoll(token.substr(first, last - first + 1)));
    }
    if (shape.size() != 1 || shape[0] <= 0) {
        throw std::runtime_error("Mixed alpha npy file must have shape (num_queries,): " + path);
    }

    std::vector<float> values(static_cast<size_t>(shape[0]));
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("Mixed alpha npy data is truncated: " + path);
    }
    for (float alpha : values) {
        if (alpha < 0.0F || alpha > 1.0F) {
            throw std::runtime_error("Mixed alpha values must be in [0, 1]: " + path);
        }
    }
    return values;
}

inline bool
ValidateMixedFiles(const std::string& alpha_file, const std::string& gt_file) {
    const bool has_alpha = !alpha_file.empty();
    const bool has_gt = !gt_file.empty();
    if (has_alpha != has_gt) {
        throw std::runtime_error(
            "--mixed_alpha_file and --mixed_gt_file must be provided together");
    }
    return has_alpha;
}

}  // namespace uhg_mixed_alpha
