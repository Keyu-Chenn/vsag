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

// Per-query reuse of FP32 dense distances from a hybrid index's dense guide.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "typing.h"

namespace vsag {

// Direct-mapped storage bounds memory to 1 MiB per search thread. Collisions only
// discard reuse opportunities: the full ID, query generation and kernel must match.
class HybridDenseDistanceCache {
public:
    void
    Reset() {
        if (entries_.empty()) {
            entries_.resize(kSlots);
        }
        generation_ += 2;
        if (generation_ == 0) {
            std::fill(entries_.begin(), entries_.end(), Entry{});
            generation_ = 2;
        }
    }

    void
    Record(const InnerIdType* ids, const float* distances, InnerIdType count) {
        const auto batch_count = count - count % 4;
        for (InnerIdType i = 0; i < count; ++i) {
            auto& entry = entries_[slot(ids[i])];
            entry.key = key(ids[i], i < batch_count);
            entry.distance = distances[i];
        }
    }

    [[nodiscard]] bool
    Find(InnerIdType id, bool batch4, float& distance) const {
        const auto& entry = entries_[slot(id)];
        if (entry.key != key(id, batch4)) {
            return false;
        }
        distance = entry.distance;
        return true;
    }

private:
    static constexpr uint32_t kSlots = 1U << 16;
    struct Entry {
        uint64_t key{0};
        float distance{0};
    };

    static uint32_t
    slot(InnerIdType id) {
        return (id ^ (id >> 16)) & (kSlots - 1);
    }

    [[nodiscard]] uint64_t
    key(InnerIdType id, bool batch4) const {
        return (static_cast<uint64_t>(id) << 32) | generation_ | static_cast<uint32_t>(batch4);
    }

    std::vector<Entry> entries_;
    uint32_t generation_{0};
};

}  // namespace vsag
