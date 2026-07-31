
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

#include "memory_io.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

namespace {

class CountingAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "CountingAllocator";
    }

    void*
    Allocate(size_t size) override {
        return std::malloc(size);
    }

    void
    Deallocate(void* p) override {
        std::free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        ++reallocate_count;
        return std::realloc(p, size);
    }

    size_t reallocate_count{0};
};

}  // namespace

TEST_CASE("MemoryIO Read and Write", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<MemoryIO>(allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("MemoryIO Serialize and Deserialize", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto wio = std::make_unique<MemoryIO>(allocator.get());
    auto rio = std::make_unique<MemoryIO>(allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}

TEST_CASE("MemoryIO grows geometrically", "[ut][MemoryIO]") {
    CountingAllocator allocator;
    MemoryIO io(&allocator);
    constexpr uint64_t count = 1U << 20U;
    constexpr uint8_t value = 0x5A;

    for (uint64_t i = 0; i < count; ++i) {
        io.Write(&value, sizeof(value), i);
    }

    REQUIRE(io.size_ == count);
    REQUIRE(allocator.reallocate_count <= 20);

    uint8_t actual = 0;
    REQUIRE(io.Read(sizeof(actual), count - 1, &actual));
    REQUIRE(actual == value);
}
