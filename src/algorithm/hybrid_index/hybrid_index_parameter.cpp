//
// Created by root on 2026/1/8.
//

#include "hybrid_index_parameter.h"
#include <fmt/format.h>
#include "datacell/flatten_datacell_parameter.h"
#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {
void
HybridIndexParameter::FromJson(const JsonType& json) {
    JsonType sparse_p = json;
    alpha = json["alpha"].GetFloat();
}

}  // namespace vsag