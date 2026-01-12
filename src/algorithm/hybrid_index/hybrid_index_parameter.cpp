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
HybridIndexParameter::HybridIndexParameter(): base_codes_param(nullptr) {
}

void
HybridIndexParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("bruteforce parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);
}

JsonType
HybridIndexParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_TYPE_BRUTE_FORCE);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    return json;
}

bool
HybridIndexParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    auto brute_force_param = std::dynamic_pointer_cast<HybridIndexParameter>(other);
    if (not brute_force_param) {
        logger::error(
            "BruteForceParameter::CheckCompatibility: "
            "other parameter is not a BruteForceParameter");
        return false;
    }
    return this->base_codes_param->CheckCompatibility(brute_force_param->base_codes_param);
}


}