#include "config.h"
#include <fstream>

using json = nlohmann::json;

Config load_config(const std::string& json_path)
{
    std::ifstream in(json_path);
    if (!in)
    {
        throw std::runtime_error("Failed to open config file: " + json_path);
    }

    json j;
    in >> j;

    EqnConfig eqn_cfg =
    {
        j["eqn_config"]["eqn_name"],
        j["eqn_config"]["total_time"],
        j["eqn_config"]["dim"],
        j["eqn_config"]["num_time_interval"]
    };

    NetConfig net_cfg =
    {
        j["net_config"]["y_init_range"].get<std::vector<float>>(),
        j["net_config"]["num_hiddens"].get<std::vector<int64_t>>(),
        j["net_config"]["lr_values"].get<std::vector<float>>(),
        j["net_config"]["lr_boundaries"].get<std::vector<int64_t>>(),
        j["net_config"]["num_iterations"],
        j["net_config"]["batch_size"],
        j["net_config"]["valid_size"],
        j["net_config"]["dtype"],
        j["net_config"]["verbose"],
        j["net_config"]["logging_frequency"],
        j["net_config"]["warmup_steps"]
    };

    return Config{ eqn_cfg, net_cfg };
}
