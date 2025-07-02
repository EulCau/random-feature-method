#pragma once

#include "json.hpp"
#include <string>
#include <vector>

struct EqnConfig
{
    std::string eqn_name;
    float total_time;
    int64_t dim;
    int64_t num_time_interval;
};

struct NetConfig
{
    std::vector<float> y_init_range;
    std::vector<int64_t> num_hiddens;
    std::vector<float> lr_values;
    std::vector<int64_t> lr_boundaries;
    int64_t num_iterations;
    int64_t batch_size;
    int64_t valid_size;
    std::string dtype;
    bool verbose;
    int64_t logging_frequency;
	int64_t warmup_steps;
};

struct Config
{
    EqnConfig eqn_config;
    NetConfig net_config;
};

Config load_config(const std::string& json_path);
