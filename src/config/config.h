#pragma once

#include "json.hpp"
#include <cstdint>
#include <string>

struct EqnConfig
{
    std::string comment;
    std::string equation_name;
    bool is_linear;
    float total_time;
    int64_t dimension;
    int64_t num_time_intervals;
    nlohmann::json params;
};

struct NonlinearSolveOptions
{
    float min_lambda;
    float max_lambda;
    float lambda_decrease;
    float lambda_increase;
    float error_tol;
    float step_tol;
    int64_t max_retries;
};

struct SolverConfig
{
    bool use_linear_solver;
    int64_t num_iterations;
    int64_t sample_size;
    int64_t hidden_dim;
    float initial_lambda;
    float alpha_init_scale;
    NonlinearSolveOptions nonlinear;
};

struct Config
{
    EqnConfig eqn_config;
    SolverConfig solver_config;
};

Config load_config(const std::string& json_path);
