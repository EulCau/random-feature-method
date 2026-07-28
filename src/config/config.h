#pragma once

#include "json.hpp"
#include <cstdint>
#include <string>
#include <vector>

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
    std::string step_solver;
    int64_t batch_size;
    bool hard_terminal_lift;
    int64_t consistency_points;
    float consistency_weight;
    bool normalize_residuals;
    bool scale_jacobian_columns;
    float column_scale_epsilon;
};

struct LinearSolveOptions
{
    std::string solver;
    int64_t batch_size;
    double ridge_lambda;
};

struct RandomFeatureScaleBand
{
    float scale_min;
    float scale_max;
    float weight;
};

struct RandomFeatureOptions
{
    float scale_min;
    float scale_max;
    float space_scale;
    float time_scale;
    float bias_scale;
    std::vector<RandomFeatureScaleBand> scale_bands;
};

struct ReferenceEvaluationOptions
{
    bool enabled;
    int64_t sample_size;
    int64_t batch_size;
    std::vector<float> time_fractions;
};

struct SolverConfig
{
    bool use_linear_solver;
    int64_t num_iterations;
    int64_t sample_size;
    int64_t test_sample_size;
    int64_t test_batch_size;
    int64_t hidden_dim;
    float initial_lambda;
    float beta_init_scale;
    RandomFeatureOptions random_feature;
    LinearSolveOptions linear;
    NonlinearSolveOptions nonlinear;
    ReferenceEvaluationOptions reference_evaluation;
};

struct Config
{
    EqnConfig eqn_config;
    SolverConfig solver_config;
};

Config load_config(const std::string& json_path);
