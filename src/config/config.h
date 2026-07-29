#pragma once

#include "json.hpp"
#include <cstdint>
#include <string>
#include <vector>

enum class NumericDType
{
    Float32,
    Float64
};

struct EqnConfig
{
    std::string comment;
    std::string equation_name;
    bool is_linear;
    NumericDType dtype;
    double total_time;
    int64_t dimension;
    int64_t num_time_intervals;
    nlohmann::json params;
};

struct NonlinearSolveOptions
{
    double min_lambda;
    double max_lambda;
    double lambda_decrease;
    double lambda_increase;
    double error_tol;
    double step_tol;
    int64_t max_retries;
    std::string step_solver;
    int64_t batch_size;
    bool hard_terminal_lift;
    int64_t consistency_points;
    std::vector<double> consistency_time_fractions;
    double consistency_weight;
    bool normalize_residuals;
    bool scale_jacobian_columns;
    double column_scale_epsilon;
};

struct LinearSolveOptions
{
    std::string solver;
    int64_t batch_size;
    double ridge_lambda;
};

struct RandomFeatureScaleBand
{
    double scale_min;
    double scale_max;
    double weight;
};

struct RandomFeatureOptions
{
    double scale_min;
    double scale_max;
    double space_scale;
    double time_scale;
    double bias_scale;
    std::vector<RandomFeatureScaleBand> scale_bands;
};

struct ReferenceEvaluationOptions
{
    bool enabled;
    int64_t sample_size;
    int64_t batch_size;
    std::vector<double> time_fractions;
};

struct SolverConfig
{
    bool use_linear_solver;
    NumericDType dtype;
    int64_t num_iterations;
    int64_t sample_size;
    int64_t test_sample_size;
    int64_t test_batch_size;
    int64_t hidden_dim;
    double initial_lambda;
    double beta_init_scale;
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
