#include "config.h"
#include <fstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

namespace
{
template <typename T>
T get_or(const json& j, const char* key, const T& default_value)
{
    if (!j.contains(key))
    {
        return default_value;
    }
    return j.at(key).get<T>();
}

std::vector<RandomFeatureScaleBand> load_scale_bands(
    const json& random_feature)
{
    std::vector<RandomFeatureScaleBand> bands;
    if (!random_feature.contains("scale_bands"))
    {
        return bands;
    }

    for (const auto& band : random_feature.at("scale_bands"))
    {
        bands.push_back(RandomFeatureScaleBand{
            band.at("min").get<float>(),
            band.at("max").get<float>(),
            get_or<float>(band, "weight", 1.0f)
        });
    }
    return bands;
}
}

Config load_config(const std::string& json_path)
{
    std::ifstream in(json_path);
    if (!in)
    {
        throw std::runtime_error("Failed to open config file: " + json_path);
    }

    json j;
    in >> j;

    const auto& eqn = j.at("eqn_config");
    const auto& solver = j.at("solver_config");
    const auto linear = get_or<json>(solver, "linear", json::object());
    const auto random_feature = get_or<json>(
        solver,
        "random_feature",
        json::object()
    );
    const auto reference_evaluation = get_or<json>(
        solver,
        "reference_evaluation",
        json::object()
    );
    const auto& nonlinear = solver.at("nonlinear");

    EqnConfig eqn_cfg{
        get_or<std::string>(eqn, "_comment", ""),
        eqn.at("equation_name").get<std::string>(),
        eqn.at("is_linear").get<bool>(),
        eqn.at("total_time").get<float>(),
        eqn.at("dimension").get<int64_t>(),
        eqn.at("num_time_intervals").get<int64_t>(),
        get_or<json>(eqn, "params", json::object())
    };

    NonlinearSolveOptions nonlinear_cfg{
        nonlinear.at("min_lambda").get<float>(),
        nonlinear.at("max_lambda").get<float>(),
        nonlinear.at("lambda_decrease").get<float>(),
        nonlinear.at("lambda_increase").get<float>(),
        nonlinear.at("error_tol").get<float>(),
        nonlinear.at("step_tol").get<float>(),
        nonlinear.at("max_retries").get<int64_t>(),
        get_or<std::string>(nonlinear, "step_solver", "normal"),
        get_or<int64_t>(nonlinear, "batch_size", int64_t{0}),
        get_or<bool>(nonlinear, "hard_terminal_lift", false),
        get_or<int64_t>(nonlinear, "consistency_points", int64_t{0}),
        get_or<float>(nonlinear, "consistency_weight", 0.0f),
        get_or<bool>(nonlinear, "normalize_residuals", false),
        get_or<bool>(nonlinear, "scale_jacobian_columns", false),
        get_or<float>(nonlinear, "column_scale_epsilon", 1.0e-6f)
    };

    LinearSolveOptions linear_cfg{
        get_or<std::string>(linear, "solver", "ridge_dual"),
        get_or<int64_t>(linear, "batch_size", int64_t{0}),
        get_or<double>(linear, "ridge_lambda", solver.at("initial_lambda").get<float>())
    };

    RandomFeatureOptions random_feature_cfg{
        get_or<float>(random_feature, "scale_min", 0.5f),
        get_or<float>(random_feature, "scale_max", 2.0f),
        get_or<float>(random_feature, "space_scale", 1.0f),
        get_or<float>(random_feature, "time_scale", 1.0f),
        get_or<float>(random_feature, "bias_scale", 1.0f),
        load_scale_bands(random_feature)
    };

    const int64_t test_sample_size = get_or<int64_t>(
        solver,
        "test_sample_size",
        solver.at("sample_size").get<int64_t>()
    );
    const int64_t test_batch_size = get_or<int64_t>(
        solver,
        "test_batch_size",
        int64_t{0}
    );
    ReferenceEvaluationOptions reference_evaluation_cfg{
        get_or<bool>(reference_evaluation, "enabled", false),
        get_or<int64_t>(
            reference_evaluation,
            "sample_size",
            test_sample_size
        ),
        get_or<int64_t>(
            reference_evaluation,
            "batch_size",
            test_batch_size
        ),
        get_or<std::vector<float>>(
            reference_evaluation,
            "time_fractions",
            std::vector<float>{0.25f, 0.5f, 0.75f}
        )
    };

    SolverConfig solver_cfg{
        solver.at("use_linear_solver").get<bool>(),
        solver.at("num_iterations").get<int64_t>(),
        solver.at("sample_size").get<int64_t>(),
        test_sample_size,
        test_batch_size,
        solver.at("hidden_dim").get<int64_t>(),
        solver.at("initial_lambda").get<float>(),
        get_or<float>(
            solver,
            "beta_init_scale",
            get_or<float>(solver, "alpha_init_scale", 0.001f)
        ),
        random_feature_cfg,
        linear_cfg,
        nonlinear_cfg,
        reference_evaluation_cfg
    };

    return Config{eqn_cfg, solver_cfg};
}
