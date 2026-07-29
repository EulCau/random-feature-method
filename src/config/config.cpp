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
            band.at("min").get<double>(),
            band.at("max").get<double>(),
            get_or<double>(band, "weight", 1.0)
        });
    }
    return bands;
}

NumericDType load_dtype(const json& solver)
{
    const auto name = get_or<std::string>(solver, "dtype", "float32");
    if (name == "float32")
    {
        return NumericDType::Float32;
    }
    if (name == "float64")
    {
        return NumericDType::Float64;
    }
    throw std::runtime_error(
        "solver_config.dtype must be \"float32\" or \"float64\", got: " +
        name
    );
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
    const auto dtype = load_dtype(solver);

    EqnConfig eqn_cfg{
        get_or<std::string>(eqn, "_comment", ""),
        eqn.at("equation_name").get<std::string>(),
        eqn.at("is_linear").get<bool>(),
        dtype,
        eqn.at("total_time").get<double>(),
        eqn.at("dimension").get<int64_t>(),
        eqn.at("num_time_intervals").get<int64_t>(),
        get_or<json>(eqn, "params", json::object())
    };

    NonlinearSolveOptions nonlinear_cfg{
        nonlinear.at("min_lambda").get<double>(),
        nonlinear.at("max_lambda").get<double>(),
        nonlinear.at("lambda_decrease").get<double>(),
        nonlinear.at("lambda_increase").get<double>(),
        nonlinear.at("error_tol").get<double>(),
        nonlinear.at("step_tol").get<double>(),
        nonlinear.at("max_retries").get<int64_t>(),
        get_or<std::string>(nonlinear, "step_solver", "normal"),
        get_or<int64_t>(nonlinear, "batch_size", int64_t{0}),
        get_or<bool>(nonlinear, "hard_terminal_lift", false),
        get_or<int64_t>(nonlinear, "consistency_points", int64_t{0}),
        get_or<std::vector<double>>(
            nonlinear,
            "consistency_time_fractions",
            std::vector<double>{}
        ),
        get_or<double>(nonlinear, "consistency_weight", 0.0),
        get_or<bool>(nonlinear, "normalize_residuals", false),
        get_or<bool>(nonlinear, "scale_jacobian_columns", false),
        get_or<double>(nonlinear, "column_scale_epsilon", 1.0e-6)
    };

    LinearSolveOptions linear_cfg{
        get_or<std::string>(linear, "solver", "ridge_dual"),
        get_or<int64_t>(linear, "batch_size", int64_t{0}),
        get_or<double>(linear, "ridge_lambda", solver.at("initial_lambda").get<double>())
    };

    RandomFeatureOptions random_feature_cfg{
        get_or<double>(random_feature, "scale_min", 0.5),
        get_or<double>(random_feature, "scale_max", 2.0),
        get_or<double>(random_feature, "space_scale", 1.0),
        get_or<double>(random_feature, "time_scale", 1.0),
        get_or<double>(random_feature, "bias_scale", 1.0),
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
        get_or<std::vector<double>>(
            reference_evaluation,
            "time_fractions",
            std::vector<double>{0.25, 0.5, 0.75}
        )
    };

    SolverConfig solver_cfg{
        solver.at("use_linear_solver").get<bool>(),
        dtype,
        solver.at("num_iterations").get<int64_t>(),
        solver.at("sample_size").get<int64_t>(),
        test_sample_size,
        test_batch_size,
        solver.at("hidden_dim").get<int64_t>(),
        solver.at("initial_lambda").get<double>(),
        get_or<double>(
            solver,
            "beta_init_scale",
            get_or<double>(solver, "alpha_init_scale", 0.001)
        ),
        random_feature_cfg,
        linear_cfg,
        nonlinear_cfg,
        reference_evaluation_cfg
    };

    return Config{eqn_cfg, solver_cfg};
}
