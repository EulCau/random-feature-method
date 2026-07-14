#include "config.h"
#include <fstream>
#include <stdexcept>

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
        get_or<int64_t>(nonlinear, "batch_size", int64_t{0})
    };

    LinearSolveOptions linear_cfg{
        get_or<std::string>(linear, "solver", "ridge_dual"),
        get_or<int64_t>(linear, "batch_size", int64_t{0}),
        get_or<double>(linear, "ridge_lambda", solver.at("initial_lambda").get<float>())
    };

    SolverConfig solver_cfg{
        solver.at("use_linear_solver").get<bool>(),
        solver.at("num_iterations").get<int64_t>(),
        solver.at("sample_size").get<int64_t>(),
        get_or<int64_t>(solver, "test_sample_size", solver.at("sample_size").get<int64_t>()),
        get_or<int64_t>(solver, "test_batch_size", int64_t{0}),
        solver.at("hidden_dim").get<int64_t>(),
        solver.at("initial_lambda").get<float>(),
        solver.at("alpha_init_scale").get<float>(),
        linear_cfg,
        nonlinear_cfg
    };

    return Config{eqn_cfg, solver_cfg};
}
