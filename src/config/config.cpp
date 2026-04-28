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
        nonlinear.at("max_retries").get<int64_t>()
    };

    SolverConfig solver_cfg{
        solver.at("use_linear_solver").get<bool>(),
        solver.at("num_iterations").get<int64_t>(),
        solver.at("sample_size").get<int64_t>(),
        solver.at("hidden_dim").get<int64_t>(),
        solver.at("initial_lambda").get<float>(),
        solver.at("alpha_init_scale").get<float>(),
        nonlinear_cfg
    };

    return Config{eqn_cfg, solver_cfg};
}
