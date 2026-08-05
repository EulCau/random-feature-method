#pragma once

#include <torch/torch.h>
#include <limits>
#include <random>
#include "equation.h"
#include "linear_solver_options.h"
#include "rff.h"

struct RFMSolverDiagnostics
{
    double objective_rmse{std::numeric_limits<double>::quiet_NaN()};
    double test_terminal_std{std::numeric_limits<double>::quiet_NaN()};
    double normalized_test_rmse{std::numeric_limits<double>::quiet_NaN()};
    double explained_terminal_variance{std::numeric_limits<double>::quiet_NaN()};
    double beta_norm{std::numeric_limits<double>::quiet_NaN()};
    double final_gradient_inf_norm{std::numeric_limits<double>::quiet_NaN()};
    double final_damping{std::numeric_limits<double>::quiet_NaN()};
    int64_t accepted_lm_iterations{0};
};

struct PathValueEvaluation
{
    torch::Tensor t;
    torch::Tensor x;
    torch::Tensor direct_value;
    torch::Tensor propagated_value;
};

class RFMSolver
{
public:
    // The default scalar model is centered at (0, x0). The optional hard
    // terminal lift additionally enforces u(0, x0) = y0 and u(T, x) = g(x).
    RFMSolver(
        const Config& config, const std::shared_ptr<Equation>& eq,
        torch::Device device, uint64_t seed);

    RFMSolver(
        Config config, const std::shared_ptr<Equation>& eq,
        torch::Device device, uint64_t seed, bool is_linear);

    RFMSolver& options(
        const std::optional<torch::Tensor>& y0,
        const std::optional<torch::Tensor>& beta,
        std::optional<double> lambda
    );

    RFMSolver& linear_options(const LinearSolverOptions& options);

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve(bool output_log = false) const;
    [[nodiscard]] double test(const torch::Tensor& y0, const torch::Tensor& beta) const;
    // Evaluate both u_theta(t_k, X_k) and the BSDE-forward value on every
    // node of a path generated consistently with equation_. Inputs have
    // shapes dw=(B, D, N) and x=(B, D, N + 1); outputs use (B, N + 1, ...).
    [[nodiscard]] PathValueEvaluation evaluate_path_values(
        const torch::Tensor& y0,
        const torch::Tensor& beta,
        const torch::Tensor& dw_sample,
        const torch::Tensor& x_sample) const;

    [[nodiscard]] uint64_t seed() const { return seed_; }
    [[nodiscard]] bool is_linear() const { return is_linear_; }
    [[nodiscard]] torch::Device device() const { return device_; }
    [[nodiscard]] const torch::Tensor& t() const { return t_; }
    [[nodiscard]] const torch::Tensor& t_end() const { return t_end_; }
    [[nodiscard]] const torch::Tensor& dw() const { return dw_; }
    [[nodiscard]] const torch::Tensor& x() const { return x_; }
    [[nodiscard]] const torch::Tensor& x_end() const { return x_end_; }

    [[nodiscard]] const torch::Tensor& L() const { return L_; }
    [[nodiscard]] const torch::Tensor& M() const { return M_; }
    [[nodiscard]] const torch::Tensor& N() const { return N_; }

    [[nodiscard]] const torch::Tensor& y0() const { return y0_; }
    [[nodiscard]] const torch::Tensor& beta() const { return beta_; }
    [[nodiscard]] double lambda() const { return lambda_; }
    [[nodiscard]] const LinearSolverOptions& linear_solver_options() const { return linear_solver_options_; }
    [[nodiscard]] const RFMSolverDiagnostics& diagnostics() const { return diagnostics_; }

protected:
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_linear() const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_nonlinear(bool output_log) const;

    [[nodiscard]] std::pair<const torch::Tensor, const torch::Tensor> compute_linear_coef() const;

    [[nodiscard]] std::pair<const torch::Tensor, const torch::Tensor> compute_linear_coef_batch(
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_linear_batched_qr(
        const LinearSolverOptions& options) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_linear_constant_baseline(
        const LinearSolverOptions& options) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_nonlinear_levenberg_marquardt(
        const torch::Tensor& y0,
        const torch::Tensor& beta,
        double lambda,
        bool output_log) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> solve_nonlinear_constant_baseline(
        const torch::Tensor& y0,
        double lambda,
        bool output_log) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_objective_residual(
        const torch::Tensor& theta) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> compute_nonlinear_objective_residual_and_jacobian(
        const torch::Tensor& theta) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_objective_residual_batch(
        const torch::Tensor& theta,
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> compute_nonlinear_objective_residual_and_jacobian_batch(
        const torch::Tensor& theta,
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::tuple<torch::Tensor, double, double> solve_nonlinear_lm_step_batched_qr(
        const torch::Tensor& theta,
        double lambda,
        int64_t batch_size,
        bool output_spectrum) const;

    [[nodiscard]] std::pair<double, double> compute_nonlinear_loss_error_batched(
        const torch::Tensor& theta,
        int64_t batch_size) const;

    [[nodiscard]] double compute_nonlinear_terminal_error_batched(
        const torch::Tensor& theta,
        int64_t batch_size) const;

    [[nodiscard]] double compute_nonlinear_gradient_inf_norm(
        const torch::Tensor& theta,
        int64_t batch_size) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sample_nonlinear_batch(
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] torch::Tensor forward_nonlinear_terminal_y(
        const torch::Tensor& y0,
        const torch::Tensor& beta) const;

    [[nodiscard]] torch::Tensor compute_z(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& beta) const;

    [[nodiscard]] torch::Tensor contract_z_features(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& weights) const;

    [[nodiscard]] torch::Tensor compute_residual_for_samples(
        const torch::Tensor& theta,
        const torch::Tensor& t,
        const torch::Tensor& t_end,
        const torch::Tensor& x,
        const torch::Tensor& x_end,
        const torch::Tensor& dw,
        bool objective) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
    compute_objective_residual_and_jacobian_for_samples(
        const torch::Tensor& theta,
        const torch::Tensor& t,
        const torch::Tensor& t_end,
        const torch::Tensor& x,
        const torch::Tensor& x_end,
        const torch::Tensor& dw) const;

    [[nodiscard]] std::tuple<double, double, double, int64_t> test_batch(
        const torch::Tensor& y0,
        const torch::Tensor& beta,
        int64_t batch_size) const;

    [[nodiscard]] torch::Tensor compute_direct_nonlinear_value(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& x_initial,
        const torch::Tensor& y0,
        const torch::Tensor& beta) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_z_features(
        const torch::Tensor& t,
        const torch::Tensor& x) const;

    [[nodiscard]] bool use_hard_terminal_lift() const;
    [[nodiscard]] std::vector<int64_t> nonlinear_consistency_indices(
        int64_t time_count) const;
    [[nodiscard]] torch::Tensor nonlinear_residual_scale() const;

    void compute_time_grid();
    void compute_txw();
    void prepare_full_linear_cache();
    void clear_full_linear_cache();
    void compute_L(const torch::Tensor& t, const torch::Tensor& x);
    void compute_M(const torch::Tensor& t, const torch::Tensor& x);
    void compute_N(const torch::Tensor& t, const torch::Tensor& x);

    void check_tx_shape(const torch::Tensor& t, const torch::Tensor& x) const;

    bool is_linear_{};
    Config config_;
    std::shared_ptr<Equation> equation_;
    uint64_t seed_;
    torch::Device device_;
    RandomFeatureFunction rff_;
    torch::Tensor t_end_;
    torch::Tensor dw_;
    torch::Tensor x_;
    torch::Tensor x_end_;
    torch::Tensor L_;
    torch::Tensor M_;
    torch::Tensor N_;
    torch::Tensor t_;
    torch::Tensor y0_;
    torch::Tensor beta_;
    double lambda_{};
    LinearSolverOptions linear_solver_options_;
    mutable RFMSolverDiagnostics diagnostics_;
    mutable torch::Tensor nonlinear_residual_scale_;
};
