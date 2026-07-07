#pragma once

#include <torch/torch.h>
#include <random>
#include "equation.h"
#include "linear_solver_options.h"
#include "rff.h"

class RFMSolver
{
public:
    RFMSolver(
        const Config& config, const std::shared_ptr<Equation>& eq,
        torch::Device device, uint64_t seed);

    RFMSolver(
        Config config, const std::shared_ptr<Equation>& eq,
        torch::Device device, uint64_t seed, bool is_linear);

    RFMSolver& options(
        const std::optional<torch::Tensor>& y0,
        const std::optional<torch::Tensor>& alpha,
        std::optional<float> lambda
    );

    RFMSolver& linear_options(const LinearSolverOptions& options);

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve(bool output_log = false) const;
    [[nodiscard]] float test(const torch::Tensor& y0, const torch::Tensor& alpha) const;

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
    [[nodiscard]] const torch::Tensor& H() const { return H_; }

    [[nodiscard]] const torch::Tensor& y0() const { return y0_; }
    [[nodiscard]] const torch::Tensor& alpha() const { return alpha_; }
    [[nodiscard]] float lambda() const { return lambda_; }
    [[nodiscard]] const LinearSolverOptions& linear_solver_options() const { return linear_solver_options_; }

protected:
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_linear() const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_nonlinear(bool output_log) const;

    [[nodiscard]] std::pair<const torch::Tensor, const torch::Tensor> compute_linear_coef() const;

    [[nodiscard]] std::pair<const torch::Tensor, const torch::Tensor> compute_linear_coef_batch(
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_linear_batched_qr(
        const LinearSolverOptions& options) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_linear_constant_baseline(
        const LinearSolverOptions& options) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_nonlinear_levenberg_marquardt(
        const torch::Tensor & y0, const torch::Tensor & alpha, float lambda, bool output_log) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_nonlinear_constant_baseline(
        const torch::Tensor& y0,
        float lambda,
        bool output_log) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_terminal_residual(
        const torch::Tensor& theta) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> compute_nonlinear_terminal_residual_and_jacobian(
        const torch::Tensor& theta) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_terminal_residual_batch(
        const torch::Tensor& theta,
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> compute_nonlinear_terminal_residual_and_jacobian_batch(
        const torch::Tensor& theta,
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] std::tuple<torch::Tensor, float, float> solve_nonlinear_lm_step_batched_qr(
        const torch::Tensor& theta,
        float lambda,
        int64_t batch_size) const;

    [[nodiscard]] std::pair<float, float> compute_nonlinear_loss_error_batched(
        const torch::Tensor& theta,
        int64_t batch_size) const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sample_nonlinear_batch(
        int64_t row_begin,
        int64_t row_end) const;

    [[nodiscard]] torch::Tensor forward_nonlinear_terminal_y(
        const torch::Tensor& y0, const torch::Tensor& alpha) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_z(const torch::Tensor& alpha) const;

    [[nodiscard]] std::pair<double, int64_t> test_batch(
        const torch::Tensor& y0,
        const torch::Tensor& alpha,
        int64_t batch_size) const;

    void compute_time_grid();
    void compute_txw();
    void prepare_full_linear_cache();
    void clear_full_linear_cache();
    void compute_L(const torch::Tensor& t, const torch::Tensor& x);
    void compute_M(const torch::Tensor& t, const torch::Tensor& x);
    void compute_N(const torch::Tensor& t, const torch::Tensor& x);
    void compute_H(const torch::Tensor& t, const torch::Tensor& x);

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
    torch::Tensor H_;
    torch::Tensor t_;
    torch::Tensor y0_;
    torch::Tensor alpha_;
    float lambda_{};
    LinearSolverOptions linear_solver_options_;
};
