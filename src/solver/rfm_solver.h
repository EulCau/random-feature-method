#pragma once

#include <torch/torch.h>
#include <random>
#include "equation.h"
#include "rff.h"
#include "utils/qr_decomposition.h"

enum class LinearSolverType
{
    RidgeDual,
    QR,
    BatchedQR
};

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

    RFMSolver& linear_options(
        LinearSolverType solver_type,
        solver_utils::QRMethod qr_method,
        int64_t qr_batch_size
    );

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
    [[nodiscard]] LinearSolverType linear_solver_type() const { return linear_solver_type_; }
    [[nodiscard]] solver_utils::QRMethod qr_method() const { return qr_method_; }
    [[nodiscard]] int64_t qr_batch_size() const { return qr_batch_size_; }

protected:
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_linear() const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_nonlinear(bool output_log) const;

    [[nodiscard]] std::pair<const torch::Tensor, const torch::Tensor> compute_linear_coef() const;

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> solve_nonlinear_levenberg_marquardt(
        const torch::Tensor & y0, const torch::Tensor & alpha, float lambda, bool output_log) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_terminal_residual(
        const torch::Tensor& theta) const;

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> compute_nonlinear_terminal_residual_and_jacobian(
        const torch::Tensor& theta) const;

    [[nodiscard]] torch::Tensor forward_nonlinear_terminal_y(
        const torch::Tensor& y0, const torch::Tensor& alpha) const;

    [[nodiscard]] torch::Tensor compute_nonlinear_z(const torch::Tensor& alpha) const;

    void compute_txw();
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
    LinearSolverType linear_solver_type_{LinearSolverType::RidgeDual};
    solver_utils::QRMethod qr_method_{solver_utils::QRMethod::Householder};
    int64_t qr_batch_size_{0};
};
