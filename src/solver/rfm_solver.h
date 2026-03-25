#pragma once

#include <torch/torch.h>
#include <random>
#include "equation.h"
#include "rff.h"

class RFMSolver
{
public:
    RFMSolver(const Config& config, const std::shared_ptr<Equation>& eq, torch::Device device, uint64_t seed);

    [[nodiscard]] uint64_t seed() const { return seed_; }

    void compute_txw();
    [[nodiscard]] torch::Device device() const { return device_; }
    [[nodiscard]] const torch::Tensor& t() const { return t_; }
    [[nodiscard]] const torch::Tensor& t_end() const { return t_end_; }
    [[nodiscard]] const torch::Tensor& dw() const { return dw_; }
    [[nodiscard]] const torch::Tensor& x() const { return x_; }
    [[nodiscard]] const torch::Tensor& x_end() const { return x_end_; }

    void compute_L(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& L() const { return L_; }
    void compute_M(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& M() const { return M_; }
    void compute_N(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& N() const { return N_; }
    void compute_H(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& H() const { return H_; }

    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> Solve() const;

protected:
    void check_tx_shape(const torch::Tensor& t, const torch::Tensor& x) const;

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
};
