#pragma once

#include <torch/torch.h>
#include <random>
#include "equation.h"

class RFMSolver
{
public:
    RFMSolver(const Config& config, const std::shared_ptr<Equation>& eq, uint64_t seed);

    [[nodiscard]] uint64_t seed() const { return seed_; }

    void compute_L(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& L() const { return L_; }
    void compute_M(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& M() const { return M_; }
    void compute_N(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& N() const { return N_; }
    void compute_H(const torch::Tensor& t, const torch::Tensor& x);
    [[nodiscard]] const torch::Tensor& H() const { return H_; }

private:
    Config config_;
    std::shared_ptr<Equation> equation_;
    uint64_t seed_;
    torch::Tensor L_;
    torch::Tensor M_;
    torch::Tensor N_;
    torch::Tensor H_;
};
