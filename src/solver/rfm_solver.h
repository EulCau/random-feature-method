#pragma once

#include <torch/torch.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include "equation.h"
#include "rff.h"

class RFMSolver
{
public:
    RFMSolver(const Config& config, const std::shared_ptr<Equation>& eq, const uint64_t seed);

    const uint64_t seed() const { return seed_; }

    void compute_L(const torch::Tensor& t, const torch::Tensor& x);
    const torch::Tensor& L() const { return L_; }
    void compute_M(const torch::Tensor& t, const torch::Tensor& x);
    const torch::Tensor& M() const { return M_; }
    void compute_N(const torch::Tensor& t, const torch::Tensor& x);
    const torch::Tensor& N() const { return N_; }
    void compute_H(const torch::Tensor& t, const torch::Tensor& x);
    const torch::Tensor& H() const { return H_; }

private:
    Config config_;
    std::shared_ptr<Equation> equation_;
    uint64_t seed_;
    torch::Tensor L_;
    torch::Tensor M_;
    torch::Tensor N_;
    torch::Tensor H_;
};
