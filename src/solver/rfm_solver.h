#pragma once

#include <torch/torch.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include "equation.h"
#include "rff.h"

struct SolveResult
{
    torch::Tensor alpha;        // (H*d)
    torch::Tensor terminal_err; // 标量，L2误差
    torch::Tensor M;            // (K,H*d) 可选，便于调试
    torch::Tensor beta;         // (K)     可选
};

SolveResult solve(const Config& config, const std::shared_ptr<Equation>& eq);
