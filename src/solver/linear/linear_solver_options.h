#pragma once

#include <cstdint>

enum class LinearSolverType
{
    Constant,
    RidgeDual,
    QR,
    BatchedQR
};

struct LinearSolverOptions
{
    LinearSolverType solver_type{LinearSolverType::RidgeDual};
    int64_t qr_batch_size{0};
    double ridge_lambda{1e-6};
};
