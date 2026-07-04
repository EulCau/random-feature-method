#pragma once

#include <cstdint>

#include "qr_decomposition.h"

enum class LinearSolverType
{
    RidgeDual,
    QR,
    BatchedQR
};

struct LinearSolverOptions
{
    LinearSolverType solver_type{LinearSolverType::RidgeDual};
    solver_utils::QRMethod qr_method{solver_utils::QRMethod::Householder};
    int64_t qr_batch_size{0};
    double ridge_lambda{1e-6};
};
