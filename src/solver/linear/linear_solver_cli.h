#pragma once

#include <cstdint>

#include "linear_solver_options.h"

[[nodiscard]] LinearSolverOptions get_linear_solver_options_from_terminal(
    int64_t default_qr_batch_size,
    double ridge_lambda);
