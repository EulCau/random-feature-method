#include "rfm_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include "linear_least_squares_solver.h"
#include "nonlinear_solver_utils.h"
#include "rff.h"

namespace
{

[[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, double> split_linear_solution(
    const torch::Tensor& x,
    const int64_t hidden_dim,
    const double rmse)
{
    const auto x_matrix = x.reshape({-1, 1}).contiguous();
    const auto y0 = x_matrix.index({0, 0}).clone();
    const auto beta = x_matrix.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({hidden_dim}).contiguous();

    return {y0, beta, rmse};
}

[[nodiscard]] int64_t resolve_batch_size(
    const int64_t configured_batch_size,
    const int64_t hidden_dim,
    const char* name)
{
    TORCH_CHECK(configured_batch_size >= 0, name, " batch_size must be nonnegative");
    if (configured_batch_size == 0)
    {
        return 4 * (1 + hidden_dim);
    }
    return configured_batch_size;
}

[[nodiscard]] uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ x >> 30) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ x >> 27) * 0x94D049BB133111EBULL;
    return x ^ x >> 31;
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> reduce_linear_qr_libtorch(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.sizes());
    TORCH_CHECK(A.size(0) == B.size(0), "A and B row mismatch: A=", A.sizes(), ", B=", B.sizes());
    TORCH_CHECK(A.size(0) >= A.size(1), "QR reduction requires rows >= cols, got ", A.sizes());

    const auto n = A.size(1);
    const auto [qr_data, tau] = torch::geqrf(A.contiguous());
    const auto R = qr_data.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).triu().contiguous();
    const auto transformed_rhs = torch::ormqr(qr_data, tau, B.contiguous(), true, true);
    const auto rhs = transformed_rhs.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).contiguous();
    return {R.contiguous(), rhs.contiguous()};
}

[[nodiscard]] double floating_point_epsilon(
    const torch::ScalarType scalar_type)
{
    if (scalar_type == torch::kFloat32)
    {
        return std::numeric_limits<float>::epsilon();
    }
    TORCH_CHECK(
        scalar_type == torch::kFloat64,
        "Jacobian spectrum diagnostics support float32 and float64, got ",
        scalar_type
    );
    return std::numeric_limits<double>::epsilon();
}

void print_jacobian_spectrum(
    const torch::Tensor& singular_values_descending,
    const int64_t row_count,
    const int64_t column_count,
    const torch::ScalarType source_scalar_type)
{
    TORCH_CHECK(
        singular_values_descending.dim() == 1 &&
        singular_values_descending.numel() == column_count,
        "singular value vector must have one entry per Jacobian column"
    );
    const auto singular_values =
        singular_values_descending.to(torch::kFloat64).contiguous();
    const double epsilon = floating_point_epsilon(source_scalar_type);
    const double sigma_max = singular_values.index({0}).item<double>();
    const double sigma_min =
        singular_values.index({column_count - 1}).item<double>();
    const double rank_tolerance =
        static_cast<double>(std::max(row_count, column_count)) *
        epsilon * sigma_max;
    const int64_t numerical_rank =
        (singular_values > rank_tolerance).sum().item<int64_t>();
    const double condition_number = sigma_min > 0.0
        ? sigma_max / sigma_min
        : std::numeric_limits<double>::infinity();
    const double caution_condition = 1.0 / std::sqrt(epsilon);
    const double unresolved_condition = 1.0 / epsilon;

    std::cout
        << "[Jacobian spectrum] rows=" << row_count
        << " cols=" << column_count
        << " dtype=" << source_scalar_type
        << " numerical_rank=" << numerical_rank << "/" << column_count
        << " rank_tolerance=" << rank_tolerance
        << " sigma_max=" << sigma_max
        << " sigma_min=" << sigma_min
        << " condition=" << condition_number
        << " condition_times_epsilon=" << condition_number * epsilon
        << std::endl;
    std::cout
        << "[Jacobian reference] caution_condition>="
        << caution_condition
        << " numerically_unresolved_condition>="
        << unresolved_condition
        << " full_rank_requires_sigma_min>"
        << rank_tolerance
        << std::endl;
    std::cout
        << "[Jacobian singular values] "
        << singular_values.cpu()
        << std::endl;
}

void print_jacobian_spectrum_from_matrix(const torch::Tensor& jacobian)
{
    const auto singular_values = torch::linalg_svdvals(
        jacobian.to(torch::kFloat64)
    );
    print_jacobian_spectrum(
        singular_values,
        jacobian.size(0),
        jacobian.size(1),
        jacobian.scalar_type()
    );
}

void print_jacobian_spectrum_from_gram(
    const torch::Tensor& jacobian_gram,
    const int64_t row_count,
    const torch::ScalarType source_scalar_type)
{
    const auto eigenvalues = torch::linalg_eigvalsh(
        jacobian_gram.to(torch::kFloat64)
    );
    const auto singular_values = eigenvalues
        .clamp_min(0.0)
        .sqrt()
        .flip({0})
        .contiguous();
    print_jacobian_spectrum(
        singular_values,
        row_count,
        jacobian_gram.size(0),
        source_scalar_type
    );
}

} // namespace

RFMSolver::RFMSolver(
    const Config& config, const std::shared_ptr<Equation> &eq,
    const torch::Device device, const uint64_t seed)
        : RFMSolver(config, eq, device, seed, config.solver_config.use_linear_solver)
{
}

RFMSolver::RFMSolver(
    Config config, const std::shared_ptr<Equation> &eq,
    const torch::Device device, const uint64_t seed, const bool is_linear)
        : is_linear_(is_linear),
          config_(std::move(config)),
          equation_(eq),
          seed_(seed),
          device_(device),
          rff_(RandomFeatureFunction(
                config_.eqn_config.dimension,
                config_.solver_config.hidden_dim,
                config_.eqn_config.total_time,
                device_,
                seed_,
                config_.solver_config.dtype,
                config_.solver_config.random_feature)),
          lambda_(config_.solver_config.initial_lambda)
{
    TORCH_CHECK(equation_ != nullptr, "equation must not be null");
    TORCH_CHECK(
        !is_linear || equation_->is_linear(),
        "Linear solver was requested, but equation '",
        config_.eqn_config.equation_name,
        "' is not marked as linear"
    );
    TORCH_CHECK(
        !equation_->is_linear() || equation_->has_coefficient(),
        "Equation '",
        config_.eqn_config.equation_name,
        "' is marked as linear, but L/M/N coefficients are not defined"
    );
    const auto& nonlinear_options = config_.solver_config.nonlinear;
    TORCH_CHECK(
        nonlinear_options.consistency_points >= 0,
        "nonlinear consistency_points must be nonnegative"
    );
    for (const double fraction :
         nonlinear_options.consistency_time_fractions)
    {
        TORCH_CHECK(
            std::isfinite(fraction) && 0.0 < fraction && fraction < 1.0,
            "nonlinear consistency_time_fractions must satisfy "
            "0 < fraction < 1, got ",
            fraction
        );
    }
    TORCH_CHECK(
        nonlinear_options.consistency_weight >= 0.0,
        "nonlinear consistency_weight must be nonnegative"
    );
    TORCH_CHECK(
        nonlinear_options.column_scale_epsilon > 0.0,
        "nonlinear column_scale_epsilon must be positive"
    );
    is_linear_ = is_linear && equation_->is_linear();

    torch::manual_seed(seed_);
    std::srand(static_cast<unsigned>(seed_));

    if (is_linear_)
    {
        compute_time_grid();
    }
    else
    {
        if (config_.solver_config.nonlinear.step_solver != "batched_qr" &&
            config_.solver_config.nonlinear.step_solver != "constant")
        {
            compute_txw();
        }

        const auto H = rff_.hidden_dim();

        y0_ = torch::randn({1}, torch::TensorOptions()
            .dtype(equation_->dtype())
            .device(device_));

        beta_ = torch::randn({H}, torch::TensorOptions()
            .dtype(equation_->dtype())
            .device(device_)) * config_.solver_config.beta_init_scale;
    }

}

/* Options
 * set the initial $y_0$, $\beta$, and $\lambda$. */

RFMSolver& RFMSolver::options(
    const std::optional<torch::Tensor>& y0,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> lambda
)
{
    if (y0.has_value())
    {
        y0_ = y0.value().to(
            torch::TensorOptions()
                .dtype(equation_->dtype())
                .device(device_)
        ).clone().detach();
    }

    if (beta.has_value())
    {
        beta_ = beta.value().to(
            torch::TensorOptions()
                .dtype(equation_->dtype())
                .device(device_)
        ).reshape({-1}).clone().detach();
    }

    if (lambda.has_value())
    {
        TORCH_CHECK(lambda.value() > 0.0, "lambda must be positive");
        lambda_ = lambda.value();
    }

    return *this;
}

RFMSolver& RFMSolver::linear_options(const LinearSolverOptions& options)
{
    auto resolved_options = options;
    if (resolved_options.solver_type == LinearSolverType::BatchedQR ||
        resolved_options.solver_type == LinearSolverType::Constant)
    {
        resolved_options.qr_batch_size = resolve_batch_size(
            resolved_options.qr_batch_size,
            rff_.hidden_dim(),
            "linear"
        );
    }
    linear_solver_options_ = resolved_options;
    if (is_linear_ &&
        (linear_solver_options_.solver_type == LinearSolverType::BatchedQR ||
         linear_solver_options_.solver_type == LinearSolverType::Constant))
    {
        clear_full_linear_cache();
    }
    else if (is_linear_)
    {
        prepare_full_linear_cache();
    }
    return *this;
}

/* Solver
 * `Solver` directs linear and nonlinear problems to different main solver functions. */

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve(const bool output_log) const
{
    auto result = is_linear_ ? solve_linear() : solve_nonlinear(output_log);
    diagnostics_.beta_norm = std::get<1>(result).norm().item<double>();
    return result;
}

double RFMSolver::test(const torch::Tensor& y0, const torch::Tensor& beta) const
{
    torch::NoGradGuard no_grad;

    const int64_t S = config_.solver_config.test_sample_size;
    TORCH_CHECK(S > 0, "test_sample_size must be positive");

    int64_t batch_size = config_.solver_config.test_batch_size;
    TORCH_CHECK(batch_size >= 0, "test_batch_size must be nonnegative");
    if (batch_size == 0)
    {
        batch_size = S;
    }
    batch_size = std::min(batch_size, S);

    // Keep evaluation paths independent of solver-specific random-number use.
    torch::manual_seed(splitmix64(seed_ ^ 0xD1B54A32D192ED03ULL));

    double squared_error_sum = 0.0;
    double terminal_sum = 0.0;
    double terminal_square_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
    {
        const int64_t current_batch_size = std::min(batch_size, S - row_begin);
        const auto [
            batch_squared_error,
            batch_terminal_sum,
            batch_terminal_square_sum,
            batch_count
        ] = test_batch(
            y0,
            beta,
            current_batch_size
        );
        squared_error_sum += batch_squared_error;
        terminal_sum += batch_terminal_sum;
        terminal_square_sum += batch_terminal_square_sum;
        residual_count += batch_count;
    }

    const double mean_square_error =
        squared_error_sum / static_cast<double>(residual_count);
    const double terminal_mean =
        terminal_sum / static_cast<double>(residual_count);
    const double terminal_variance = std::max(
        0.0,
        terminal_square_sum / static_cast<double>(residual_count) -
            terminal_mean * terminal_mean
    );
    const double terminal_std = std::sqrt(terminal_variance);
    const double test_rmse = std::sqrt(mean_square_error);
    diagnostics_.test_terminal_std = terminal_std;
    diagnostics_.normalized_test_rmse = terminal_std > 0.0
        ? test_rmse / terminal_std
        : std::numeric_limits<double>::infinity();
    diagnostics_.explained_terminal_variance = terminal_variance > 0.0
        ? 1.0 - mean_square_error / terminal_variance
        : -std::numeric_limits<double>::infinity();

    return test_rmse;
}

InternalPathEvaluation RFMSolver::evaluate_internal_paths(
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const torch::Tensor& dw_sample,
    const torch::Tensor& x_sample,
    const std::vector<int64_t>& time_indices
) const
{
    using namespace torch::indexing;
    torch::NoGradGuard no_grad;

    const int64_t time_count = config_.eqn_config.num_time_intervals;
    const int64_t dimension = equation_->dim();
    const int64_t hidden_dim = rff_.hidden_dim();
    TORCH_CHECK(!time_indices.empty(), "time_indices must not be empty");
    TORCH_CHECK(
        dw_sample.dim() == 3 &&
        dw_sample.size(1) == dimension &&
        dw_sample.size(2) == time_count,
        "dw_sample must have shape (B, ", dimension, ", ", time_count,
        "), but got ", dw_sample.sizes()
    );
    TORCH_CHECK(
        x_sample.dim() == 3 &&
        x_sample.size(0) == dw_sample.size(0) &&
        x_sample.size(1) == dimension &&
        x_sample.size(2) == time_count + 1,
        "x_sample must have shape (B, ", dimension, ", ", time_count + 1,
        "), but got ", x_sample.sizes()
    );
    for (size_t i = 0; i < time_indices.size(); ++i)
    {
        TORCH_CHECK(
            0 < time_indices[i] && time_indices[i] < time_count,
            "internal time index must satisfy 0 < index < ", time_count,
            ", got ", time_indices[i]
        );
        TORCH_CHECK(
            i == 0 || time_indices[i - 1] < time_indices[i],
            "time_indices must be strictly increasing"
        );
    }
    TORCH_CHECK(
        y0.numel() == 1,
        "y0 must have one element, got ", y0.numel()
    );
    TORCH_CHECK(
        beta.numel() == hidden_dim,
        "beta must have ", hidden_dim, " elements, got ", beta.numel()
    );

    const int64_t batch_size = dw_sample.size(0);
    const auto evaluation_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    const auto dw = dw_sample.to(evaluation_options).contiguous();
    const auto x_all = x_sample.to(evaluation_options)
        .permute({0, 2, 1})
        .contiguous(); // (B, T + 1, D)
    const auto x = x_all.index({Slice(), Slice(0, time_count), Slice()})
        .unsqueeze(2)
        .contiguous(); // (B, T, 1, D)
    const auto x_initial = x.index({
        Slice(),
        Slice(0, 1),
        Slice(),
        Slice()
    }).contiguous();
    const auto tensor_options = evaluation_options;
    const auto t_full = torch::linspace(
        0.0,
        config_.eqn_config.total_time,
        time_count + 1,
        tensor_options
    );
    const auto t = t_full.slice(0, 0, time_count)
        .reshape({1, time_count, 1, 1})
        .expand({batch_size, time_count, 1, 1})
        .contiguous();
    const auto beta_eval = beta.to(evaluation_options)
        .reshape({hidden_dim})
        .contiguous();
    auto y = y0.to(evaluation_options)
        .reshape({1, 1, 1, 1})
        .expand({batch_size, 1, 1, 1})
        .contiguous();
    const auto z_all = compute_z(t, x, beta_eval);
    const auto dw_all = dw.permute({0, 2, 1})
        .unsqueeze(2)
        .contiguous();

    std::vector<torch::Tensor> time_blocks;
    std::vector<torch::Tensor> state_blocks;
    std::vector<torch::Tensor> propagated_blocks;
    time_blocks.reserve(time_indices.size());
    state_blocks.reserve(time_indices.size());
    propagated_blocks.reserve(time_indices.size());
    size_t index_cursor = 0;

    for (int64_t k = 0; k < time_count; ++k)
    {
        const auto t_k = t.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto f_k = equation_->f(t_k, x_k, y, z_k);
        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(),
            ", but got ", f_k.sizes()
        );
        y = y - equation_->delta_t() * f_k +
            torch::sum(dw_k * z_k, -1, true);

        const int64_t next_index = k + 1;
        if (index_cursor < time_indices.size() &&
            time_indices[index_cursor] == next_index)
        {
            time_blocks.push_back(
                t.index({
                    Slice(),
                    Slice(next_index, next_index + 1),
                    Slice(),
                    Slice()
                })
            );
            state_blocks.push_back(
                x.index({
                    Slice(),
                    Slice(next_index, next_index + 1),
                    Slice(),
                    Slice()
                })
            );
            propagated_blocks.push_back(y);
            ++index_cursor;
        }
    }
    TORCH_CHECK(
        index_cursor == time_indices.size(),
        "not all requested internal time indices were evaluated"
    );

    const auto evaluation_t = torch::cat(time_blocks, 1).contiguous();
    const auto evaluation_x = torch::cat(state_blocks, 1).contiguous();
    const auto propagated_value =
        torch::cat(propagated_blocks, 1).contiguous();
    const auto direct_value = compute_direct_nonlinear_value(
        evaluation_t,
        evaluation_x,
        x_initial,
        y0.to(evaluation_options).reshape({1}),
        beta_eval
    );
    TORCH_CHECK(
        direct_value.sizes() == propagated_value.sizes(),
        "direct and propagated values must have the same shape, got ",
        direct_value.sizes(), " and ", propagated_value.sizes()
    );

    return {
        evaluation_t.detach(),
        evaluation_x.detach(),
        direct_value.detach(),
        propagated_value.detach()
    };
}

std::tuple<double, double, double, int64_t> RFMSolver::test_batch(
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const int64_t batch_size
) const
{
    using namespace torch::indexing;

    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t Hdim = rff_.hidden_dim();
    const double dt = equation_->delta_t();

    TORCH_CHECK(beta.numel() == Hdim, "beta must have ", Hdim, " elements, got ", beta.numel());

    const auto evaluation_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    const auto y0_eval = y0.to(evaluation_options).reshape({1});
    const auto beta_eval =
        beta.to(evaluation_options).reshape({Hdim}).contiguous();

    const auto [dw_sample, x_sample] = equation_->sample(batch_size);
    const auto dw_eval = dw_sample.to(evaluation_options).contiguous();
    const auto x_all = x_sample.to(evaluation_options)
        .permute({0, 2, 1})
        .contiguous();

    const auto x_eval = x_all.index({
        Slice(),
        Slice(0, -1),
        Slice()
    }).unsqueeze(2).contiguous(); // (S, T, 1, D)

    const auto x_end_eval = x_all.index({
        Slice(),
        Slice(-1, None),
        Slice()
    }).unsqueeze(2).contiguous(); // (S, 1, 1, D)

    const auto opts = evaluation_options;
    const auto t_full = torch::linspace(0, config_.eqn_config.total_time, T + 1, opts);
    const auto t = t_full.slice(0, 0, T)
        .reshape({1, T, 1, 1})
        .expand({batch_size, T, 1, 1})
        .contiguous();
    const auto t_end = t_full.slice(0, T, T + 1)
        .reshape({1, 1, 1, 1})
        .expand({batch_size, 1, 1, 1})
        .contiguous();

    auto y = y0_eval.reshape({1, 1, 1, 1}).expand({batch_size, 1, 1, 1});
    const auto z_all = compute_z(t, x_eval, beta_eval); // (S, T, 1, D)
    const auto dw_all = dw_eval.permute({0, 2, 1}).unsqueeze(2).contiguous(); // (S, T, 1, D)

    for (int64_t k = 0; k < T; ++k)
    {
        const auto t_k = t.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x_eval.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto f_k = equation_->f(t_k, x_k, y, z_k);
        const auto martingale = torch::sum(dw_k * z_k, -1, true);

        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(), ", but got ", f_k.sizes()
        );

        y = y - dt * f_k + martingale;
    }

    const auto g_terminal = equation_->g(t_end, x_end_eval);
    TORCH_CHECK(
        g_terminal.sizes() == y.sizes(),
        "equation_->g must return shape ", y.sizes(), ", but got ", g_terminal.sizes()
    );

    const auto residual = y - g_terminal;
    return {
        residual.pow(2).sum().item<double>(),
        g_terminal.sum().item<double>(),
        g_terminal.pow(2).sum().item<double>(),
        residual.numel()
    };
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_linear() const
{
    auto options = linear_solver_options_;
    options.ridge_lambda = config_.solver_config.initial_lambda;

    if (options.solver_type == LinearSolverType::Constant)
    {
        return solve_linear_constant_baseline(options);
    }

    if (options.solver_type == LinearSolverType::BatchedQR)
    {
        return solve_linear_batched_qr(options);
    }

    const auto [A, B] = compute_linear_coef();

    const auto [y0, beta, rmse] = solve_linear_least_squares(
        A,
        B,
        config_.solver_config.hidden_dim,
        options
    );

    return {y0, beta, rmse};
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_nonlinear(const bool output_log) const
{
    TORCH_CHECK(y0_.defined(), "y0_ is not initialized");
    TORCH_CHECK(beta_.defined(), "beta_ is not initialized");
    TORCH_CHECK(lambda_ > 0.0, "lambda_ must be positive");

    if (config_.solver_config.nonlinear.step_solver == "constant")
    {
        return solve_nonlinear_constant_baseline(y0_, lambda_, output_log);
    }

    return solve_nonlinear_levenberg_marquardt(y0_, beta_, lambda_, output_log);
}

/* Utils
 * including calculating intermediate quantities, building the solver, checking tensor status, etc. */

std::pair<const torch::Tensor, const torch::Tensor> RFMSolver::compute_linear_coef() const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const double dt = equation_->delta_t();

    TORCH_CHECK(L_.defined() && M_.defined() && N_.defined() &&
        dw_.defined() && x_.defined() && x_end_.defined(),
        "full linear cache is not initialized");

    const auto L = L_.squeeze(-1).squeeze(-1).contiguous();   // (S, T)
    const auto M = M_.squeeze(2).contiguous();                // (S, T, D)
    const auto N = N_.squeeze(-1).squeeze(-1).contiguous();   // (S, T)
    const auto dW = dw_.permute({0, 2, 1}).contiguous();      // (S, T, D)

    const auto a  = 1.0 - dt * L;       // (S, T)
    const auto xi = dW - dt * M;        // (S, T, D)
    const auto c  = dt * N;             // (S, T)

    const auto suffix_inclusive = torch::flip(
        torch::cumprod(torch::flip(a, {1}), 1),
        {1}
    ); // (S, T), suffix_inclusive[:, k] = prod_{j=k}^{T-1} a_j

    auto weights = torch::ones_like(a); // (S, T)
    if (T > 1)
    {
        weights.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, T - 1)},
            suffix_inclusive.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(1, torch::indexing::None)
            })
        );
    }

    auto coef_y0 = a.prod(1, true); // (S, 1)
    const auto weighted_xi = xi * weights.unsqueeze(-1);
    const auto coef_beta = contract_z_features(
        t_,
        x_,
        weighted_xi
    ); // (S, H)

    const auto A = torch::cat({coef_y0, coef_beta}, 1).contiguous(); // (S, 1 + H)
    const auto constant_part = (weights * c).sum(1, true); // (S, 1)
    const auto g_XN = equation_->g(t_end_, x_end_).reshape({S, 1}).to(device_);
    const auto B = g_XN - constant_part; // (S, 1)

    TORCH_CHECK(
        A.device().type() == device_.type() &&
        B.device().type() == device_.type(),
        "A, B must be on ", device_.type(), ", but got ", A.device().type(), " & ", B.device().type());

    return {A, B};
}

std::pair<const torch::Tensor, const torch::Tensor> RFMSolver::compute_linear_coef_batch(
    const int64_t row_begin,
    const int64_t row_end
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const double dt = equation_->delta_t();
    const int64_t batch_size = row_end - row_begin;

    TORCH_CHECK(0 <= row_begin && row_begin < row_end && row_end <= S,
        "invalid linear coefficient row range [", row_begin, ", ", row_end, ") for S=", S);

    const auto [dw_sample, x_sample] = equation_->sample(batch_size);
    const auto tensor_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    const auto dw_batch = dw_sample.to(tensor_options).contiguous();
    const auto x_all = x_sample.to(tensor_options)
        .permute({0, 2, 1})
        .contiguous(); // (B, T+1, D)
    const auto x = x_all.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(0, -1),
        torch::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (B, T, 1, D)
    const auto x_end = x_all.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(-1, torch::indexing::None),
        torch::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (B, 1, 1, D)

    const auto t = t_.index({
        torch::indexing::Slice(0, batch_size),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice()
    }).contiguous();
    const auto t_end = t_end_.index({
        torch::indexing::Slice(0, batch_size),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice()
    }).contiguous();

    const auto L = equation_->coef().L(t, x).squeeze(-1).squeeze(-1).contiguous(); // (B, T)
    const auto M = equation_->coef().M(t, x).squeeze(2).contiguous();              // (B, T, D)
    const auto N = equation_->coef().N(t, x).squeeze(-1).squeeze(-1).contiguous(); // (B, T)
    const auto dW = dw_batch.permute({0, 2, 1}).contiguous();                     // (B, T, D)

    // 线性递推中的三块
    const auto a  = 1.0 - dt * L;       // (S, T)
    const auto xi = dW - dt * M;        // (S, T, D)
    const auto c  = dt * N;             // (S, T)

    // weights[k] = prod_{j=k+1}^{T-1} a_j
    const auto suffix_inclusive = torch::flip(
        torch::cumprod(torch::flip(a, {1}), 1),
        {1}
    ); // (S, T), suffix_inclusive[:, k] = prod_{j=k}^{T-1} a_j

    auto weights = torch::ones_like(a); // (S, T)
    if (T > 1)
    {
        weights.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, T - 1)},
            suffix_inclusive.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(1, torch::indexing::None)
            })
        );
    }

    // 矩阵第一块: y0 系数
    auto coef_y0 = a.prod(1, true); // (B, 1)

    // Matrix block for beta.
    const auto weighted_xi = xi * weights.unsqueeze(-1);
    const auto coef_beta = contract_z_features(t, x, weighted_xi); // (B, H)

    // 拼接设计矩阵
    const auto A = torch::cat({coef_y0, coef_beta}, 1).contiguous(); // (B, 1 + H)

    // 右端项
    const auto constant_part = (weights * c).sum(1, true); // (B, 1)
    const auto g_XN = equation_->g(t_end, x_end).reshape({batch_size, 1}).to(device_);

    const auto B = g_XN - constant_part; // (B, 1)

    TORCH_CHECK(
        A.device().type() == device_.type() &&
        B.device().type() == device_.type(),
        "A, B must be on ", device_.type(), ", but got ", A.device().type(), " & ", B.device().type())

    return {A, B};
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_linear_batched_qr(
    const LinearSolverOptions& options
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t Hdim = config_.solver_config.hidden_dim;
    const int64_t parameter_count = 1 + Hdim;

    TORCH_CHECK(options.qr_batch_size > 0, "qr_batch_size must be positive");

    bool initialized = false;
    int64_t pending_rows = 0;
    std::vector<torch::Tensor> pending_A;
    std::vector<torch::Tensor> pending_B;
    torch::Tensor R;
    torch::Tensor rhs;

    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);

        if (!initialized)
        {
            pending_A.push_back(A_batch);
            pending_B.push_back(B_batch);
            pending_rows += A_batch.size(0);

            if (pending_rows < parameter_count)
            {
                continue;
            }

            const auto A_init = torch::cat(pending_A, 0).contiguous();
            const auto B_init = torch::cat(pending_B, 0).contiguous();
            std::tie(R, rhs) = reduce_linear_qr_libtorch(
                A_init,
                B_init
            );
            rhs = rhs.reshape({parameter_count, 1}).contiguous();
            initialized = true;
            pending_A.clear();
            pending_B.clear();
            continue;
        }

        const auto stacked_A = torch::cat({R, A_batch}, 0).contiguous();
        const auto stacked_B = torch::cat({rhs, B_batch}, 0).contiguous();
        std::tie(R, rhs) = reduce_linear_qr_libtorch(
            stacked_A,
            stacked_B
        );
        rhs = rhs.reshape({parameter_count, 1}).contiguous();
    }

    TORCH_CHECK(initialized, "sample_size must be at least 1 + hidden_dim for batched QR");

    const auto x = torch::linalg_solve_triangular(R, rhs, true).reshape({parameter_count, 1});

    double squared_error_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);
        const auto residual = torch::matmul(A_batch, x) - B_batch;
        squared_error_sum += residual.pow(2).sum().item<double>();
        residual_count += residual.numel();
    }

    const auto rmse =
        std::sqrt(squared_error_sum / static_cast<double>(residual_count));
    return split_linear_solution(x, Hdim, rmse);
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_linear_constant_baseline(
    const LinearSolverOptions& options
) const
{
    TORCH_CHECK(options.qr_batch_size > 0, "constant baseline batch size must be positive");

    const int64_t S = config_.solver_config.sample_size;
    const int64_t Hdim = rff_.hidden_dim();

    double numerator = 0.0;
    double denominator = 0.0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);
        const auto A0 = A_batch.index({
            torch::indexing::Slice(),
            0
        }).reshape({-1, 1}).contiguous();
        numerator += (A0 * B_batch).sum().item<double>();
        denominator += (A0 * A0).sum().item<double>();
    }
    TORCH_CHECK(denominator > 0.0, "constant baseline has zero denominator");

    const auto y0 = torch::full(
        {1},
        numerator / denominator,
        torch::TensorOptions().dtype(equation_->dtype()).device(device_)
    );
    const auto beta = torch::zeros({Hdim}, y0.options());

    double squared_error_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);
        const auto A0 = A_batch.index({
            torch::indexing::Slice(),
            0
        }).reshape({-1, 1}).contiguous();
        const auto residual = A0 * y0.reshape({1, 1}) - B_batch;
        squared_error_sum += residual.pow(2).sum().item<double>();
        residual_count += residual.numel();
    }

    const auto rmse =
        std::sqrt(squared_error_sum / static_cast<double>(residual_count));
    return {y0.detach().clone(), beta, rmse};
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_nonlinear_constant_baseline(
    const torch::Tensor& y0,
    const double lambda,
    const bool output_log
) const
{
    TORCH_CHECK(lambda > 0.0, "lambda must be positive");

    const auto& nonlinear_options = config_.solver_config.nonlinear;
    const int64_t S = config_.solver_config.sample_size;
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t batch_size = resolve_batch_size(
        nonlinear_options.batch_size,
        Hdim,
        "nonlinear"
    );

    auto y0_value = y0.to(device_).reshape({1}).item<double>();
    double damping = lambda;
    double final_error = 0.0;
    int64_t accepted_iterations = 0;
    const auto beta_zero = torch::zeros(
        {Hdim},
        torch::TensorOptions().dtype(equation_->dtype()).device(device_)
    );

    auto make_theta = [&beta_zero, this](const double value)
    {
        const auto y0_tensor = torch::full(
            {1},
            value,
            torch::TensorOptions().dtype(equation_->dtype()).device(device_)
        );
        return solver_utils::pack_nonlinear_parameters(y0_tensor, beta_zero)
            .detach()
            .contiguous();
    };

    auto evaluate = [this, S, batch_size](const torch::Tensor& theta)
    {
        double squared_error_sum = 0.0;
        double jacobian_square_sum = 0.0;
        double jacobian_residual_sum = 0.0;
        int64_t residual_count = 0;

        for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
        {
            const int64_t row_end = std::min(row_begin + batch_size, S);
            const auto [residual_batch, jacobian_batch] =
                compute_nonlinear_objective_residual_and_jacobian_batch(theta, row_begin, row_end);
            const auto j0 = jacobian_batch.index({
                torch::indexing::Slice(),
                0
            }).reshape({-1, 1}).contiguous();
            const auto residual = residual_batch.reshape({-1, 1}).contiguous();
            squared_error_sum += residual.pow(2).sum().item<double>();
            jacobian_square_sum += j0.pow(2).sum().item<double>();
            jacobian_residual_sum += (j0 * residual).sum().item<double>();
            residual_count += residual.numel();
        }

        return std::tuple{
            squared_error_sum,
            jacobian_square_sum,
            jacobian_residual_sum,
            residual_count
        };
    };

    for (int64_t iter = 0; iter < config_.solver_config.num_iterations; ++iter)
    {
        const auto theta = make_theta(y0_value);
        const auto [
            squared_error_sum,
            jacobian_square_sum,
            jacobian_residual_sum,
            residual_count
        ] = evaluate(theta);
        const auto curr_loss = 0.5 * squared_error_sum;
        const auto curr_error =
            std::sqrt(squared_error_sum / static_cast<double>(residual_count));

        bool accepted = false;
        double accepted_y0 = y0_value;
        double accepted_error = curr_error;
        double accepted_step_norm = 0.0;

        for (int64_t retry = 0; retry <= nonlinear_options.max_retries; ++retry)
        {
            const double denominator = jacobian_square_sum + damping;
            TORCH_CHECK(denominator > 0.0, "constant baseline LM denominator must be positive");
            const auto delta = -jacobian_residual_sum / denominator;
            const double trial_y0 = y0_value + delta;
            const auto trial_theta = make_theta(trial_y0);
            const auto [trial_loss, trial_error] = compute_nonlinear_loss_error_batched(
                trial_theta,
                batch_size
            );
            const bool trial_accepted = trial_loss < curr_loss;

            if (output_log)
            {
                std::cout
                    << "[LM constant] iter=" << iter
                    << " retry=" << retry
                    << " loss=" << curr_loss
                    << " error=" << curr_error
                    << " trial_error=" << trial_error
                    << " lambda=" << damping
                    << " step_norm=" << std::abs(delta)
                    << " accepted=" << std::boolalpha << trial_accepted
                    << " y_0=" << trial_y0
                    << std::noboolalpha
                    << std::endl;
            }

            if (trial_accepted)
            {
                accepted = true;
                accepted_y0 = trial_y0;
                accepted_error = trial_error;
                accepted_step_norm = std::abs(delta);
                break;
            }

            if (retry < nonlinear_options.max_retries)
            {
                damping = std::min(nonlinear_options.max_lambda, damping * nonlinear_options.lambda_increase);
            }
        }

        if (!accepted)
        {
            break;
        }

        y0_value = accepted_y0;
        ++accepted_iterations;
        final_error = accepted_error;
        damping = std::max(nonlinear_options.min_lambda, damping * nonlinear_options.lambda_decrease);

        if (final_error <= nonlinear_options.error_tol || accepted_step_norm <= nonlinear_options.step_tol)
        {
            break;
        }
    }

    const auto final_y0 = torch::full(
        {1},
        y0_value,
        torch::TensorOptions().dtype(equation_->dtype()).device(device_)
    );
    const auto final_theta = make_theta(y0_value);
    std::tie(std::ignore, final_error) = compute_nonlinear_loss_error_batched(
        final_theta,
        batch_size
    );
    const auto [
        final_squared_error_sum,
        final_jacobian_square_sum,
        final_jacobian_residual_sum,
        final_residual_count
    ] = evaluate(final_theta);
    (void)final_squared_error_sum;
    (void)final_jacobian_square_sum;
    diagnostics_.objective_rmse = final_error;
    diagnostics_.accepted_lm_iterations = accepted_iterations;
    diagnostics_.final_damping = damping;
    diagnostics_.final_gradient_inf_norm =
        std::abs(final_jacobian_residual_sum) /
        static_cast<double>(final_residual_count);
    return {final_y0.detach().clone(), beta_zero.detach().clone(), final_error};
}

std::tuple<torch::Tensor, torch::Tensor, double> RFMSolver::solve_nonlinear_levenberg_marquardt(
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const double lambda,
    const bool output_log) const
{
    const int64_t max_iters = config_.solver_config.num_iterations;
    const auto& nonlinear_options = config_.solver_config.nonlinear;

    const auto tensor_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    torch::Tensor theta = solver_utils::pack_nonlinear_parameters(y0, beta)
        .detach()
        .clone()
        .to(tensor_options);
    double damping = lambda;
    double final_error = 0.0;
    int64_t accepted_iterations = 0;
    const int64_t nonlinear_batch_size = resolve_batch_size(
        config_.solver_config.nonlinear.batch_size,
        rff_.hidden_dim(),
        "nonlinear"
    );

    for (int64_t iter = 0; iter < max_iters; ++iter)
    {
        torch::Tensor residual;
        torch::Tensor jacobian;
        torch::Tensor column_scales;
        double curr_loss = 0.0;
        double curr_error = 0.0;

        if (nonlinear_options.step_solver != "batched_qr")
        {
            const auto [residual_raw, jacobian_raw] =
                compute_nonlinear_objective_residual_and_jacobian(theta);
            residual = residual_raw.reshape({-1});
            jacobian = jacobian_raw;
            if (output_log)
            {
                print_jacobian_spectrum_from_matrix(jacobian);
            }
            column_scales = nonlinear_options.scale_jacobian_columns
                ? solver_utils::jacobian_column_scales(
                    jacobian,
                    nonlinear_options.column_scale_epsilon
                )
                : torch::ones({jacobian.size(1)}, jacobian.options());
            curr_loss = 0.5 * residual.pow(2).sum().item<double>();
            curr_error = std::sqrt(residual.pow(2).mean().item<double>());
        }

        bool accepted = false;
        torch::Tensor accepted_theta;
        double accepted_error = 0.0;
        double accepted_step_norm = 0.0;

        for (int64_t retry = 0;
             retry <= nonlinear_options.max_retries;
             ++retry)
        {
            torch::Tensor delta;
            if (nonlinear_options.step_solver == "qr")
            {
                const auto scaled_delta = solver_utils::solve_lm_step_qr(
                    jacobian / column_scales,
                    residual,
                    damping
                );
                delta = scaled_delta / column_scales;
            }
            else if (nonlinear_options.step_solver == "batched_qr")
            {
                std::tie(delta, curr_loss, curr_error) = solve_nonlinear_lm_step_batched_qr(
                    theta,
                    damping,
                    nonlinear_batch_size,
                    output_log && retry == 0
                );
            }
            else
            {
                TORCH_CHECK(
                    nonlinear_options.step_solver == "normal",
                    "unknown nonlinear step_solver: ",
                    nonlinear_options.step_solver
                );
                const auto scaled_delta = solver_utils::solve_lm_step(
                    jacobian / column_scales,
                    residual,
                    damping
                );
                delta = scaled_delta / column_scales;
            }
            const auto step_norm = delta.norm().item<double>();

            const auto trial_theta = (theta + delta).detach();
            double trial_loss = 0.0;
            double trial_error = 0.0;
            if (nonlinear_options.step_solver == "batched_qr")
            {
                std::tie(trial_loss, trial_error) = compute_nonlinear_loss_error_batched(
                    trial_theta,
                    nonlinear_batch_size
                );
            }
            else
            {
                const auto trial_residual = compute_nonlinear_objective_residual(trial_theta).reshape({-1});
                trial_loss =
                    0.5 * trial_residual.pow(2).sum().item<double>();
                trial_error = std::sqrt(
                    trial_residual.pow(2).mean().item<double>()
                );
            }
            accepted = trial_loss < curr_loss;

            if (output_log)
            {
                std::cout
                    << "[LM] iter=" << iter
                    << " retry=" << retry
                    << " loss=" << curr_loss
                    << " error=" << curr_error
                    << " trial_error=" << trial_error
                    << " lambda=" << damping
                    << " step_norm=" << step_norm
                    << " accepted=" << std::boolalpha << accepted
                    << " y_0=" << trial_theta.index({0}).item<double>()
                    << std::noboolalpha
                    << std::endl;
            }

            if (accepted)
            {
                accepted_theta = trial_theta;
                accepted_error = trial_error;
                accepted_step_norm = step_norm;
                break;
            }

            if (retry < nonlinear_options.max_retries)
            {
                damping = std::min(
                    nonlinear_options.max_lambda,
                    damping * nonlinear_options.lambda_increase
                );
            }
        }

        if (!accepted)
        {
            if (output_log)
            {
                std::cout
                    << "[LM] stop: no accepted step after "
                    << nonlinear_options.max_retries + 1
                    << " attempts at iter=" << iter
                    << std::endl;
            }
            break;
        }

        theta = accepted_theta;
        ++accepted_iterations;
        damping = std::max(
            nonlinear_options.min_lambda,
            damping * nonlinear_options.lambda_decrease
        );
        final_error = accepted_error;

        if (final_error <= nonlinear_options.error_tol ||
            accepted_step_norm <= nonlinear_options.step_tol)
        {
            break;
        }
    }

    const int64_t Hdim = rff_.hidden_dim();

    const auto final_y0 = theta.index({0}).reshape({1});
    const auto final_beta = theta.index({
        torch::indexing::Slice(1, torch::indexing::None)
    }).reshape({Hdim}).contiguous();
    double terminal_error = 0.0;
    if (nonlinear_options.step_solver == "batched_qr")
    {
        std::tie(std::ignore, final_error) = compute_nonlinear_loss_error_batched(
            theta,
            nonlinear_batch_size
        );
        terminal_error = compute_nonlinear_terminal_error_batched(
            theta,
            nonlinear_batch_size
        );
    }
    else
    {
        const auto final_residual = compute_nonlinear_objective_residual(theta).reshape({-1});
        final_error =
            std::sqrt(final_residual.pow(2).mean().item<double>());
        const auto terminal_residual = compute_residual_for_samples(
            theta,
            t_,
            t_end_,
            x_,
            x_end_,
            dw_,
            false
        );
        terminal_error = std::sqrt(
            terminal_residual.pow(2).mean().item<double>()
        );
    }
    diagnostics_.objective_rmse = final_error;
    diagnostics_.accepted_lm_iterations = accepted_iterations;
    diagnostics_.final_damping = damping;
    diagnostics_.final_gradient_inf_norm =
        compute_nonlinear_gradient_inf_norm(theta, nonlinear_batch_size);

    return {
        final_y0.detach().clone(),
        final_beta.detach().clone(),
        terminal_error
    };
}

torch::Tensor RFMSolver::compute_nonlinear_objective_residual(
    const torch::Tensor& theta
) const
{
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + Hdim;

    TORCH_CHECK(
        theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes()
    );

    return compute_residual_for_samples(
        theta,
        t_,
        t_end_,
        x_,
        x_end_,
        dw_,
        true
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RFMSolver::sample_nonlinear_batch(
    const int64_t row_begin,
    const int64_t row_end
) const
{
    using namespace torch::indexing;

    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t batch_size = row_end - row_begin;

    TORCH_CHECK(0 <= row_begin && row_begin < row_end && row_end <= S,
        "invalid nonlinear sample row range [", row_begin, ", ", row_end, ") for S=", S);

    const uint64_t batch_seed = splitmix64(seed_ + static_cast<uint64_t>(row_begin));
    torch::manual_seed(batch_seed);

    const auto [dw_sample, x_sample] = equation_->sample(batch_size);
    const auto tensor_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    const auto dw = dw_sample.to(tensor_options).contiguous();
    const auto x_all = x_sample.to(tensor_options)
        .permute({0, 2, 1})
        .contiguous();
    const auto x = x_all.index({Slice(), Slice(0, -1), Slice()})
        .unsqueeze(2)
        .contiguous();
    const auto x_end = x_all.index({Slice(), Slice(-1, None), Slice()})
        .unsqueeze(2)
        .contiguous();

    const auto opts = tensor_options;
    const auto t_full = torch::linspace(0, config_.eqn_config.total_time, T + 1, opts);
    const auto t = t_full.slice(0, 0, T)
        .reshape({1, T, 1, 1})
        .expand({batch_size, T, 1, 1})
        .contiguous();
    const auto t_end = t_full.slice(0, T, T + 1)
        .reshape({1, 1, 1, 1})
        .expand({batch_size, 1, 1, 1})
        .contiguous();

    check_tx_shape(t, x);
    return {t, t_end, x, x_end, dw};
}

torch::Tensor RFMSolver::compute_nonlinear_objective_residual_batch(
    const torch::Tensor& theta,
    const int64_t row_begin,
    const int64_t row_end
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + Hdim;

    TORCH_CHECK(theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes());
    TORCH_CHECK(0 <= row_begin && row_begin < row_end && row_end <= S,
        "invalid nonlinear residual row range [", row_begin, ", ", row_end, ") for S=", S);

    const auto [t, t_end, x, x_end, dw] = sample_nonlinear_batch(row_begin, row_end);
    return compute_residual_for_samples(
        theta,
        t,
        t_end,
        x,
        x_end,
        dw,
        true
    );
}

torch::Tensor RFMSolver::compute_residual_for_samples(
    const torch::Tensor& theta,
    const torch::Tensor& t,
    const torch::Tensor& t_end,
    const torch::Tensor& x,
    const torch::Tensor& x_end,
    const torch::Tensor& dw,
    const bool objective
) const
{
    using namespace torch::indexing;

    const int64_t batch_size = x.size(0);
    const int64_t time_count = x.size(1);
    const int64_t hidden_dim = rff_.hidden_dim();
    const double dt = equation_->delta_t();
    const auto& nonlinear_options = config_.solver_config.nonlinear;

    const auto y0 = theta.index({0}).reshape({1});
    const auto beta = theta.index({Slice(1, None)})
        .reshape({hidden_dim})
        .contiguous();
    const auto x_initial = x.index({
        Slice(),
        Slice(0, 1),
        Slice(),
        Slice()
    }).contiguous();
    auto y = y0.reshape({1, 1, 1, 1})
        .expand({batch_size, 1, 1, 1})
        .contiguous();
    const auto z_all = compute_z(t, x, beta);
    const auto dw_all = dw.permute({0, 2, 1})
        .unsqueeze(2)
        .contiguous();
    const auto checkpoints =
        objective ? nonlinear_consistency_indices(time_count)
                  : std::vector<int64_t>{};
    std::vector<torch::Tensor> consistency_residuals;
    consistency_residuals.reserve(checkpoints.size());
    size_t checkpoint_cursor = 0;

    for (int64_t k = 0; k < time_count; ++k)
    {
        const auto t_k = t.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto f_k = equation_->f(t_k, x_k, y, z_k);
        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(),
            ", but got ", f_k.sizes()
        );
        y = y - dt * f_k + torch::sum(dw_k * z_k, -1, true);

        const int64_t next_time_index = k + 1;
        if (checkpoint_cursor < checkpoints.size() &&
            checkpoints[checkpoint_cursor] == next_time_index)
        {
            const auto t_checkpoint = t.index({
                Slice(),
                Slice(next_time_index, next_time_index + 1),
                Slice(),
                Slice()
            });
            const auto x_checkpoint = x.index({
                Slice(),
                Slice(next_time_index, next_time_index + 1),
                Slice(),
                Slice()
            });
            const auto direct_y = compute_direct_nonlinear_value(
                t_checkpoint,
                x_checkpoint,
                x_initial,
                y0,
                beta
            );
            consistency_residuals.push_back(
                (y - direct_y).reshape({batch_size, 1}).contiguous()
            );
            ++checkpoint_cursor;
        }
    }

    const auto g_terminal = equation_->g(t_end, x_end);
    TORCH_CHECK(
        g_terminal.sizes() == y.sizes(),
        "equation_->g must return shape ", y.sizes(),
        ", but got ", g_terminal.sizes()
    );
    const auto terminal_residual =
        (y - g_terminal).reshape({batch_size, 1}).contiguous();
    if (!objective)
    {
        return terminal_residual;
    }

    const auto residual_scale = nonlinear_residual_scale();

    std::vector<torch::Tensor> residual_blocks;
    residual_blocks.reserve(1 + consistency_residuals.size());
    residual_blocks.push_back(terminal_residual / residual_scale);
    if (!consistency_residuals.empty())
    {
        const double consistency_scale = std::sqrt(
            nonlinear_options.consistency_weight /
            static_cast<double>(consistency_residuals.size())
        );
        for (const auto& consistency_residual : consistency_residuals)
        {
            residual_blocks.push_back(
                consistency_scale * consistency_residual / residual_scale
            );
        }
    }
    return torch::cat(residual_blocks, 0).contiguous();
}

std::pair<torch::Tensor, torch::Tensor> RFMSolver::compute_nonlinear_objective_residual_and_jacobian_batch(
    const torch::Tensor& theta,
    const int64_t row_begin,
    const int64_t row_end
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + Hdim;

    TORCH_CHECK(theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes());
    TORCH_CHECK(0 <= row_begin && row_begin < row_end && row_end <= S,
        "invalid nonlinear Jacobian row range [", row_begin, ", ", row_end, ") for S=", S);

    const auto [t, t_end, x, x_end, dw] = sample_nonlinear_batch(row_begin, row_end);
    return compute_objective_residual_and_jacobian_for_samples(
        theta,
        t,
        t_end,
        x,
        x_end,
        dw
    );
}

std::pair<torch::Tensor, torch::Tensor> RFMSolver::compute_nonlinear_objective_residual_and_jacobian(
    const torch::Tensor& theta
) const
{
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + Hdim;

    TORCH_CHECK(
        theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes()
    );

    return compute_objective_residual_and_jacobian_for_samples(
        theta,
        t_,
        t_end_,
        x_,
        x_end_,
        dw_
    );
}

std::pair<torch::Tensor, torch::Tensor>
RFMSolver::compute_objective_residual_and_jacobian_for_samples(
    const torch::Tensor& theta,
    const torch::Tensor& t,
    const torch::Tensor& t_end,
    const torch::Tensor& x,
    const torch::Tensor& x_end,
    const torch::Tensor& dw
) const
{
    using namespace torch::indexing;

    const int64_t batch_size = x.size(0);
    const int64_t time_count = x.size(1);
    const int64_t hidden_dim = rff_.hidden_dim();
    const int64_t expected_size = 1 + hidden_dim;
    const double dt = equation_->delta_t();
    const auto& nonlinear_options = config_.solver_config.nonlinear;

    TORCH_CHECK(
        theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes()
    );

    const auto y0 = theta.index({0}).reshape({1});
    const auto beta = theta.index({Slice(1, None)})
        .reshape({hidden_dim})
        .contiguous();
    const auto x_initial = x.index({
        Slice(),
        Slice(0, 1),
        Slice(),
        Slice()
    }).contiguous();
    auto y = y0.reshape({1, 1, 1, 1})
        .expand({batch_size, 1, 1, 1})
        .contiguous();
    auto sensitivity_y0 = torch::ones_like(y);
    auto sensitivity_beta = torch::zeros(
        {batch_size, 1, hidden_dim},
        theta.options()
    );
    const auto dw_all = dw.permute({0, 2, 1})
        .unsqueeze(2)
        .contiguous(); // (B, T, 1, D)
    const auto checkpoints = nonlinear_consistency_indices(time_count);
    std::vector<torch::Tensor> consistency_residuals;
    std::vector<torch::Tensor> consistency_jacobians;
    consistency_residuals.reserve(checkpoints.size());
    consistency_jacobians.reserve(checkpoints.size());
    size_t checkpoint_cursor = 0;

    for (int64_t k = 0; k < time_count; ++k)
    {
        const auto t_k = t.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto z_features =
            compute_nonlinear_z_features(t_k, x_k); // (B, 1, H, D)
        auto z_k = (
            z_features * beta.reshape({1, 1, hidden_dim, 1})
        ).sum(2, true); // (B, 1, 1, D)
        if (use_hard_terminal_lift())
        {
            const auto terminal_t = torch::full_like(
                t_k,
                config_.eqn_config.total_time
            );
            const auto terminal_gradient =
                equation_->terminal_gradient(terminal_t, x_k);
            z_k = z_k + equation_->gradient_to_z(
                t_k,
                x_k,
                terminal_gradient
            );
        }
        TORCH_CHECK(
            z_features.dim() == 4 &&
            z_features.size(0) == batch_size &&
            z_features.size(1) == 1 &&
            z_features.size(2) == hidden_dim &&
            z_features.size(3) == equation_->dim(),
            "nonlinear z features have wrong shape: ", z_features.sizes()
        );

        auto y_local = y.detach().requires_grad_(true);
        auto z_local = z_k.detach().requires_grad_(true);
        const auto f_k = equation_->f(t_k, x_k, y_local, z_local);
        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(),
            ", but got ", f_k.sizes()
        );

        auto f_y = torch::zeros_like(y);
        auto f_z = torch::zeros_like(z_k);
        if (f_k.requires_grad())
        {
            const auto gradients = torch::autograd::grad(
                {f_k},
                {y_local, z_local},
                {torch::ones_like(f_k)},
                false,
                false,
                true
            );
            if (gradients[0].defined())
            {
                f_y = gradients[0];
            }
            if (gradients[1].defined())
            {
                f_z = gradients[1];
            }
        }

        // dY_{k+1} = (1 - dt*f_y)dY_k + (dW_k - dt*f_z)dZ_k.
        const auto scale = 1.0 - dt * f_y;
        const auto z_sensitivity = (
            (dw_k - dt * f_z) * z_features
        ).sum(-1); // (B, 1, H)
        sensitivity_y0 = scale * sensitivity_y0;
        sensitivity_beta =
            scale.reshape({batch_size, 1, 1}) * sensitivity_beta +
            z_sensitivity;

        const auto martingale = torch::sum(dw_k * z_k, -1, true);
        y = (y - dt * f_k + martingale).detach();

        const int64_t next_time_index = k + 1;
        if (checkpoint_cursor < checkpoints.size() &&
            checkpoints[checkpoint_cursor] == next_time_index)
        {
            const auto t_checkpoint = t.index({
                Slice(),
                Slice(next_time_index, next_time_index + 1),
                Slice(),
                Slice()
            });
            const auto x_checkpoint = x.index({
                Slice(),
                Slice(next_time_index, next_time_index + 1),
                Slice(),
                Slice()
            });
            const auto direct_y = compute_direct_nonlinear_value(
                t_checkpoint,
                x_checkpoint,
                x_initial,
                y0,
                beta
            );
            const auto phi_checkpoint =
                rff_.phi(t_checkpoint, x_checkpoint).squeeze(-1);
            const auto t_initial = torch::zeros_like(t_checkpoint);
            const auto phi_initial =
                rff_.phi(t_initial, x_initial).squeeze(-1);
            auto direct_y0_sensitivity = torch::ones_like(y);
            auto direct_beta_sensitivity =
                phi_checkpoint - phi_initial;
            if (use_hard_terminal_lift())
            {
                const auto remaining_time =
                    config_.eqn_config.total_time - t_checkpoint;
                direct_y0_sensitivity =
                    remaining_time / config_.eqn_config.total_time;
                direct_beta_sensitivity =
                    remaining_time.reshape({batch_size, 1, 1}) *
                    direct_beta_sensitivity;
            }

            consistency_residuals.push_back(
                (y - direct_y).reshape({batch_size, 1}).contiguous()
            );
            consistency_jacobians.push_back(torch::cat({
                (sensitivity_y0 - direct_y0_sensitivity)
                    .reshape({batch_size, 1}),
                (sensitivity_beta - direct_beta_sensitivity)
                    .reshape({batch_size, hidden_dim})
            }, 1).contiguous());
            ++checkpoint_cursor;
        }
    }

    const auto g_terminal = equation_->g(t_end, x_end);
    TORCH_CHECK(
        g_terminal.sizes() == y.sizes(),
        "equation_->g must return shape ", y.sizes(),
        ", but got ", g_terminal.sizes()
    );

    const auto terminal_residual = (y - g_terminal)
        .reshape({batch_size, 1})
        .contiguous();
    const auto terminal_jacobian = torch::cat({
        sensitivity_y0.reshape({batch_size, 1}),
        sensitivity_beta.reshape({batch_size, hidden_dim})
    }, 1).contiguous();

    const auto residual_scale = nonlinear_residual_scale();

    std::vector<torch::Tensor> residual_blocks{
        terminal_residual / residual_scale
    };
    std::vector<torch::Tensor> jacobian_blocks{
        terminal_jacobian / residual_scale
    };
    if (!consistency_residuals.empty())
    {
        const double consistency_scale = std::sqrt(
            nonlinear_options.consistency_weight /
            static_cast<double>(consistency_residuals.size())
        );
        for (size_t i = 0; i < consistency_residuals.size(); ++i)
        {
            residual_blocks.push_back(
                consistency_scale * consistency_residuals[i] / residual_scale
            );
            jacobian_blocks.push_back(
                consistency_scale * consistency_jacobians[i] / residual_scale
            );
        }
    }

    return {
        torch::cat(residual_blocks, 0).contiguous(),
        torch::cat(jacobian_blocks, 0).contiguous()
    };
}

std::tuple<torch::Tensor, double, double> RFMSolver::solve_nonlinear_lm_step_batched_qr(
    const torch::Tensor& theta,
    const double lambda,
    const int64_t batch_size,
    const bool output_spectrum
) const
{
    TORCH_CHECK(lambda > 0.0, "lambda must be positive");
    TORCH_CHECK(batch_size > 0, "nonlinear batch_size must be positive");

    const int64_t S = config_.solver_config.sample_size;
    const int64_t num_param = theta.numel();
    const auto opts = torch::TensorOptions().dtype(theta.dtype()).device(theta.device());
    const auto& nonlinear_options = config_.solver_config.nonlinear;

    auto column_scales = torch::ones({num_param}, opts);
    auto jacobian_gram = torch::zeros(
        {num_param, num_param},
        opts.dtype(torch::kFloat64)
    );
    int64_t spectrum_row_count = 0;
    bool spectrum_collected = false;
    if (nonlinear_options.scale_jacobian_columns)
    {
        auto column_square_sum = torch::zeros({num_param}, opts);
        int64_t jacobian_row_count = 0;
        for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
        {
            const int64_t row_end = std::min(row_begin + batch_size, S);
            const auto [residual_batch, jacobian_batch] =
                compute_nonlinear_objective_residual_and_jacobian_batch(
                    theta,
                    row_begin,
                    row_end
                );
            column_square_sum += jacobian_batch.square().sum(0);
            jacobian_row_count += jacobian_batch.size(0);
            if (output_spectrum)
            {
                const auto jacobian_double =
                    jacobian_batch.to(torch::kFloat64);
                jacobian_gram += torch::matmul(
                    jacobian_double.transpose(0, 1),
                    jacobian_double
                );
                spectrum_row_count += jacobian_batch.size(0);
            }
        }
        spectrum_collected = output_spectrum;
        column_scales = (
            column_square_sum / static_cast<double>(jacobian_row_count)
        ).sqrt().clamp_min(
            nonlinear_options.column_scale_epsilon
        ).contiguous();
    }

    auto R = std::sqrt(lambda) * torch::eye(num_param, opts);
    auto rhs = torch::zeros({num_param, 1}, opts);
    double squared_error_sum = 0.0;
    int64_t residual_count = 0;

    for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
    {
        const int64_t row_end = std::min(row_begin + batch_size, S);
        const auto [residual_batch, jacobian_batch] =
            compute_nonlinear_objective_residual_and_jacobian_batch(theta, row_begin, row_end);
        if (output_spectrum && !spectrum_collected)
        {
            const auto jacobian_double =
                jacobian_batch.to(torch::kFloat64);
            jacobian_gram += torch::matmul(
                jacobian_double.transpose(0, 1),
                jacobian_double
            );
            spectrum_row_count += jacobian_batch.size(0);
        }
        const auto reduction_rhs = -residual_batch.reshape({-1, 1}).contiguous();
        std::tie(R, rhs) = reduce_linear_qr_libtorch(
            torch::cat({R, jacobian_batch / column_scales}, 0).contiguous(),
            torch::cat({rhs, reduction_rhs}, 0).contiguous()
        );

        squared_error_sum += residual_batch.pow(2).sum().item<double>();
        residual_count += residual_batch.numel();
    }

    if (output_spectrum)
    {
        print_jacobian_spectrum_from_gram(
            jacobian_gram,
            spectrum_row_count,
            theta.scalar_type()
        );
    }

    const auto scaled_delta =
        torch::linalg_solve_triangular(R, rhs, true)
            .reshape({-1})
            .contiguous();
    const auto delta = (scaled_delta / column_scales).contiguous();
    const auto loss = 0.5 * squared_error_sum;
    const auto error =
        std::sqrt(squared_error_sum / static_cast<double>(residual_count));
    return {delta, loss, error};
}

std::pair<double, double> RFMSolver::compute_nonlinear_loss_error_batched(
    const torch::Tensor& theta,
    const int64_t batch_size
) const
{
    TORCH_CHECK(batch_size > 0, "nonlinear batch_size must be positive");

    const int64_t S = config_.solver_config.sample_size;
    double squared_error_sum = 0.0;
    int64_t residual_count = 0;

    for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
    {
        const int64_t row_end = std::min(row_begin + batch_size, S);
        const auto residual_batch = compute_nonlinear_objective_residual_batch(theta, row_begin, row_end);
        squared_error_sum += residual_batch.pow(2).sum().item<double>();
        residual_count += residual_batch.numel();
    }

    return {
        0.5 * squared_error_sum,
        std::sqrt(squared_error_sum / static_cast<double>(residual_count))
    };
}

double RFMSolver::compute_nonlinear_terminal_error_batched(
    const torch::Tensor& theta,
    const int64_t batch_size
) const
{
    TORCH_CHECK(batch_size > 0, "nonlinear batch_size must be positive");

    const int64_t sample_size = config_.solver_config.sample_size;
    double squared_error_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < sample_size; row_begin += batch_size)
    {
        const int64_t row_end =
            std::min(row_begin + batch_size, sample_size);
        const auto [t, t_end, x, x_end, dw] =
            sample_nonlinear_batch(row_begin, row_end);
        const auto residual = compute_residual_for_samples(
            theta,
            t,
            t_end,
            x,
            x_end,
            dw,
            false
        );
        squared_error_sum += residual.square().sum().item<double>();
        residual_count += residual.numel();
    }
    return std::sqrt(
        squared_error_sum / static_cast<double>(residual_count)
    );
}

double RFMSolver::compute_nonlinear_gradient_inf_norm(
    const torch::Tensor& theta,
    const int64_t batch_size
) const
{
    torch::Tensor gradient_sum = torch::zeros_like(theta);
    int64_t residual_count = 0;
    if (config_.solver_config.nonlinear.step_solver == "batched_qr" ||
        config_.solver_config.nonlinear.step_solver == "constant")
    {
        const int64_t sample_size = config_.solver_config.sample_size;
        for (int64_t row_begin = 0;
             row_begin < sample_size;
             row_begin += batch_size)
        {
            const int64_t row_end =
                std::min(row_begin + batch_size, sample_size);
            const auto [residual, jacobian] =
                compute_nonlinear_objective_residual_and_jacobian_batch(
                    theta,
                    row_begin,
                    row_end
                );
            gradient_sum += torch::matmul(
                jacobian.transpose(0, 1),
                residual.reshape({-1, 1})
            ).reshape({-1});
            residual_count += residual.numel();
        }
    }
    else
    {
        const auto [residual, jacobian] =
            compute_nonlinear_objective_residual_and_jacobian(theta);
        gradient_sum = torch::matmul(
            jacobian.transpose(0, 1),
            residual.reshape({-1, 1})
        ).reshape({-1});
        residual_count = residual.numel();
    }
    return (
        gradient_sum.abs().max() / static_cast<double>(residual_count)
    ).item<double>();
}

torch::Tensor RFMSolver::forward_nonlinear_terminal_y(
    const torch::Tensor& y0,
    const torch::Tensor& beta
) const
{
    using namespace torch::indexing;

    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const double dt = equation_->delta_t();

    auto y = y0.reshape({1, 1, 1, 1}).expand({S, 1, 1, 1});
    const auto z_all = compute_z(t_, x_, beta);
    const auto dw_all = dw_.permute({0, 2, 1}).unsqueeze(2).contiguous(); // (S, T, 1, D)

    for (int64_t k = 0; k < T; ++k)
    {
        const auto t_k = t_.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x_.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto f_k = equation_->f(t_k, x_k, y, z_k);
        const auto martingale = torch::sum(dw_k * z_k, -1, true);

        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(), ", but got ", f_k.sizes()
        );

        y = y - dt * f_k + martingale;
    }

    return y.contiguous();
}

torch::Tensor RFMSolver::compute_z(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const torch::Tensor& beta
) const
{
    auto spatial_gradient = rff_.spatial_gradient(t, x, beta);
    if (use_hard_terminal_lift())
    {
        const auto terminal_t = torch::full_like(
            t,
            config_.eqn_config.total_time
        );
        const auto remaining_time =
            config_.eqn_config.total_time - t;
        spatial_gradient =
            equation_->terminal_gradient(terminal_t, x) +
            remaining_time * spatial_gradient;
    }
    const auto z = equation_->gradient_to_z(t, x, spatial_gradient);
    TORCH_CHECK(
        z.sizes() == spatial_gradient.sizes(),
        "gradient_to_z must preserve spatial-gradient shape ",
        spatial_gradient.sizes(), ", but got ", z.sizes()
    );
    return z.contiguous();
}

torch::Tensor RFMSolver::compute_nonlinear_z_features(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    auto gradient_features = rff_.spatial_gradient_features(t, x);
    if (use_hard_terminal_lift())
    {
        gradient_features =
            (config_.eqn_config.total_time - t) *
            gradient_features;
    }
    const auto z_features = equation_->gradient_to_z(
        t,
        x,
        gradient_features
    );
    TORCH_CHECK(
        z_features.sizes() == gradient_features.sizes(),
        "gradient_to_z must preserve feature-gradient shape ",
        gradient_features.sizes(), ", but got ", z_features.sizes()
    );
    return z_features.contiguous();
}

torch::Tensor RFMSolver::compute_direct_nonlinear_value(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const torch::Tensor& x_initial,
    const torch::Tensor& y0,
    const torch::Tensor& beta
) const
{
    const int64_t batch_size = x.size(0);
    const int64_t time_count = x.size(1);
    const auto phi_value = rff_.phi(t, x).squeeze(-1);
    const auto t_initial = torch::zeros(
        {batch_size, 1, 1, 1},
        t.options()
    );
    const auto phi_initial = rff_.phi(t_initial, x_initial)
        .squeeze(-1)
        .expand({batch_size, time_count, rff_.hidden_dim()});
    const auto centered_features = phi_value - phi_initial;
    const auto feature_value = torch::matmul(
        centered_features,
        beta.reshape({rff_.hidden_dim()})
    ).unsqueeze(2).unsqueeze(3);

    if (!use_hard_terminal_lift())
    {
        return (
            y0.reshape({1, 1, 1, 1}) + feature_value
        ).contiguous();
    }

    const auto terminal_t = torch::full_like(
        t,
        config_.eqn_config.total_time
    );
    const auto terminal_t_initial = torch::full_like(
        t_initial,
        config_.eqn_config.total_time
    );
    const auto g_value = equation_->g(terminal_t, x);
    const auto g_initial = equation_->g(terminal_t_initial, x_initial);
    const auto remaining_time =
        config_.eqn_config.total_time - t;
    return (
        g_value +
        remaining_time * (
            (y0.reshape({1, 1, 1, 1}) - g_initial) /
                config_.eqn_config.total_time +
            feature_value
        )
    ).contiguous();
}

bool RFMSolver::use_hard_terminal_lift() const
{
    return !is_linear_ &&
        config_.solver_config.nonlinear.step_solver != "constant" &&
        config_.solver_config.nonlinear.hard_terminal_lift;
}

std::vector<int64_t> RFMSolver::nonlinear_consistency_indices(
    const int64_t time_count
) const
{
    const auto& options = config_.solver_config.nonlinear;
    if (options.step_solver == "constant" ||
        options.consistency_weight == 0.0 ||
        time_count <= 1)
    {
        return {};
    }

    if (!options.consistency_time_fractions.empty())
    {
        std::vector<int64_t> indices;
        indices.reserve(options.consistency_time_fractions.size());
        for (const double fraction :
             options.consistency_time_fractions)
        {
            indices.push_back(std::clamp(
                static_cast<int64_t>(std::llround(
                    fraction * static_cast<double>(time_count)
                )),
                int64_t{1},
                time_count - 1
            ));
        }
        std::sort(indices.begin(), indices.end());
        indices.erase(
            std::unique(indices.begin(), indices.end()),
            indices.end()
        );
        return indices;
    }

    if (options.consistency_points == 0)
    {
        return {};
    }

    const int64_t point_count = std::min(
        options.consistency_points,
        time_count - 1
    );
    std::vector<int64_t> indices;
    indices.reserve(static_cast<size_t>(point_count));
    for (int64_t i = 1; i <= point_count; ++i)
    {
        const int64_t index = std::clamp(
            i * time_count / (point_count + 1),
            int64_t{1},
            time_count - 1
        );
        if (indices.empty() || indices.back() != index)
        {
            indices.push_back(index);
        }
    }
    return indices;
}

torch::Tensor RFMSolver::nonlinear_residual_scale() const
{
    const auto& options = config_.solver_config.nonlinear;
    const auto tensor_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    if (options.step_solver == "constant" || !options.normalize_residuals)
    {
        return torch::ones({}, tensor_options);
    }
    if (nonlinear_residual_scale_.defined())
    {
        return nonlinear_residual_scale_;
    }

    auto terminal_sum = torch::zeros({}, tensor_options);
    auto terminal_square_sum = torch::zeros({}, tensor_options);
    int64_t terminal_count = 0;
    if (options.step_solver == "batched_qr")
    {
        const int64_t sample_size = config_.solver_config.sample_size;
        const int64_t batch_size = resolve_batch_size(
            options.batch_size,
            rff_.hidden_dim(),
            "nonlinear"
        );
        for (int64_t row_begin = 0;
             row_begin < sample_size;
             row_begin += batch_size)
        {
            const int64_t row_end =
                std::min(row_begin + batch_size, sample_size);
            const auto [t, t_end, x, x_end, dw] =
                sample_nonlinear_batch(row_begin, row_end);
            const auto terminal_value = equation_->g(t_end, x_end);
            terminal_sum += terminal_value.sum();
            terminal_square_sum += terminal_value.square().sum();
            terminal_count += terminal_value.numel();
        }
    }
    else
    {
        const auto terminal_value = equation_->g(t_end_, x_end_);
        terminal_sum = terminal_value.sum();
        terminal_square_sum = terminal_value.square().sum();
        terminal_count = terminal_value.numel();
    }

    TORCH_CHECK(terminal_count > 0, "terminal residual scale has no samples");
    const auto terminal_mean =
        terminal_sum / static_cast<double>(terminal_count);
    const auto terminal_variance = (
        terminal_square_sum / static_cast<double>(terminal_count) -
        terminal_mean.square()
    ).clamp_min(0.0);
    nonlinear_residual_scale_ =
        terminal_variance.sqrt().clamp_min(1.0e-6).contiguous();
    return nonlinear_residual_scale_;
}

torch::Tensor RFMSolver::contract_z_features(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const torch::Tensor& weights
) const
{
    using namespace torch::indexing;

    check_tx_shape(t, x);
    TORCH_CHECK(
        weights.dim() == 3 &&
        weights.size(0) == x.size(0) &&
        weights.size(1) == x.size(1) &&
        weights.size(2) == equation_->dim(),
        "weights must have shape (", x.size(0), ", ", x.size(1), ", ",
        equation_->dim(), "), but got ", weights.sizes()
    );

    constexpr int64_t feature_batch_size = 512;
    const int64_t sample_count = x.size(0);
    const int64_t time_count = x.size(1);
    const int64_t hidden_dim = rff_.hidden_dim();
    auto result = torch::zeros({sample_count, hidden_dim}, weights.options());

    for (int64_t row_begin = 0; row_begin < sample_count; row_begin += feature_batch_size)
    {
        const int64_t row_end = std::min(
            row_begin + feature_batch_size,
            sample_count
        );
        const int64_t current_batch_size = row_end - row_begin;
        auto result_batch = torch::zeros(
            {current_batch_size, hidden_dim},
            weights.options()
        );

        for (int64_t k = 0; k < time_count; ++k)
        {
            const auto t_k = t.size(0) == 1
                ? t.index({Slice(), Slice(k, k + 1), Slice(), Slice()})
                : t.index({
                    Slice(row_begin, row_end),
                    Slice(k, k + 1),
                    Slice(),
                    Slice()
                });
            const auto x_k = x.index({
                Slice(row_begin, row_end),
                Slice(k, k + 1),
                Slice(),
                Slice()
            });
            const auto gradient_features =
                rff_.spatial_gradient_features(t_k, x_k); // (B, 1, H, D)
            const auto z_features = equation_->gradient_to_z(
                t_k,
                x_k,
                gradient_features
            );
            TORCH_CHECK(
                z_features.sizes() == gradient_features.sizes(),
                "gradient_to_z must preserve feature-gradient shape ",
                gradient_features.sizes(), ", but got ", z_features.sizes()
            );
            const auto weights_k = weights.index({
                Slice(row_begin, row_end),
                Slice(k, k + 1),
                Slice()
            }).unsqueeze(2); // (B, 1, 1, D)
            result_batch = result_batch +
                (weights_k * z_features).sum(3).sum(1);
        }

        result.index_put_({Slice(row_begin, row_end)}, result_batch);
    }

    return result.contiguous();
}

void RFMSolver::compute_time_grid()
{
    const double total_time = config_.eqn_config.total_time;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t S = config_.solver_config.sample_size;

    const auto opts =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    const auto t_full = torch::linspace(0, total_time, T + 1, opts);

    const auto t_base = t_full.slice(0, 0, T).reshape({1, T, 1, 1});
    t_ = t_base.expand({S, T, 1, 1}).contiguous();

    const auto t_end_base = t_full.slice(0, T, T + 1).reshape({1, 1, 1, 1});
    t_end_ = t_end_base.expand({S, 1, 1, 1}).contiguous();
}

void RFMSolver::compute_txw()
{
    compute_time_grid();

    const int64_t S = config_.solver_config.sample_size;

    const auto [fst, snd] = equation_->sample(S);

    const auto tensor_options =
        torch::TensorOptions().dtype(equation_->dtype()).device(device_);
    dw_ = fst.to(tensor_options).contiguous();

    const auto x_all = snd.to(tensor_options)
        .permute({0, 2, 1})
        .contiguous(); // (S, T+1, D)
    x_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(0, -1),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (S, T, 1, D)

    x_end_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(-1, at::indexing::None),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (S, 1, 1, D)

    check_tx_shape(t_, x_);
}

void RFMSolver::prepare_full_linear_cache()
{
    TORCH_CHECK(is_linear_, "full linear cache is only available for linear equations");
    if (L_.defined())
    {
        return;
    }

    compute_txw();
    compute_L(t_, x_);
    compute_M(t_, x_);
    compute_N(t_, x_);
}

void RFMSolver::clear_full_linear_cache()
{
    dw_ = torch::Tensor();
    x_ = torch::Tensor();
    x_end_ = torch::Tensor();
    L_ = torch::Tensor();
    M_ = torch::Tensor();
    N_ = torch::Tensor();

    if (!t_.defined() || !t_end_.defined())
    {
        compute_time_grid();
    }
}

void RFMSolver::compute_L(const torch::Tensor &t, const torch::Tensor &x)
{
    check_tx_shape(t, x);

    const auto result = equation_->coef().L(t, x);

    TORCH_CHECK(
        result.dim() == 4 &&
        result.size(0) == x.size(0) &&
        result.size(1) == x.size(1) &&
        result.size(2) == 1 &&
        result.size(3) == 1,
        "Invalid shape for L(t, x). Expected (",
        x.size(0), ", ",
        x.size(1), ", 1, 1), but got ",
        result.sizes()
    );

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_L must be on ", device_.type(), ", but got ", result.device().type()
    );

    L_ = result;
}

void RFMSolver::compute_M(const torch::Tensor& t, const torch::Tensor& x)
{
    check_tx_shape(t, x);

    const torch::Tensor result = equation_->coef().M(t, x);

    TORCH_CHECK(
        result.sizes() == x.sizes(),
        "Invalid shape for M(t, x). Expected ",
        x.sizes(), ", but got ", result.sizes()
    );

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_M must be on ", device_.type(), ", but got ", result.device().type()
    );

    M_ = result;
}

void RFMSolver::compute_N(const torch::Tensor& t, const torch::Tensor& x)
{
    check_tx_shape(t, x);

    const torch::Tensor result = equation_->coef().N(t, x);

    TORCH_CHECK(
        result.dim() == 4 &&
        result.size(0) == x.size(0) &&
        result.size(1) == x.size(1) &&
        result.size(2) == 1 &&
        result.size(3) == 1,
        "Invalid shape for N(t, x). Expected (",
        x.size(0), ", ",
        x.size(1), ", 1, 1), but got ",
        result.sizes()
    );

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_N must be on ", device_.type(), ", but got ", result.device().type()
    );

    N_ = result;
}

void RFMSolver::check_tx_shape(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    // check t
    TORCH_CHECK(
        t.dim() == 4,
        "t must be a 4D tensor, got dim = ", t.dim()
        );

    TORCH_CHECK(
        t.dtype() == equation_->dtype(),
        "t must have dtype ", equation_->dtype(), ", but got ", t.dtype()
        );

    TORCH_CHECK(
        (t.size(0) == x.size(0) || t.size(0) == 1) &&
        t.size(1) == config_.eqn_config.num_time_intervals &&
        t.size(2) == 1 &&
        t.size(3) == 1,
        "Invalid shape for t. Expected (",
        x.size(0), " or 1, ",
        config_.eqn_config.num_time_intervals,
        ", 1, 1), but got ", t.sizes()
    );

    // check x
    TORCH_CHECK(
        x.dim() == 4,
        "x must be a 4D tensor, got dim = ", x.dim()
    );

    TORCH_CHECK(
        x.dtype() == equation_->dtype(),
        "x must have dtype ", equation_->dtype(), ", but got ", x.dtype()
        );

    TORCH_CHECK(
        x.size(0) > 0 &&
        x.size(1) == config_.eqn_config.num_time_intervals &&
        x.size(2) == 1 &&
        x.size(3) == config_.eqn_config.dimension,
        "Invalid shape for x. Expected (",
        "positive batch size, ",
        config_.eqn_config.num_time_intervals,
        ", 1, ", config_.eqn_config.dimension,
        "), but got ", x.sizes()
    );

    TORCH_CHECK(
        x.device().type() == device_.type() &&
        t.device().type() == device_.type(),
        "x, t must be on ", device_.type(), ", but got ", x.device().type(), " & ", t.device().type()
    );
}
