#include "equation.h"
#include "register_equation.h"

#include <algorithm>
#include <cmath>

class HighFrequencyAsymmetricHJBLQ final : public Equation
{
public:
    explicit HighFrequencyAsymmetricHJBLQ(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full(
              {dim_},
              eqn_config.params.value("x_init", 0.0),
              torch::TensorOptions().dtype(dtype_))),
          sigma_(eqn_config.params.value("sigma", std::sqrt(2.0))),
          lambda_(eqn_config.params.value("lambda", 1.0)),
          asymmetry_scale_(eqn_config.params.value("asymmetry_scale", 0.1)),
          base_frequency_(eqn_config.params.value("base_frequency", 4.0)),
          frequency_bandwidth_(eqn_config.params.value("frequency_bandwidth", 4.0)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{20}))
    {
        TORCH_CHECK(base_frequency_ > 0.0, "base_frequency must be positive");
        TORCH_CHECK(frequency_bandwidth_ >= 0.0, "frequency_bandwidth must be nonnegative");
        TORCH_CHECK(direction_count_ > 0, "direction_count must be positive");

        const auto opts = torch::TensorOptions().dtype(dtype_);
        const auto dim_idx = torch::arange(1, dim_ + 1, opts).reshape({1, dim_});
        const auto dir_idx = torch::arange(1, direction_count_ + 1, opts)
            .reshape({direction_count_, 1});
        directions_ = torch::sin(41.713 * dir_idx * dim_idx + 0.271) +
            0.5 * torch::cos(17.519 * (dir_idx + 1.0) * dim_idx + 0.613);
        directions_ = directions_ / directions_.norm(2, 1, true);

        const auto direction_position = (dir_idx.squeeze(1) - 1.0) /
            static_cast<double>(
                std::max<int64_t>(direction_count_ - 1, 1));
        frequencies_ = base_frequency_ + frequency_bandwidth_ * direction_position;
        phase_shifts_ = 2.0 * std::acos(-1.0) *
            torch::frac(0.61803398875 * dir_idx.squeeze(1));
        direction_weights_ = 0.75 +
            0.5 * torch::frac(0.41421356237 * dir_idx.squeeze(1));
    }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> sample(
        const int64_t num_sample) const override
    {
        const auto opts = torch::TensorOptions().dtype(dtype_);
        auto dw = torch::randn(
            {num_sample, dim_, num_time_interval_},
            opts
        ) * sqrt_delta_t_;
        auto x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1},
            opts
        );
        x.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(), 0},
            x_init_.expand({num_sample, dim_}));

        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            x.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(), i + 1},
                x.index({torch::indexing::Slice(), torch::indexing::Slice(), i}) +
                    sigma_ * dw.index(
                        {torch::indexing::Slice(), torch::indexing::Slice(), i}));
        }
        return {dw, x};
    }

    [[nodiscard]] torch::Tensor f(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& y,
        const torch::Tensor& z) const override
    {
        return -0.5 * lambda_ * torch::sum(z * z, -1, true);
    }

    [[nodiscard]] torch::Tensor g(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        const auto radial = torch::log(
            (1.0 + torch::sum(x * x, -1, true)) / 2.0);
        const auto projected = torch::matmul(
            x.squeeze(2),
            directions_.to(x.device()).transpose(0, 1));
        const auto frequencies = frequencies_.to(x.device());
        const auto phase_shifts = phase_shifts_.to(x.device());
        const auto direction_weights = direction_weights_.to(x.device());
        const auto asymmetric = asymmetry_scale_ *
            (direction_weights *
                torch::sin(frequencies * projected + phase_shifts)).sum(-1, true) /
            direction_weights.norm();
        return radial + asymmetric.unsqueeze(2);
    }

    [[nodiscard]] torch::Tensor gradient_to_z(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& spatial_gradient) const override
    {
        return sigma_ * spatial_gradient;
    }

private:
    torch::Tensor x_init_;
    double sigma_;
    double lambda_;
    double asymmetry_scale_;
    double base_frequency_;
    double frequency_bandwidth_;
    int64_t direction_count_;
    torch::Tensor directions_;
    torch::Tensor frequencies_;
    torch::Tensor phase_shifts_;
    torch::Tensor direction_weights_;
};

REGISTER_EQUATION_CLASS(HighFrequencyAsymmetricHJBLQ)

extern "C" void force_link_HighFrequencyAsymmetricHJBLQ() {}
