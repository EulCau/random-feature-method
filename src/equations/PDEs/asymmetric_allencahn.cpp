#include "equation.h"
#include "register_equation.h"

#include <cmath>

class AsymmetricAllenCahn final : public Equation
{
public:
    explicit AsymmetricAllenCahn(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full(
              {dim_},
              eqn_config.params.value("x_init", 0.0),
              torch::TensorOptions().dtype(dtype_))),
          sigma_(eqn_config.params.value("sigma", std::sqrt(2.0))),
          lambda_(eqn_config.params.value("lambda", 1.0)),
          terminal_offset_(eqn_config.params.value("terminal_offset", 2.0)),
          terminal_scale_(eqn_config.params.value("terminal_scale", 0.4)),
          asymmetry_scale_(eqn_config.params.value("asymmetry_scale", 0.05)),
          asymmetry_frequency_(eqn_config.params.value("asymmetry_frequency", 1.0)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{20}))
    {
        TORCH_CHECK(direction_count_ > 0, "direction_count must be positive");
        const auto opts = torch::TensorOptions().dtype(dtype_);
        const auto dim_idx = torch::arange(1, dim_ + 1, opts).reshape({1, dim_});
        const auto dir_idx = torch::arange(1, direction_count_ + 1, opts).reshape({direction_count_, 1});
        directions_ = torch::sin(12.9898 * dir_idx * dim_idx + 0.123) +
            0.5 * torch::cos(78.233 * (dir_idx + 1.0) * dim_idx + 0.456);
        directions_ = directions_ / directions_.norm(2, 1, true);
    }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> sample(const int64_t num_sample) const override
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
        x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
            x_init_.expand({num_sample, dim_}));
        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), i + 1},
                x.index({torch::indexing::Slice(), torch::indexing::Slice(), i}) +
                    sigma_ * dw.index({torch::indexing::Slice(), torch::indexing::Slice(), i}));
        }
        return {dw, x};
    }

    [[nodiscard]] torch::Tensor f(
        const torch::Tensor& t, const torch::Tensor& x,
        const torch::Tensor& y, const torch::Tensor& z) const override
    {
        return lambda_ * (y - torch::pow(y, 3));
    }

    [[nodiscard]] torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        const auto radial = 1.0 /
            (terminal_offset_ + terminal_scale_ * torch::sum(x * x, -1, true));
        const auto directions = directions_.to(x.device());
        const auto projected = torch::matmul(x.squeeze(2), directions.transpose(0, 1));
        const auto asymmetric = asymmetry_scale_ *
            torch::tanh(asymmetry_frequency_ * projected).sum(-1, true) /
            std::sqrt(static_cast<double>(direction_count_));
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
    double terminal_offset_;
    double terminal_scale_;
    double asymmetry_scale_;
    double asymmetry_frequency_;
    int64_t direction_count_;
    torch::Tensor directions_;
};

REGISTER_EQUATION_CLASS(AsymmetricAllenCahn)

extern "C" void force_link_AsymmetricAllenCahn() {}
