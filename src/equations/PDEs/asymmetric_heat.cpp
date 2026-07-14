#include "equation.h"
#include "register_equation.h"

#include <cmath>

struct AsymmetricHeatCoefficient final : Coefficient
{
    [[nodiscard]] torch::Tensor L(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return torch::zeros({x.size(0), x.size(1), 1, 1}, x.options());
    }

    [[nodiscard]] torch::Tensor M(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return torch::zeros_like(x);
    }

    [[nodiscard]] torch::Tensor N(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return torch::zeros({x.size(0), x.size(1), 1, 1}, x.options());
    }
};

class AsymmetricHeat final : public Equation
{
public:
    explicit AsymmetricHeat(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(eqn_config.params.value("x_init", 0.0f)),
          cos_weight_(eqn_config.params.value("cos_weight", 0.5f)),
          tanh_weight_(eqn_config.params.value("tanh_weight", 0.1f)),
          sin_frequency_(eqn_config.params.value("sin_frequency", 1.0f)),
          cos_frequency_(eqn_config.params.value("cos_frequency", 1.0f)),
          tanh_frequency_(eqn_config.params.value("tanh_frequency", 1.0f)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{20}))
    {
        TORCH_CHECK(direction_count_ > 0, "direction_count must be positive");

        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        const auto dim_idx = torch::arange(1, dim_ + 1, opts).reshape({1, dim_});
        const auto dir_idx = torch::arange(1, direction_count_ + 1, opts).reshape({direction_count_, 1});

        directions_ = torch::sin(12.9898f * dir_idx * dim_idx + 0.123f) +
            0.5f * torch::cos(78.233f * (dir_idx + 1.0f) * dim_idx + 0.456f);
        directions_ = directions_ / directions_.norm(2, 1, true);

        if (linear_)
        {
            coefficient_ = std::make_shared<AsymmetricHeatCoefficient>();
        }
    }

    [[nodiscard("Return Need to be Used")]]
    std::pair<torch::Tensor, torch::Tensor> sample(const int64_t num_sample) const override
    {
        const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        const auto dw = torch::randn(
            {num_sample, dim_, num_time_interval_},
            opts
        ) * sqrt_delta_t_;

        auto x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1},
            opts
        );
        x.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(), 0},
            torch::full({num_sample, dim_}, x_init_, opts)
        );

        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            using namespace torch::indexing;
            x.index_put_(
                {Slice(), Slice(), i + 1},
                x.index({Slice(), Slice(), i}) + dw.index({Slice(), Slice(), i})
            );
        }

        return {dw, x};
    }

    [[nodiscard("Return Need to be Used")]]
    torch::Tensor f(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& y,
        const torch::Tensor& z) const override
    {
        return torch::zeros_like(y);
    }

    [[nodiscard("Return Need to be Used")]]
    torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        TORCH_CHECK(x.dim() >= 4, "x must have at least 4 dimensions");

        const auto directions = directions_.to(x.device());
        const auto projected = torch::matmul(x.squeeze(2), directions.transpose(0, 1));
        const auto values = torch::sin(sin_frequency_ * projected) +
            cos_weight_ * torch::cos(cos_frequency_ * projected) +
            tanh_weight_ * torch::tanh(tanh_frequency_ * projected);

        return (values.sum(-1, true) / std::sqrt(static_cast<float>(direction_count_)))
            .unsqueeze(2);
    }

private:
    float x_init_;
    float cos_weight_;
    float tanh_weight_;
    float sin_frequency_;
    float cos_frequency_;
    float tanh_frequency_;
    int64_t direction_count_;
    torch::Tensor directions_;
};

REGISTER_EQUATION_CLASS(AsymmetricHeat)

extern "C" void force_link_AsymmetricHeat() {}
