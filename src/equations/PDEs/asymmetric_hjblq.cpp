#include "equation.h"
#include "register_equation.h"

#include <cmath>

class AsymmetricHJBLQ final : public Equation
{
public:
    explicit AsymmetricHJBLQ(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full({dim_}, eqn_config.params.value("x_init", 0.0f))),
          sigma_(eqn_config.params.value("sigma", static_cast<float>(std::sqrt(2.0)))),
          lambda_(eqn_config.params.value("lambda", 1.0f)),
          asymmetry_scale_(eqn_config.params.value("asymmetry_scale", 0.1f)),
          asymmetry_frequency_(eqn_config.params.value("asymmetry_frequency", 1.0f)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{20}))
    {
        TORCH_CHECK(direction_count_ > 0, "direction_count must be positive");
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        const auto dim_idx = torch::arange(1, dim_ + 1, opts).reshape({1, dim_});
        const auto dir_idx = torch::arange(1, direction_count_ + 1, opts).reshape({direction_count_, 1});
        directions_ = torch::sin(37.719f * dir_idx * dim_idx + 0.789f) +
            0.5f * torch::cos(11.131f * (dir_idx + 1.0f) * dim_idx);
        directions_ = directions_ / directions_.norm(2, 1, true);
    }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> sample(const int64_t num_sample) const override
    {
        auto dw = torch::randn({num_sample, dim_, num_time_interval_}, torch::kFloat) * sqrt_delta_t_;
        auto x = torch::zeros({num_sample, dim_, num_time_interval_ + 1}, torch::kFloat);
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
        return -0.5f * lambda_ * torch::sum(z * z, -1, true);
    }

    [[nodiscard]] torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        const auto radial = torch::log((1.0f + torch::sum(x * x, -1, true)) / 2.0f);
        const auto directions = directions_.to(x.device());
        const auto projected = torch::matmul(x.squeeze(2), directions.transpose(0, 1));
        const auto asymmetric = asymmetry_scale_ *
            torch::sin(asymmetry_frequency_ * projected).sum(-1, true) /
            std::sqrt(static_cast<float>(direction_count_));
        return radial + asymmetric.unsqueeze(2);
    }

    [[nodiscard]] torch::Tensor gradient_to_z(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& spatial_gradient) const override
    {
        return sigma_ * spatial_gradient;
    }

    [[nodiscard]] torch::Tensor terminal_gradient(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        const auto directions = directions_.to(x.device());
        const auto radial_gradient =
            2.0f * x / (1.0f + torch::sum(x * x, -1, true));
        const auto projected = torch::matmul(
            x.squeeze(2),
            directions.transpose(0, 1)
        );
        const auto asymmetric_gradient =
            asymmetry_scale_ * asymmetry_frequency_ *
            torch::matmul(
                torch::cos(asymmetry_frequency_ * projected),
                directions
            ) / std::sqrt(static_cast<float>(direction_count_));
        return radial_gradient + asymmetric_gradient.unsqueeze(2);
    }

private:
    torch::Tensor x_init_;
    float sigma_;
    float lambda_;
    float asymmetry_scale_;
    float asymmetry_frequency_;
    int64_t direction_count_;
    torch::Tensor directions_;
};

REGISTER_EQUATION_CLASS(AsymmetricHJBLQ)

extern "C" void force_link_AsymmetricHJBLQ() {}
