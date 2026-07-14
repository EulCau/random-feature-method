#include "equation.h"
#include "register_equation.h"

#include <cmath>

class AsymmetricAllenCahn final : public Equation
{
public:
    explicit AsymmetricAllenCahn(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full({dim_}, eqn_config.params.value("x_init", 0.0f))),
          sigma_(eqn_config.params.value("sigma", static_cast<float>(std::sqrt(2.0)))),
          lambda_(eqn_config.params.value("lambda", 1.0f)),
          terminal_offset_(eqn_config.params.value("terminal_offset", 2.0f)),
          terminal_scale_(eqn_config.params.value("terminal_scale", 0.4f)),
          asymmetry_scale_(eqn_config.params.value("asymmetry_scale", 0.05f)),
          asymmetry_frequency_(eqn_config.params.value("asymmetry_frequency", 1.0f)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{20}))
    {
        TORCH_CHECK(direction_count_ > 0, "direction_count must be positive");
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        const auto dim_idx = torch::arange(1, dim_ + 1, opts).reshape({1, dim_});
        const auto dir_idx = torch::arange(1, direction_count_ + 1, opts).reshape({direction_count_, 1});
        directions_ = torch::sin(12.9898f * dir_idx * dim_idx + 0.123f) +
            0.5f * torch::cos(78.233f * (dir_idx + 1.0f) * dim_idx + 0.456f);
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
        return lambda_ * (y - torch::pow(y, 3));
    }

    [[nodiscard]] torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        const auto radial = 1.0f /
            (terminal_offset_ + terminal_scale_ * torch::sum(x * x, -1, true));
        const auto directions = directions_.to(x.device());
        const auto projected = torch::matmul(x.squeeze(2), directions.transpose(0, 1));
        const auto asymmetric = asymmetry_scale_ *
            torch::tanh(asymmetry_frequency_ * projected).sum(-1, true) /
            std::sqrt(static_cast<float>(direction_count_));
        return radial + asymmetric.unsqueeze(2);
    }

    [[nodiscard]] bool has_analytic_jacobian() const override { return true; }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> terminal_residual_and_jacobian(
        const torch::Tensor& t, const torch::Tensor& t_end,
        const torch::Tensor& x, const torch::Tensor& x_end,
        const torch::Tensor& dw, const torch::Tensor& H,
        const torch::Tensor& y0, const torch::Tensor& alpha) const override
    {
        using namespace torch::indexing;
        const int64_t S = x.size(0);
        const int64_t T = x.size(1);
        const int64_t D = alpha.size(0);
        const int64_t Hdim = alpha.size(1);
        auto y = y0.reshape({1, 1}).expand({S, 1}).contiguous();
        auto sensitivity_y0 = torch::ones({S, 1}, alpha.options());
        auto sensitivity_alpha = torch::zeros({S, D, Hdim}, alpha.options());
        const auto features = H.squeeze(-1).contiguous();
        const auto z_all = torch::matmul(features, alpha.transpose(0, 1));
        const auto dw_all = dw.permute({0, 2, 1}).contiguous();

        for (int64_t k = 0; k < T; ++k)
        {
            const auto h_k = features.index({Slice(), k, Slice()});
            const auto z_k = z_all.index({Slice(), k, Slice()});
            const auto dw_k = dw_all.index({Slice(), k, Slice()});
            const auto scale = 1.0f - delta_t_ * lambda_ * (1.0f - 3.0f * y * y);
            sensitivity_y0 = scale * sensitivity_y0;
            sensitivity_alpha = scale.reshape({S, 1, 1}) * sensitivity_alpha +
                dw_k.unsqueeze(2) * h_k.unsqueeze(1);
            y = y - delta_t_ * lambda_ * (y - torch::pow(y, 3)) +
                torch::sum(dw_k * z_k, -1, true);
        }

        const auto residual = y - g(t_end, x_end).reshape({S, 1});
        const auto jacobian = torch::cat({sensitivity_y0,
            sensitivity_alpha.reshape({S, D * Hdim})}, 1).contiguous();
        return {residual.contiguous(), jacobian};
    }

private:
    torch::Tensor x_init_;
    float sigma_;
    float lambda_;
    float terminal_offset_;
    float terminal_scale_;
    float asymmetry_scale_;
    float asymmetry_frequency_;
    int64_t direction_count_;
    torch::Tensor directions_;
};

REGISTER_EQUATION_CLASS(AsymmetricAllenCahn)

extern "C" void force_link_AsymmetricAllenCahn() {}
