#include "equation.h"
#include "register_equation.h"

struct HeatCoefficient final : Coefficient
{
    [[nodiscard]] torch::Tensor L(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return torch::zeros(
            {x.size(0), x.size(1), 1, 1},
            x.options()
        );
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
        return torch::zeros(
            {x.size(0), x.size(1), 1, 1},
            x.options()
        );
    }
};

class Heat final : public Equation
{
public:
    explicit Heat(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(eqn_config.params.value("x_init", 0.0f)),
          terminal_scale_(
              eqn_config.params.value("terminal_scale", static_cast<float>(dim_)))
    {
        TORCH_CHECK(terminal_scale_ > 0.0f, "terminal_scale must be positive");

        if (linear_)
        {
            coefficient_ = std::make_shared<HeatCoefficient>();
        }
    }

    // Heat equation: dX_t = dW_t, f(t, x, y, z) = 0.
    [[nodiscard("Return Need to be Used")]]
    std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const override
    {
        const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        const auto dw = torch::randn(
            {num_sample, dim_, num_time_interval_}, opts
        ) * sqrt_delta_t_;

        auto x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1}, opts
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

    // g(x) = exp(-||x||^2 / terminal_scale).
    [[nodiscard("Return Need to be Used")]]
    torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        TORCH_CHECK(x.dim() >= 4, "x must have at least 4 dimensions");

        const auto norm_sq = torch::sum(
            x * x,
            /*dim=*/-1,
            /*keepdim=*/true
        );

        return torch::exp(-norm_sq / terminal_scale_);
    }

private:
    float x_init_;
    float terminal_scale_;
};

REGISTER_EQUATION_CLASS(Heat)

extern "C" void force_link_Heat() {}
