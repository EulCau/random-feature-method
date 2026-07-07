#include "equation.h"
#include "register_equation.h"

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
          tanh_frequency_(eqn_config.params.value("tanh_frequency", 1.0f))
    {
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        const auto idx = torch::arange(1, dim_ + 1, opts);

        direction_a_ = torch::sin(12.9898f * idx + 0.123f);
        direction_b_ = torch::cos(78.233f * idx + 0.456f);
        direction_c_ = torch::sin(37.719f * idx + 0.789f) +
            0.5f * torch::cos(11.131f * idx);

        direction_a_ = direction_a_ / direction_a_.norm();
        direction_b_ = direction_b_ / direction_b_.norm();
        direction_c_ = direction_c_ / direction_c_.norm();

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

        const auto direction_a = direction_a_.to(x.device()).reshape({1, 1, 1, dim_});
        const auto direction_b = direction_b_.to(x.device()).reshape({1, 1, 1, dim_});
        const auto direction_c = direction_c_.to(x.device()).reshape({1, 1, 1, dim_});

        const auto xa = torch::sum(x * direction_a, -1, true);
        const auto xb = torch::sum(x * direction_b, -1, true);
        const auto xc = torch::sum(x * direction_c, -1, true);

        return torch::sin(sin_frequency_ * xa) +
            cos_weight_ * torch::cos(cos_frequency_ * xb) +
            tanh_weight_ * torch::tanh(tanh_frequency_ * xc);
    }

private:
    float x_init_;
    float cos_weight_;
    float tanh_weight_;
    float sin_frequency_;
    float cos_frequency_;
    float tanh_frequency_;
    torch::Tensor direction_a_;
    torch::Tensor direction_b_;
    torch::Tensor direction_c_;
};

REGISTER_EQUATION_CLASS(AsymmetricHeat)

extern "C" void force_link_AsymmetricHeat() {}
