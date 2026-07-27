#include "equation.h"
#include "register_equation.h"

#include <algorithm>
#include <cmath>

class AnisotropicQuadraticHJBLQ final : public Equation
{
public:
    explicit AnisotropicQuadraticHJBLQ(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full({dim_}, eqn_config.params.value("x_init", 0.0f))),
          sigma_(eqn_config.params.value("sigma", static_cast<float>(std::sqrt(2.0)))),
          lambda_(eqn_config.params.value("lambda", 1.0f)),
          eigenvalue_min_(eqn_config.params.value("eigenvalue_min", 0.05f)),
          eigenvalue_max_(eqn_config.params.value("eigenvalue_max", 0.25f)),
          direction_count_(eqn_config.params.value("direction_count", int64_t{10}))
    {
        TORCH_CHECK(sigma_ > 0.0f, "sigma must be positive");
        TORCH_CHECK(lambda_ > 0.0f, "lambda must be positive");
        TORCH_CHECK(eigenvalue_min_ > 0.0f, "eigenvalue_min must be positive");
        TORCH_CHECK(
            eigenvalue_max_ >= eigenvalue_min_,
            "eigenvalue_max must be at least eigenvalue_min"
        );
        TORCH_CHECK(
            direction_count_ > 0 && direction_count_ < dim_,
            "direction_count must satisfy 0 < direction_count < dimension"
        );

        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        const auto coordinate = torch::arange(0, dim_, opts).reshape({1, dim_});
        const auto mode = torch::arange(
            1,
            direction_count_ + 1,
            opts
        ).reshape({direction_count_, 1});
        const float pi = static_cast<float>(std::acos(-1.0));
        directions_ = std::sqrt(2.0f / static_cast<float>(dim_)) *
            torch::cos(
                pi * (coordinate + 0.5f) * mode /
                static_cast<float>(dim_)
            );
        eigenvalues_ = torch::linspace(
            eigenvalue_min_,
            eigenvalue_max_,
            direction_count_,
            opts
        );
    }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> sample(
        const int64_t num_sample) const override
    {
        auto dw = torch::randn(
            {num_sample, dim_, num_time_interval_},
            torch::kFloat
        ) * sqrt_delta_t_;
        auto x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1},
            torch::kFloat
        );
        x.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(), 0},
            x_init_.expand({num_sample, dim_})
        );

        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            x.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(), i + 1},
                x.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    i
                }) +
                sigma_ * dw.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    i
                })
            );
        }
        return {dw, x};
    }

    [[nodiscard]] torch::Tensor f(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& y,
        const torch::Tensor& z) const override
    {
        return -0.5f * lambda_ * torch::sum(z * z, -1, true);
    }

    [[nodiscard]] torch::Tensor g(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        const auto projected = torch::matmul(
            x.squeeze(2),
            directions_.to(x.device()).transpose(0, 1)
        );
        const auto weighted_square =
            projected.square() * eigenvalues_.to(x.device());
        return weighted_square.sum(-1, true).unsqueeze(2);
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
    float sigma_;
    float lambda_;
    float eigenvalue_min_;
    float eigenvalue_max_;
    int64_t direction_count_;
    torch::Tensor directions_;
    torch::Tensor eigenvalues_;
};

REGISTER_EQUATION_CLASS(AnisotropicQuadraticHJBLQ)

extern "C" void force_link_AnisotropicQuadraticHJBLQ() {}
