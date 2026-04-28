#include "equation.h"
#include "register_equation.h"

struct BSMCoefficient final : Coefficient
{
    explicit BSMCoefficient(const float r) : r_{r} {}
    float r_;
    [[nodiscard]] torch::Tensor L(const torch::Tensor& t,
                    const torch::Tensor& x) const override
    {
        const auto sizes = x.sizes();
        TORCH_CHECK(sizes.size() >= 2, "x must have at least 2 dimensions");

        return torch::full(
            {sizes[0], sizes[1], 1, 1},
            -r_,
            x.options()
        );
    }

    [[nodiscard]] torch::Tensor M(const torch::Tensor& t,
                    const torch::Tensor& x) const override
    {
        return torch::zeros_like(x);
    }

    [[nodiscard]] torch::Tensor N(const torch::Tensor& t,
                    const torch::Tensor& x) const override
    {
        const auto sizes = x.sizes();
        TORCH_CHECK(sizes.size() >= 2, "x must have at least 2 dimensions");

        return torch::zeros(
            {sizes[0], sizes[1], 1, 1},
            x.options()
        );
    }
};

class BSM final : public Equation
{
public:
    explicit BSM(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::ones({dim_})),
          sigma_(0.2f), r_(0.05f), K_(1.0f)
    {
        linear_ = true;
        coefficient_ = std::make_shared<BSMCoefficient>(r_);
    }

    // Sample function, Generate path of dW & X
    [[nodiscard("Return Need to be Used")]]
    std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const override
    {
        const auto device = torch::cuda::is_available()?torch::kCUDA:torch::kCPU;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        // dW ~ N(0, delta_t)
        torch::Tensor dw = torch::randn(
            {num_sample, dim_, num_time_interval_}, opts) * sqrt_delta_t_;

        // Init X: x_0 = x_init
        torch::Tensor x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1}, opts);
        x.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(), 0},
            x_init_.expand({num_sample, dim_}));

        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            using namespace at::indexing;
            auto xi = x.index({Slice(), Slice(), i});
            auto dwi = dw.index({Slice(), Slice(), i});

            x.index_put_(
                {Slice(), Slice(), i + 1},
                xi * torch::exp(
                    (r_ - 0.5 * sigma_ * sigma_) * delta_t_
                    + sigma_ * dwi
                )
            );
        }

        return {dw, x};
    }

    // f(t, x, y, z) = -r * y
    [[nodiscard("Return Need to be Used")]]
    torch::Tensor f(
        const torch::Tensor& t, const torch::Tensor& x,
        const torch::Tensor& y, const torch::Tensor& z) const override
    {
        return -r_ * y;
    }

    // g(x) = max(mean(x) − K, 0)
    [[nodiscard("Return Need to be Used")]]
    torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
    {
        TORCH_CHECK(x.dim() >= 4, "x must have at least 4 dimensions");

        const auto mean_x = torch::mean(x, -1, true);

        return torch::relu(mean_x - K_);
    }

    [[nodiscard]] bool has_analytic_jacobian() const override { return true; }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> terminal_residual_and_jacobian(
        const torch::Tensor& t,
        const torch::Tensor& t_end,
        const torch::Tensor& x,
        const torch::Tensor& x_end,
        const torch::Tensor& dw,
        const torch::Tensor& H,
        const torch::Tensor& y0,
        const torch::Tensor& alpha) const override
    {
        using namespace torch::indexing;

        const int64_t S = x.size(0);
        const int64_t T = x.size(1);
        const int64_t D = alpha.size(0);
        const int64_t Hdim = alpha.size(1);
        const float a = 1.0f + r_ * delta_t_;

        auto y = y0.reshape({1, 1}).expand({S, 1}).contiguous();
        auto sensitivity_y0 = torch::ones({S, 1}, alpha.options());
        auto sensitivity_alpha = torch::zeros({S, D, Hdim}, alpha.options());

        const auto features = H.squeeze(-1).contiguous(); // (S, T, H)
        const auto z_all = torch::matmul(features, alpha.transpose(0, 1)); // (S, T, D)
        const auto dw_all = dw.permute({0, 2, 1}).contiguous(); // (S, T, D)

        for (int64_t k = 0; k < T; ++k)
        {
            const auto h_k = features.index({Slice(), k, Slice()});
            const auto z_k = z_all.index({Slice(), k, Slice()});
            const auto dw_k = dw_all.index({Slice(), k, Slice()});

            sensitivity_y0 = a * sensitivity_y0;
            sensitivity_alpha = a * sensitivity_alpha
                + dw_k.unsqueeze(2) * h_k.unsqueeze(1);

            y = a * y + torch::sum(dw_k * z_k, -1, true);
        }

        const auto residual = y - g(t_end, x_end).reshape({S, 1});
        const auto jacobian = torch::cat({
            sensitivity_y0,
            sensitivity_alpha.reshape({S, D * Hdim})
        }, 1).contiguous();

        return {residual.contiguous(), jacobian};
    }

private:
    torch::Tensor x_init_;
    float sigma_;
    float r_;
    float K_;
};

REGISTER_EQUATION_CLASS(BSM)

extern "C" void force_link_BSM() {}
