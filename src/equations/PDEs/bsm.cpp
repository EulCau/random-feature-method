#include "equation.h"
#include "register_equation.h"

struct BSMCoefficient final : Coefficient {
    explicit BSMCoefficient(const float r) : r_{r} {}
    float r_;
    [[nodiscard]] torch::Tensor L(const torch::Tensor& t,
                    const torch::Tensor& x) const override {
        const auto sizes = x.sizes();
        TORCH_CHECK(sizes.size() >= 2, "x must have at least 2 dimensions");

        return torch::full(
            {sizes[0], sizes[1], 1, 1},
            -r_,
            x.options()
        );
    }

    [[nodiscard]] torch::Tensor M(const torch::Tensor& t,
                    const torch::Tensor& x) const override {
        return torch::zeros_like(x);
    }

    [[nodiscard]] torch::Tensor N(const torch::Tensor& t,
                    const torch::Tensor& x) const override {
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

private:
    torch::Tensor x_init_;
    float sigma_;
    float r_;
    float K_;
};

REGISTER_EQUATION_CLASS(BSM)

extern "C" void force_link_BSM() {}
