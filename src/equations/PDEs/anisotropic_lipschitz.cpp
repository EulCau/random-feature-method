#include "equation.h"
#include "register_equation.h"

#include <cmath>

class AnisotropicLipschitz final : public Equation
{
public:
    explicit AnisotropicLipschitz(const EqnConfig& eqn_config)
        : Equation(eqn_config),
          x_init_(torch::full(
              {dim_},
              eqn_config.params.value("x_init", 0.0),
              torch::TensorOptions().dtype(dtype_))),
          sigma_min_(eqn_config.params.value("sigma_min", 0.6)),
          sigma_max_(eqn_config.params.value("sigma_max", 1.4)),
          direction_count_(
              eqn_config.params.value("direction_count", int64_t{10})),
          frequency_min_(
              eqn_config.params.value("frequency_min", 0.7)),
          frequency_max_(
              eqn_config.params.value("frequency_max", 1.8)),
          phase_min_(eqn_config.params.value("phase_min", 0.2)),
          phase_max_(eqn_config.params.value("phase_max", 1.1)),
          time_weight_min_(
              eqn_config.params.value("time_weight_min", 0.5)),
          time_weight_max_(
              eqn_config.params.value("time_weight_max", 1.5)),
          baseline_(eqn_config.params.value("baseline", 0.25)),
          time_shift_(eqn_config.params.value("time_shift", 1.0)),
          spatial_scale_(eqn_config.params.value("spatial_scale", 0.5)),
          time_modulation_(
              eqn_config.params.value("time_modulation", 0.5)),
          reaction_strength_(
              eqn_config.params.value("reaction_strength", 0.5)),
          control_strength_(
              eqn_config.params.value("control_strength", 0.25))
    {
        TORCH_CHECK(!linear_, "AnisotropicLipschitz must be nonlinear");
        TORCH_CHECK(sigma_min_ > 0.0, "sigma_min must be positive");
        TORCH_CHECK(
            sigma_max_ >= sigma_min_,
            "sigma_max must be at least sigma_min"
        );
        TORCH_CHECK(
            direction_count_ > 0 && direction_count_ < dim_,
            "direction_count must satisfy 0 < direction_count < dimension"
        );
        TORCH_CHECK(frequency_min_ > 0.0, "frequency_min must be positive");
        TORCH_CHECK(
            frequency_max_ >= frequency_min_,
            "frequency_max must be at least frequency_min"
        );
        TORCH_CHECK(
            time_weight_min_ > 0.0,
            "time_weight_min must be positive"
        );
        TORCH_CHECK(
            time_weight_max_ >= time_weight_min_,
            "time_weight_max must be at least time_weight_min"
        );
        TORCH_CHECK(spatial_scale_ > 0.0, "spatial_scale must be positive");
        TORCH_CHECK(
            time_modulation_ >= 0.0,
            "time_modulation must be nonnegative"
        );
        TORCH_CHECK(
            reaction_strength_ >= 0.0,
            "reaction_strength must be nonnegative"
        );
        TORCH_CHECK(
            control_strength_ >= 0.0,
            "control_strength must be nonnegative"
        );

        const auto options = torch::TensorOptions().dtype(dtype_);
        const auto coordinate = torch::arange(0, dim_, options)
            .reshape({1, dim_});
        const auto modes = torch::arange(
            1,
            direction_count_ + 1,
            options
        ).reshape({direction_count_, 1});
        const double pi = std::acos(-1.0);
        directions_ = std::sqrt(2.0 / static_cast<double>(dim_)) *
            torch::cos(
                pi * (coordinate + 0.5) * modes /
                static_cast<double>(dim_)
            );

        sigma_ = torch::linspace(sigma_min_, sigma_max_, dim_, options);
        frequencies_ = torch::linspace(
            frequency_min_,
            frequency_max_,
            direction_count_,
            options
        );
        phases_ = torch::linspace(
            phase_min_,
            phase_max_,
            direction_count_,
            options
        );
        time_weights_ = torch::linspace(
            time_weight_min_,
            time_weight_max_,
            direction_count_,
            options
        );
        control_direction_ = torch::matmul(time_weights_, directions_);
        control_direction_ = control_direction_ /
            control_direction_.norm().clamp_min(1.0e-12);
        mode_diffusion_variances_ = (
            directions_.square() * sigma_.square().reshape({1, dim_})
        ).sum(1);
    }

    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> sample(
        const int64_t num_sample) const override
    {
        const auto options = torch::TensorOptions().dtype(dtype_);
        auto dw = torch::randn(
            {num_sample, dim_, num_time_interval_},
            options
        ) * sqrt_delta_t_;
        auto x = torch::zeros(
            {num_sample, dim_, num_time_interval_ + 1},
            options
        );
        x.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(), 0},
            x_init_.expand({num_sample, dim_})
        );

        const auto sigma = sigma_.reshape({1, dim_});
        for (int64_t i = 0; i < num_time_interval_; ++i)
        {
            using namespace torch::indexing;
            x.index_put_(
                {Slice(), Slice(), i + 1},
                x.index({Slice(), Slice(), i}) +
                    sigma * dw.index({Slice(), Slice(), i})
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
        check_tx(t, x);
        TORCH_CHECK(
            y.dim() == 4 && y.size(0) == x.size(0) &&
                y.size(1) == x.size(1) &&
                y.size(-2) == 1 && y.size(-1) == 1,
            "y must have shape (B, T, 1, 1), got ", y.sizes()
        );
        TORCH_CHECK(
            z.dim() == 4 && z.size(0) == x.size(0) &&
                z.size(1) == x.size(1) &&
                z.size(-2) == 1 && z.size(-1) == dim_,
            "z must have shape (B, T, 1, ", dim_, "), got ", z.sizes()
        );

        const auto control_direction = control_direction_.to(z.device());
        const auto control_projection = (
            z * control_direction.reshape({1, 1, 1, dim_})
        ).sum(-1, true);
        return reaction_strength_ * torch::tanh(y) +
            control_strength_ * torch::tanh(control_projection) +
            manufactured_source(t, x);
    }

    [[nodiscard]] torch::Tensor g(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return exact_value(torch::full_like(t, total_time_), x);
    }

    [[nodiscard]] bool has_reference_solution() const override
    {
        return true;
    }

    [[nodiscard]] torch::Tensor reference_solution(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return exact_value(t, x);
    }

    [[nodiscard]] torch::Tensor terminal_gradient(
        const torch::Tensor& t,
        const torch::Tensor& x) const override
    {
        return exact_spatial_gradient(torch::full_like(t, total_time_), x);
    }

    [[nodiscard]] torch::Tensor gradient_to_z(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& spatial_gradient) const override
    {
        TORCH_CHECK(
            spatial_gradient.size(-1) == dim_,
            "spatial_gradient last dimension must be ", dim_,
            ", got ", spatial_gradient.sizes()
        );
        return spatial_gradient * sigma_.to(spatial_gradient.device());
    }

private:
    void check_tx(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        TORCH_CHECK(
            t.dim() == 4 && t.size(-2) == 1 && t.size(-1) == 1,
            "t must have shape (1 or B, T, 1, 1), got ", t.sizes()
        );
        TORCH_CHECK(
            x.dim() == 4 && x.size(-2) == 1 && x.size(-1) == dim_,
            "x must have shape (B, T, 1, ", dim_, "), got ", x.sizes()
        );
        TORCH_CHECK(
            t.size(1) == x.size(1) &&
                (t.size(0) == 1 || t.size(0) == x.size(0)),
            "t and x leading dimensions are incompatible: ",
            t.sizes(), " and ", x.sizes()
        );
    }

    [[nodiscard]] torch::Tensor mode_angles(
        const torch::Tensor& x) const
    {
        const auto directions = directions_.to(x.device());
        const auto frequencies = frequencies_.to(x.device());
        const auto phases = phases_.to(x.device());
        return torch::matmul(
            x.squeeze(2),
            directions.transpose(0, 1)
        ) * frequencies + phases;
    }

    [[nodiscard]] torch::Tensor mode_amplitudes(
        const torch::Tensor& t) const
    {
        const auto remaining_time = (total_time_ - t)
            .squeeze(-1)
            .squeeze(-1)
            .unsqueeze(-1);
        return spatial_scale_ * (
            1.0 + time_modulation_ * remaining_time *
                time_weights_.to(t.device())
        );
    }

    [[nodiscard]] torch::Tensor exact_value(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        check_tx(t, x);
        const auto remaining_time = (total_time_ - t)
            .squeeze(-1)
            .squeeze(-1)
            .unsqueeze(-1);
        const auto directional_value = (
            mode_amplitudes(t) * torch::sin(mode_angles(x))
        ).sum(-1, true) /
            std::sqrt(static_cast<double>(direction_count_));
        return (
            baseline_ + time_shift_ * remaining_time + directional_value
        ).unsqueeze(2).contiguous();
    }

    [[nodiscard]] torch::Tensor exact_spatial_gradient(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        check_tx(t, x);
        const auto mode_weights =
            mode_amplitudes(t) *
            frequencies_.to(x.device()) *
            torch::cos(mode_angles(x)) /
            std::sqrt(static_cast<double>(direction_count_));
        return torch::matmul(
            mode_weights,
            directions_.to(x.device())
        ).unsqueeze(2).contiguous();
    }

    [[nodiscard]] torch::Tensor exact_time_derivative(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        check_tx(t, x);
        const auto directional_derivative =
            -spatial_scale_ * time_modulation_ *
            (
                time_weights_.to(x.device()) *
                torch::sin(mode_angles(x))
            ).sum(-1, true) /
            std::sqrt(static_cast<double>(direction_count_));
        return (
            -time_shift_ + directional_derivative
        ).unsqueeze(2).contiguous();
    }

    [[nodiscard]] torch::Tensor exact_diffusion_operator(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        check_tx(t, x);
        const auto mode_terms =
            mode_amplitudes(t) *
            frequencies_.to(x.device()).square() *
            mode_diffusion_variances_.to(x.device()) *
            torch::sin(mode_angles(x));
        return (
            -0.5 * mode_terms.sum(-1, true) /
            std::sqrt(static_cast<double>(direction_count_))
        ).unsqueeze(2).contiguous();
    }

    [[nodiscard]] torch::Tensor manufactured_source(
        const torch::Tensor& t,
        const torch::Tensor& x) const
    {
        const auto exact_y = exact_value(t, x);
        const auto exact_z = gradient_to_z(
            t,
            x,
            exact_spatial_gradient(t, x)
        );
        const auto control_projection = (
            exact_z * control_direction_.to(x.device())
        ).sum(-1, true);

        // Enforce u_t + 0.5 Tr(sigma sigma^T Hess u) + f = 0 at u = u*.
        return -exact_time_derivative(t, x) -
            exact_diffusion_operator(t, x) -
            reaction_strength_ * torch::tanh(exact_y) -
            control_strength_ * torch::tanh(control_projection);
    }

    torch::Tensor x_init_;
    double sigma_min_;
    double sigma_max_;
    int64_t direction_count_;
    double frequency_min_;
    double frequency_max_;
    double phase_min_;
    double phase_max_;
    double time_weight_min_;
    double time_weight_max_;
    double baseline_;
    double time_shift_;
    double spatial_scale_;
    double time_modulation_;
    double reaction_strength_;
    double control_strength_;
    torch::Tensor sigma_;
    torch::Tensor directions_;
    torch::Tensor control_direction_;
    torch::Tensor frequencies_;
    torch::Tensor phases_;
    torch::Tensor time_weights_;
    torch::Tensor mode_diffusion_variances_;
};

REGISTER_EQUATION_CLASS(AnisotropicLipschitz)

extern "C" void force_link_AnisotropicLipschitz() {}
