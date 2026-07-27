#include "equation.h"
#include "register_equation.h"

class AllenCahn final : public Equation
{
public:
	explicit AllenCahn(const EqnConfig& eqn_config)
		: Equation(eqn_config),
		  x_init_(torch::full({dim_}, eqn_config.params.value("x_init", 0.0f))),
		  sigma_(eqn_config.params.value("sigma", static_cast<float>(std::sqrt(2.0)))),
		  lambda_(eqn_config.params.value("lambda", 1.0f)),
		  terminal_offset_(eqn_config.params.value("terminal_offset", 2.0f)),
		  terminal_scale_(eqn_config.params.value("terminal_scale", 0.4f))
	{
	}

	// Sample function, Generate path of dW & X
	[[nodiscard("Return Need to be Used")]]
	std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const override
	{
		// dW ~ N(0, delta_t)
		torch::Tensor dw = torch::randn(
			{num_sample, dim_, num_time_interval_}, torch::kFloat) * sqrt_delta_t_;

		// Init X: x_0 = x_init
		torch::Tensor x = torch::zeros(
			{num_sample, dim_, num_time_interval_ + 1}, torch::kFloat);
		x.index_put_(
			{torch::indexing::Slice(), torch::indexing::Slice(), 0},
			x_init_.expand({num_sample, dim_}));

		for (int64_t i = 0; i < num_time_interval_; ++i)
		{
			x.index_put_(
				{torch::indexing::Slice(), torch::indexing::Slice(), i + 1},
				x.index({torch::indexing::Slice(), torch::indexing::Slice(), i}) +
					sigma_ * dw.index(
						{torch::indexing::Slice(), torch::indexing::Slice(), i}));
		}

		return {dw, x};
	}

	// f(t, x, y, z) = lambda * (y - y^3)
	[[nodiscard("Return Need to be Used")]]
	torch::Tensor f(
		const torch::Tensor& t, const torch::Tensor& x,
		const torch::Tensor& y, const torch::Tensor& z) const override
	{
		return lambda_ * (y - torch::pow(y, 3));
	}

	// g(x) = 1 / (terminal_offset + terminal_scale * ||x||^2)
	[[nodiscard("Return Need to be Used")]]
	torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
	{
		return 1.0f / (
			terminal_offset_ + terminal_scale_ * torch::sum(x * x, /*dim=*/-1, /*keepdim=*/true)
		);
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
	float terminal_offset_;
	float terminal_scale_;
};

REGISTER_EQUATION_CLASS(AllenCahn)

extern "C" void force_link_AllenCahn() {}
