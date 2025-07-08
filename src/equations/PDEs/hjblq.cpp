#include "equation.h"
#include "register_equation.h"

class HJBLQ final : public Equation
{
public:
	explicit HJBLQ(const EqnConfig& eqn_config)
		: Equation(eqn_config),
		  x_init_(torch::zeros({dim_})),
		  sigma_(static_cast<float>(std::sqrt(2.0))),
		  lambda_(1.0)
	{
	}

	// Sample function, Generate path of dW & X
	[[nodiscard("Reture Need to be Used")]]
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

	// f(t, x, y, z) = -lambda * 0.5 * ||z||^2
	[[nodiscard("Reture Need to be Used")]]
	torch::Tensor f(
		const torch::Tensor& t, const torch::Tensor& x,
		const torch::Tensor& y, const torch::Tensor& z) const override
	{
		return -0.5 * lambda_ * torch::sum(z * z, /*dim=*/1, /*keepdim=*/true);
	}

	// g(x) = log((1 + ||x||^2) / 2)
	[[nodiscard("Reture Need to be Used")]]
	torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
	{
		return torch::log((1 + torch::sum(x * x, /*dim=*/1, /*keepdim=*/true)) / 2.0);
	}

private:
	torch::Tensor x_init_;
	float sigma_;
	float lambda_;
};

REGISTER_EQUATION_CLASS(HJBLQ)

extern "C" void force_link_HJBLQ() {}
