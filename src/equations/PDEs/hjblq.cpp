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

	// f(t, x, y, z) = -lambda * 0.5 * ||z||^2
	[[nodiscard("Return Need to be Used")]]
	torch::Tensor f(
		const torch::Tensor& t, const torch::Tensor& x,
		const torch::Tensor& y, const torch::Tensor& z) const override
	{
		return -0.5 * lambda_ * torch::sum(z * z, /*dim=*/-1, /*keepdim=*/true);
	}

	// g(x) = log((1 + ||x||^2) / 2)
	[[nodiscard("Return Need to be Used")]]
	torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const override
	{
		return torch::log((1 + torch::sum(x * x, /*dim=*/-1, /*keepdim=*/true)) / 2.0);
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

		auto y = y0.reshape({1, 1}).expand({S, 1}).contiguous();
		auto sensitivity_alpha = torch::zeros({S, D, Hdim}, alpha.options());

		const auto features = H.squeeze(-1).contiguous(); // (S, T, H)
		const auto z_all = torch::matmul(features, alpha.transpose(0, 1)); // (S, T, D)
		const auto dw_all = dw.permute({0, 2, 1}).contiguous(); // (S, T, D)

		for (int64_t k = 0; k < T; ++k)
		{
			const auto h_k = features.index({Slice(), k, Slice()});
			const auto z_k = z_all.index({Slice(), k, Slice()});
			const auto dw_k = dw_all.index({Slice(), k, Slice()});

			const auto coef = lambda_ * delta_t_ * z_k + dw_k; // (S, D)
			sensitivity_alpha = sensitivity_alpha + coef.unsqueeze(2) * h_k.unsqueeze(1);

			y = y
				+ 0.5f * lambda_ * delta_t_ * torch::sum(z_k * z_k, -1, true)
				+ torch::sum(dw_k * z_k, -1, true);
		}

		const auto residual = y - g(t_end, x_end).reshape({S, 1});
		const auto jacobian = torch::cat({
			torch::ones({S, 1}, alpha.options()),
			sensitivity_alpha.reshape({S, D * Hdim})
		}, 1).contiguous();

		return {residual.contiguous(), jacobian};
	}

private:
	torch::Tensor x_init_;
	float sigma_;
	float lambda_;
};

REGISTER_EQUATION_CLASS(HJBLQ)

extern "C" void force_link_HJBLQ() {}
