#pragma once

#include "config.h"
#include <cmath>
#include <memory>
#include <torch/torch.h>
#include <utility>

struct Coefficient {
	virtual ~Coefficient() = default;

	[[nodiscard]] virtual torch::Tensor L(const torch::Tensor& t, const torch::Tensor& x) const = 0;
	[[nodiscard]] virtual torch::Tensor M(const torch::Tensor& t, const torch::Tensor& x) const = 0;
	[[nodiscard]] virtual torch::Tensor N(const torch::Tensor& t, const torch::Tensor& x) const = 0;
};

class Equation
{
public:
	explicit Equation(const EqnConfig& eqn_config)
		: dim_(eqn_config.dimension),
		  linear_(eqn_config.is_linear),
		  total_time_(eqn_config.total_time),
		  num_time_interval_(eqn_config.num_time_intervals),
		  delta_t_(eqn_config.total_time / static_cast<float>(eqn_config.num_time_intervals)),
		  sqrt_delta_t_(std::sqrt(delta_t_)) {}

	virtual ~Equation() = default;

	[[nodiscard]] virtual std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const = 0;

	[[nodiscard]] virtual torch::Tensor f(const torch::Tensor& t, const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) const = 0;

	[[nodiscard]] virtual torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const = 0;

	[[nodiscard]] virtual bool has_analytic_jacobian() const { return false; }

	[[nodiscard]] virtual std::pair<torch::Tensor, torch::Tensor> terminal_residual_and_jacobian(
		const torch::Tensor& t,
		const torch::Tensor& t_end,
		const torch::Tensor& x,
		const torch::Tensor& x_end,
		const torch::Tensor& dw,
		const torch::Tensor& H,
		const torch::Tensor& y0,
		const torch::Tensor& alpha) const
	{
		TORCH_CHECK(false, "Analytic terminal residual/Jacobian is not implemented for this equation");
	}

	[[nodiscard]] int64_t dim() const { return dim_; }
	[[nodiscard]] float total_time() const { return total_time_; }
	[[nodiscard]] int64_t num_time_interval() const { return num_time_interval_; }
	[[nodiscard]] float delta_t() const { return delta_t_; }
	[[nodiscard]] float sqrt_delta_t() const { return sqrt_delta_t_; }
    [[nodiscard]] bool is_linear() const { return linear_; }
    [[nodiscard]] bool has_coefficient() const { return coefficient_ != nullptr; }
    [[nodiscard]] const Coefficient& coef() const
    {
        TORCH_CHECK(coefficient_ != nullptr, "Linear coefficient L/M/N is not defined for this equation");
        return *coefficient_;
    }

protected:
	int64_t dim_;
    bool linear_;
	float total_time_;
	int64_t num_time_interval_;
	float delta_t_;
	float sqrt_delta_t_;
	std::shared_ptr<Coefficient> coefficient_;
};
