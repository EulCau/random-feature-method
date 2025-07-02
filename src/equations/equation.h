#pragma once

#include "config.h"
#include <cmath>
#include <memory>
#include <torch/torch.h>
#include <vector>

class Equation
{
public:
	Equation(const EqnConfig& eqn_config)
		: dim_(eqn_config.dim),
		  total_time_(eqn_config.total_time),
		  num_time_interval_(eqn_config.num_time_interval),
		  delta_t_(eqn_config.total_time / eqn_config.num_time_interval),
		  sqrt_delta_t_(std::sqrt(delta_t_)) {}

	virtual ~Equation() = default;

	virtual std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const = 0;

	virtual torch::Tensor f(const torch::Tensor& t, const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) const = 0;

	virtual torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const = 0;

	int64_t dim() const { return dim_; }
	float total_time() const { return total_time_; }
	int64_t num_time_interval() const { return num_time_interval_; }
	float delta_t() const { return delta_t_; }
	float sqrt_delta_t() const { return sqrt_delta_t_; }

protected:
	int64_t dim_;
	float total_time_;
	int64_t num_time_interval_;
	float delta_t_;
	float sqrt_delta_t_;
};
