#include "non_shared_model.h"

NonSharedModelImpl::NonSharedModelImpl(const Config& config, const std::shared_ptr<Equation>& bsde)
	: eqn_config_(config.eqn_config), net_config_(config.net_config), bsde_(bsde)
{
	int64_t dim = eqn_config_.dim;

	// Init y_init & z_init parameters
	y_init_ = register_parameter(
		"y_init",
		torch::empty({ 1 }).uniform_(net_config_.y_init_range[0], net_config_.y_init_range[1]));

	z_init_ = register_parameter(
		"z_init",
		torch::empty({ 1, dim }).uniform_(-0.1f, 0.1f));

	// Initialize subnets
	for (int64_t i = 0; i < eqn_config_.num_time_interval - 1; ++i)
	{
		auto subnet = MLP(std::make_shared<MLPImpl>(config));
		subnets_.push_back(register_module("subnet_" + std::to_string(i), subnet));
	}
}

torch::Tensor NonSharedModelImpl::forward(const std::pair<torch::Tensor, torch::Tensor>& inputs, bool training)
{
	torch::Tensor dw = inputs.first;					// shape: [batch_size, dim, N]
	torch::Tensor x = inputs.second;					// shape: [batch_size, dim, N+1]

	int64_t batch_size = dw.size(0);
	int64_t dim = eqn_config_.dim;
	int64_t N = eqn_config_.num_time_interval;

	torch::Tensor all_one = torch::ones({ batch_size, 1 });
	torch::Tensor y = all_one * y_init_;				// shape: [batch_size, 1]
	torch::Tensor z = all_one.matmul(z_init_);			// shape: [batch_size, dim]

	for (int64_t t = 0; t < N - 1; ++t)
	{
		float time = t * bsde_->delta_t();

		auto x_t = x.select(2, t);						// x[:,:,t]
		auto dw_t = dw.select(2, t);					// dw[:,:,t]

		auto f_val = bsde_->f(torch::tensor(time), x_t, y, z);
		y = y - bsde_->delta_t() * f_val + torch::sum(z * dw_t, 1, true);
		z = subnets_[t]->forward(x.select(2, t + 1));	// x[:,:,t+1]
		z = z / dim;
	}

	// Last step
	float final_time = (N - 1) * bsde_->delta_t();
	auto x_last = x.select(2, N - 2);					// x[:,:,N-2]
	auto dw_last = dw.select(2, N - 1);					// dw[:,:,N-1]

	y = y - bsde_->delta_t() * bsde_->f(torch::tensor(final_time), x_last, y, z)
		+ torch::sum(z * dw_last, 1, true);

	return y;											// shape: [batch_size, 1]
}

std::vector<torch::Tensor> NonSharedModelImpl::parameters_flattened() const
{
	std::vector<torch::Tensor> params;

	// Add y_init & z_init
	params.push_back(y_init_);
	params.push_back(z_init_);

	// Add parameters from subnets
	for (const auto& subnet : subnets_)
	{
		for (const auto& p : subnet->parameters())
		{
			params.push_back(p);
		}
	}

	return params;
}
