#include "bsde_solver.h"

constexpr float DELTA_CLIP = 50.0f;

BSDESolver::BSDESolver(const Config& config, std::shared_ptr<Equation> bsde)
	: eqn_config_(config.eqn_config),
	net_config_(config.net_config),
	bsde_(bsde),
	model_(std::make_shared<NonSharedModelImpl>(config, bsde))
{
	std::vector<torch::optim::OptimizerParamGroup> param_groups =
	{
		torch::optim::OptimizerParamGroup(model_->parameters_flattened())
	};

	auto& options = torch::optim::AdamWOptions(net_config_.lr_values[0]).eps(1e-8f);

	optimizer_ = std::make_unique<torch::optim::AdamW>(param_groups, options);

	int64_t warmup_steps = net_config_.warmup_steps;
	int64_t total_steps = net_config_.num_iterations;

	auto lr_lambda = create_lr_lambda(warmup_steps, total_steps);

	lr_scheduler_ = std::make_unique<LambdaLRScheduler>(*optimizer_, lr_lambda);
}

torch::Tensor BSDESolver::loss_fn(const std::pair<torch::Tensor, torch::Tensor>& inputs, bool training)
{
	auto y_pred = model_->forward(inputs, training);  // shape: [batch_size, 1]
	auto y_true = bsde_->g(torch::tensor(eqn_config_.total_time), inputs.second.select(2, eqn_config_.num_time_interval));  // x[:, :, -1]

	auto delta = y_pred - y_true;
	auto abs_delta = delta.abs();

	auto loss = torch::mean(
		torch::where(abs_delta < DELTA_CLIP,
			delta.pow(2),
			2 * DELTA_CLIP * abs_delta - DELTA_CLIP * DELTA_CLIP));
	return loss;
}

void BSDESolver::train()
{
	auto start_time = std::chrono::steady_clock::now();
	auto valid_data = bsde_->sample(net_config_.valid_size);

	for (int64_t step = 0; step < net_config_.num_iterations; ++step)
	{
		if ((step + 1) % net_config_.logging_frequency == 0)
		{
			auto loss = loss_fn(valid_data, false).item<float>();
			auto y0 = model_->y_init().item<float>();
			auto now = std::chrono::steady_clock::now();
			auto elapsed_sec = std::chrono::duration<double>(now - start_time).count();

			std::cout << std::fixed << std::setprecision(6)
				<< "step: " << std::setw(5) << step + 1
				<< ", loss: " << loss
				<< ", Y0: " << y0
				<< std::fixed << std::setprecision(3)
				<< ", elapsed time: " << elapsed_sec << "s\n";
		}

		auto train_data = bsde_->sample(net_config_.batch_size);
		train_step(train_data);
	}
}

void BSDESolver::train_step(const std::pair<torch::Tensor, torch::Tensor>& batch)
{
	model_->train();

	auto loss = loss_fn(batch, true);
	optimizer_->zero_grad();
	loss.backward();
	torch::nn::utils::clip_grad_norm_(model_->parameters_flattened(), 1.0);
	optimizer_->step();
	lr_scheduler_->step();
}
