#pragma once

#include "config.h"
#include "equation.h"
#include "non_shared_model.h"
#include <memory>
#include <torch/torch.h>
#include "lr_scheduler_utils.h"

class BSDESolver
{
public:
	BSDESolver(const Config& config, const std::shared_ptr<Equation>& bsde);

	void train();

private:
	torch::Tensor loss_fn(const std::pair<torch::Tensor, torch::Tensor>& inputs, bool training);
	void train_step(const std::pair<torch::Tensor, torch::Tensor>& batch);

	EqnConfig eqn_config_;
	NetConfig net_config_;
	std::shared_ptr<Equation> bsde_;
	NonSharedModel model_;
	std::unique_ptr<torch::optim::Optimizer> optimizer_;
	std::unique_ptr<LambdaLRScheduler> lr_scheduler_;
};
