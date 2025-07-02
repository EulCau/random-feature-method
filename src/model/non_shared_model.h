#pragma once

#include "config.h"
#include "equation.h"
#include "mlp.h"
#include <memory>
#include <torch/torch.h>
#include <vector>

class NonSharedModelImpl : public torch::nn::Module {
public:
	NonSharedModelImpl(const Config& config, const std::shared_ptr<Equation>& bsde);

	torch::Tensor forward(const std::pair<torch::Tensor, torch::Tensor>& inputs, bool training);

	torch::Tensor y_init() const { return y_init_; }

	std::vector<torch::Tensor> parameters_flattened() const;

private:
	EqnConfig eqn_config_;
	NetConfig net_config_;
	std::shared_ptr<Equation> bsde_;

	torch::Tensor y_init_;
	torch::Tensor z_init_;

	std::vector<MLP> subnets_;
};

TORCH_MODULE(NonSharedModel);
