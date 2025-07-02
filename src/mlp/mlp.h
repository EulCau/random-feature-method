#pragma once

#include "config.h"
#include <torch/torch.h>

class MLPImpl : public torch::nn::Module
{
public:
	MLPImpl(const Config& config);

	torch::Tensor forward(torch::Tensor x);

private:
	std::vector<torch::nn::BatchNorm1d> bn_layers_;
	std::vector<torch::nn::Linear> dense_layers_;
};

TORCH_MODULE(MLP);
