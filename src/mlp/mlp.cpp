#include "mlp.h"

MLPImpl::MLPImpl(const Config& config)
{
    int64_t input_dim = config.eqn_config.dim;
    int64_t output_dim = config.eqn_config.dim;
	std::vector<int64_t> num_hiddens = config.net_config.num_hiddens;

    bn_layers_.push_back(register_module("bn_input", torch::nn::BatchNorm1d(input_dim)));

    for (size_t i = 0; i < num_hiddens.size(); ++i)
    {
        int64_t hidden = num_hiddens[i];

        auto linear = torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden).bias(false));
        dense_layers_.push_back(register_module("dense_" + std::to_string(i), linear));

        auto bn = torch::nn::BatchNorm1d(hidden);
        bn_layers_.push_back(register_module("bn_" + std::to_string(i), bn));

        input_dim = hidden;
    }

    dense_layers_.push_back(register_module("dense_final", torch::nn::Linear(input_dim, output_dim)));
    bn_layers_.push_back(register_module("bn_final", torch::nn::BatchNorm1d(output_dim)));
}

torch::Tensor MLPImpl::forward(torch::Tensor x)
{
    x = bn_layers_[0]->forward(x);

    for (int64_t i = 0; i < dense_layers_.size() - 1; ++i)
    {
        x = dense_layers_[i]->forward(x);
        x = bn_layers_[i + 1]->forward(x);
        x = torch::relu(x);
    }

    x = dense_layers_.back()->forward(x);
    x = bn_layers_.back()->forward(x);

    return x;
}
