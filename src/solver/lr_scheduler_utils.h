#pragma once

#include <torch/torch.h>
#include <cmath>
#include <functional>
#include <algorithm>
#include <memory>
#include <iostream>

inline std::function<float(int64_t)> create_lr_lambda(
	int64_t warmup_steps,
	int64_t total_steps)
{
	return [=](int64_t step)
		{
			if (step < warmup_steps)
			{
				// Linear warmup
				return static_cast<float>(step) / std::max<int64_t>(warmup_steps, 1);
			}
			else
			{
				// Cosine decay
				float progress = static_cast<float>(step - warmup_steps) / std::max<int64_t>(total_steps - warmup_steps, 1);
				progress = std::min(progress, 1.0f);
				return 0.5f * (1.0f + std::cos(float(M_PI) * progress));
			}
		};
}

class LambdaLRScheduler
{
public:
	LambdaLRScheduler(torch::optim::Optimizer& optimizer, std::function<float(int64_t)> lr_lambda)
		: optimizer_(optimizer), lr_lambda_(lr_lambda), step_count_(0)
	{
		// Save base_lr
		for (auto& group : optimizer_.param_groups())
		{
			float base_lr = static_cast<torch::optim::AdamWOptions&>(group.options()).lr();
			base_lrs_.push_back(base_lr);
		}
	}

	void step()
	{
		float factor = lr_lambda_(step_count_);
		for (size_t i = 0; i < optimizer_.param_groups().size(); ++i)
		{
			auto& group = optimizer_.param_groups()[i];
			auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());

			// Update lr
			options.lr(base_lrs_[i] * factor);
		}
		++step_count_;
	}

	int64_t step_count() const { return step_count_; }

private:
	torch::optim::Optimizer& optimizer_;
	std::function<float(int64_t)> lr_lambda_;
	int64_t step_count_;
	std::vector<float> base_lrs_;  // ¹Ø¼ü²¹³ä
};
