#pragma once

#include "config.h"
#include <torch/torch.h>
#include <vector>

// phi_h(t, x) = tanh(
//     s_h * (q_h^T (x / space_scale) + gamma_h * t / total_time) + c_h)
class RandomFeatureFunction {
public:
    RandomFeatureFunction(
        int64_t dim,
        int64_t hidden_dim,
        float total_time,
        torch::Device device,
        uint64_t seed = 42,
        const RandomFeatureOptions& options = RandomFeatureOptions{
            0.5f, 2.0f, 1.0f, 1.0f, 1.0f, {}});

    // Resample the fixed inner parameters (q, s, gamma, c).
    void resample_params(uint64_t seed);

    // t: (1 or B, T, 1, 1), x: (B, T, 1, d)
    // Return phi with shape (B, T, H, 1).
    [[nodiscard]] torch::Tensor phi(const torch::Tensor& t, const torch::Tensor& x) const;

    // Return beta0 + sum_h beta_h phi_h with shape (B, T, 1, 1).
    [[nodiscard]] torch::Tensor value(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& beta0,
        const torch::Tensor& beta) const;

    // Return d phi_h / d x_j with shape (B, T, H, d).
    [[nodiscard]] torch::Tensor spatial_gradient_features(
        const torch::Tensor& t,
        const torch::Tensor& x) const;

    // Return grad_x sum_h beta_h phi_h with shape (B, T, 1, d).
    [[nodiscard]] torch::Tensor spatial_gradient(
        const torch::Tensor& t,
        const torch::Tensor& x,
        const torch::Tensor& beta) const;

    [[nodiscard]] int64_t dim() const { return dim_; }
    [[nodiscard]] int64_t hidden_dim() const { return hidden_; }

    [[nodiscard]] uint64_t seed() const { return seed_; }
    [[nodiscard]] const torch::Tensor& q() const { return q_; }
    [[nodiscard]] const torch::Tensor& scales() const { return scales_; }
    [[nodiscard]] const torch::Tensor& gamma() const { return gamma_; }
    [[nodiscard]] const torch::Tensor& c() const { return c_; }

protected:
    int64_t dim_;
    int64_t hidden_;
    float total_time_;
    float space_scale_;
    float time_scale_;
    float bias_scale_;
    std::vector<RandomFeatureScaleBand> scale_bands_;
    uint64_t seed_;
    torch::Device device_;
    torch::Tensor q_;      // (d, H), unit direction in each column
    torch::Tensor scales_; // (1, H), log-uniform frequency scale
    torch::Tensor gamma_;  // (1, H), time direction
    torch::Tensor c_;      // (1, H), bias
};
