#ifndef __SD_MODEL_COMMON_SANA_HPP__
#define __SD_MODEL_COMMON_SANA_HPP__

#include "model/common/ggml_block.hpp"

namespace Sana {
    class PaddedLinear : public ::Linear {
        int64_t input_features;

        void init_params(ggml_context* ctx, const String2TensorStorage& weights, const std::string prefix) override {
            auto it = weights.find(prefix + "weight");
            if (it != weights.end()) {
                const auto& w = it->second;
                GGML_ASSERT(w.ne[1] == out_features);
                GGML_ASSERT(w.ne[0] == input_features || w.ne[0] == ((input_features + 255) / 256) * 256);
                in_features = w.ne[0];
                // BF16 dot kernels also round activations; keep the floating reference in F32.
                force_f32 = w.type == GGML_TYPE_BF16;
            }
            ::Linear::init_params(ctx, weights, prefix);
        }

    public:
        PaddedLinear(int64_t in_features, int64_t out_features, bool bias = true)
            : ::Linear(in_features, out_features, bias), input_features(in_features) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(x->ne[0] == input_features);
            // Zero columns align Sana's 1152/2880-wide matrices to K-quant blocks.
            if (in_features != input_features) {
                x = ggml_pad(ctx->ggml_ctx, x, static_cast<int>(in_features - input_features), 0, 0, 0);
            }
            return ::Linear::forward(ctx, x);
        }
    };

    inline ggml_tensor* image_to_tokens(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
        return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    }

    inline ggml_tensor* tokens_to_image(ggml_context* ctx, ggml_tensor* x, int64_t height, int64_t width) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        return ggml_reshape_4d(ctx, x, width, height, x->ne[1], x->ne[2]);
    }

    inline ggml_tensor* linear_attention(ggml_context* ctx,
                                         ggml_tensor* q,
                                         ggml_tensor* k,
                                         ggml_tensor* v,
                                         int64_t heads) {
        const int64_t dim    = q->ne[0];
        const int64_t tokens = q->ne[1];
        const int64_t batch  = q->ne[2];
        auto split_heads     = [&](ggml_tensor* x) {
            x = ggml_reshape_4d(ctx, x, dim / heads, heads, tokens, batch);
            return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        };
        q       = ggml_relu(ctx, split_heads(q));
        k       = ggml_relu(ctx, split_heads(k));
        v       = split_heads(v);
        auto kt = ggml_cont(ctx, ggml_transpose(ctx, k));
        auto vt = ggml_cont(ctx, ggml_transpose(ctx, v));

        // Associate V K^T Q before normalization; never allocate a token-by-token matrix.
        auto kv = ggml_mul_mat(ctx, kt, vt);
        ggml_prec_set_acc(kv, GGML_PREC_F32);
        auto out = ggml_mul_mat(ctx, kv, q);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        auto sum         = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, kt)));
        auto denominator = ggml_mul_mat(ctx, sum, q);
        ggml_prec_set_acc(denominator, GGML_PREC_F32);
        out = ggml_div(ctx, out, ggml_scale_bias(ctx, denominator, 1.f, 1e-15f));
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, out, dim, tokens, batch);
    }

    class ChannelRMSNorm : public GGMLBlock {
        int64_t channels;
        bool bias;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

    public:
        ChannelRMSNorm(int64_t channels, bool bias = true) : channels(channels), bias(bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            const int64_t h = x->ne[1], w = x->ne[0];
            x = image_to_tokens(ctx->ggml_ctx, x);
            x = ggml_mul(ctx->ggml_ctx, ggml_rms_norm(ctx->ggml_ctx, x, 1e-5f), params["weight"]);
            if (bias) {
                x = ggml_add(ctx->ggml_ctx, x, params["bias"]);
            }
            return tokens_to_image(ctx->ggml_ctx, x, h, w);
        }
    };

    class GLUMBConv : public GGMLBlock {
        bool residual;
        bool norm;

    public:
        GLUMBConv(int64_t channels, int64_t hidden, bool residual = false, bool norm = false)
            : residual(residual), norm(norm) {
            blocks["conv_inverted"] = std::make_shared<PaddedLinear>(channels, hidden * 2);
            blocks["conv_depth"]    = std::make_shared<Conv2d_grouped>(hidden * 2, hidden * 2, hidden * 2,
                                                                       std::make_pair(3, 3), std::make_pair(1, 1), std::make_pair(1, 1));
            blocks["conv_point"]    = std::make_shared<PaddedLinear>(hidden, channels, false);
            if (norm) {
                blocks["norm"] = std::make_shared<ChannelRMSNorm>(channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto g          = ctx->ggml_ctx;
            auto original   = x;
            const int64_t h = x->ne[1], w = x->ne[0];
            x          = std::dynamic_pointer_cast<PaddedLinear>(blocks["conv_inverted"])->forward(ctx, image_to_tokens(g, x));
            x          = tokens_to_image(g, ggml_silu(g, x), h, w);
            x          = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv_depth"])->forward(ctx, x);
            auto parts = ggml_ext_chunk(g, x, 2, 2);
            x          = ggml_mul(g, parts[0], ggml_silu(g, parts[1]));
            x          = std::dynamic_pointer_cast<PaddedLinear>(blocks["conv_point"])->forward(ctx, image_to_tokens(g, x));
            x          = tokens_to_image(g, x, h, w);
            if (norm) {
                x = std::dynamic_pointer_cast<ChannelRMSNorm>(blocks["norm"])->forward(ctx, x);
            }
            return residual ? ggml_add(g, original, x) : x;
        }
    };
}  // namespace Sana

#endif  // __SD_MODEL_COMMON_SANA_HPP__
