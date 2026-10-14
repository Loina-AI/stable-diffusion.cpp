#ifndef __SD_MODEL_DIFFUSION_SANA_HPP__
#define __SD_MODEL_DIFFUSION_SANA_HPP__

#include <cinttypes>

#include "model/common/sana.hpp"
#include "model/diffusion/model.hpp"

namespace Sana {
    struct SanaConfig {
        int64_t channels    = 32;
        int64_t dim         = 1152;
        int64_t caption_dim = 2304;
        int64_t head_dim    = 32;
        int64_t cross_heads = 16;
        int64_t ffn_dim     = 2880;
        int layers          = 28;

        static SanaConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            SanaConfig config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = weights.find(prefix + "." + suffix);
                return it == weights.end() ? nullptr : &it->second;
            };
            if (auto w = find("patch_embed.proj.weight")) {
                config.dim      = w->ne[1];
                config.channels = w->ne[0];
            }
            if (auto w = find("caption_projection.linear_1.weight")) {
                config.caption_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.ff.conv_inverted.weight")) {
                config.ffn_dim = w->ne[1] / 2;
            }
            int layers                     = 0;
            const std::string block_prefix = prefix + ".transformer_blocks.";
            for (const auto& [name, _] : weights) {
                if (starts_with(name, block_prefix)) {
                    layers = std::max(layers, atoi(name.substr(block_prefix.size()).c_str()) + 1);
                }
            }
            if (layers > 0) {
                config.layers = layers;
                LOG_VERBOSE("sana sprint: layers = %d, hidden_size = %" PRId64, config.layers, config.dim);
            }
            return config;
        }
    };

    class TimestepEmbedding : public GGMLBlock {
    public:
        TimestepEmbedding(int64_t dim) {
            blocks["linear_1"] = std::make_shared<PaddedLinear>(256, dim);
            blocks["linear_2"] = std::make_shared<PaddedLinear>(dim, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto x = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, 256, 10000);
            x      = std::dynamic_pointer_cast<PaddedLinear>(blocks["linear_1"])->forward(ctx, x);
            return std::dynamic_pointer_cast<PaddedLinear>(blocks["linear_2"])->forward(ctx, ggml_silu(ctx->ggml_ctx, x));
        }
    };

    class Attention : public GGMLBlock {
        int64_t heads;
        bool self;

    public:
        Attention(int64_t dim, int64_t heads, bool self) : heads(heads), self(self) {
            for (const auto& name : {"to_q", "to_k", "to_v"}) {
                blocks[name] = std::make_shared<PaddedLinear>(dim, dim, !self);
            }
            blocks["to_out.0"] = std::make_shared<PaddedLinear>(dim, dim);
            blocks["norm_q"]   = std::make_shared<RMSNorm>(dim, 1e-5f);
            blocks["norm_k"]   = std::make_shared<RMSNorm>(dim, 1e-5f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, ggml_tensor* mask) {
            auto q   = std::dynamic_pointer_cast<PaddedLinear>(blocks["to_q"])->forward(ctx, x);
            auto k   = std::dynamic_pointer_cast<PaddedLinear>(blocks["to_k"])->forward(ctx, self ? x : context);
            auto v   = std::dynamic_pointer_cast<PaddedLinear>(blocks["to_v"])->forward(ctx, self ? x : context);
            q        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"])->forward(ctx, q);
            k        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"])->forward(ctx, k);
            auto out = self ? linear_attention(ctx->ggml_ctx, q, k, v, heads)
                            : ggml_ext_attention_ext(ctx, q, k, v, heads, mask, false, ctx->flash_attn_enabled);
            return std::dynamic_pointer_cast<PaddedLinear>(blocks["to_out.0"])->forward(ctx, out);
        }
    };

    inline ggml_tensor* modulated_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* shift, ggml_tensor* scale) {
        x = ggml_norm(ctx, x, 1e-6f);
        return ggml_add(ctx, ggml_mul(ctx, x, ggml_scale_bias(ctx, scale, 1.f, 1.f)), shift);
    }

    class TransformerBlock : public GGMLBlock {
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 6);
        }

    public:
        TransformerBlock(const SanaConfig& config) : dim(config.dim) {
            blocks["attn1"] = std::make_shared<Attention>(dim, dim / config.head_dim, true);
            blocks["attn2"] = std::make_shared<Attention>(dim, config.cross_heads, false);
            blocks["ff"]    = std::make_shared<GLUMBConv>(dim, config.ffn_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* t, ggml_tensor* context, ggml_tensor* mask, int64_t height, int64_t width) {
            auto g     = ctx->ggml_ctx;
            auto table = ggml_add(g, params["scale_shift_table"], ggml_reshape_2d(g, t, dim, 6));
            auto parts = ggml_ext_chunk(g, table, 6, 1);
            auto h     = modulated_norm(g, x, parts[0], parts[1]);
            h          = std::dynamic_pointer_cast<Attention>(blocks["attn1"])->forward(ctx, h, nullptr, nullptr);
            x          = ggml_add(g, x, ggml_mul(g, h, parts[2]));
            h          = std::dynamic_pointer_cast<Attention>(blocks["attn2"])->forward(ctx, x, context, mask);
            x          = ggml_add(g, x, h);
            h          = tokens_to_image(g, modulated_norm(g, x, parts[3], parts[4]), height, width);
            h          = std::dynamic_pointer_cast<GLUMBConv>(blocks["ff"])->forward(ctx, h);
            return ggml_add(g, x, ggml_mul(g, image_to_tokens(g, h), parts[5]));
        }
    };

    class SanaModel : public GGMLBlock {
        SanaConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.dim, 2);
        }

    public:
        SanaModel(const SanaConfig& config) : config(config) {
            blocks["patch_embed.proj"]             = std::make_shared<PaddedLinear>(config.channels, config.dim);
            blocks["time_embed.timestep_embedder"] = std::make_shared<TimestepEmbedding>(config.dim);
            blocks["time_embed.guidance_embedder"] = std::make_shared<TimestepEmbedding>(config.dim);
            blocks["time_embed.linear"]            = std::make_shared<PaddedLinear>(config.dim, config.dim * 6);
            blocks["caption_projection.linear_1"]  = std::make_shared<PaddedLinear>(config.caption_dim, config.dim);
            blocks["caption_projection.linear_2"]  = std::make_shared<PaddedLinear>(config.dim, config.dim);
            blocks["caption_norm"]                 = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            blocks["proj_out"]                     = std::make_shared<PaddedLinear>(config.dim, config.channels);
            for (int i = 0; i < config.layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(config);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* guidance, ggml_tensor* context, ggml_tensor* mask) {
            auto g               = ctx->ggml_ctx;
            const int64_t height = x->ne[1], width = x->ne[0];
            x             = std::dynamic_pointer_cast<PaddedLinear>(blocks["patch_embed.proj"])->forward(ctx, image_to_tokens(g, x));
            auto t        = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_embed.timestep_embedder"])->forward(ctx, timestep);
            auto gd       = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_embed.guidance_embedder"])->forward(ctx, ggml_scale(g, guidance, 0.1f));
            auto embedded = ggml_add(g, t, gd);
            t             = std::dynamic_pointer_cast<PaddedLinear>(blocks["time_embed.linear"])->forward(ctx, ggml_silu(g, embedded));
            context       = std::dynamic_pointer_cast<PaddedLinear>(blocks["caption_projection.linear_1"])->forward(ctx, context);
            context       = ggml_gelu(g, context);
            context       = std::dynamic_pointer_cast<PaddedLinear>(blocks["caption_projection.linear_2"])->forward(ctx, context);
            context       = std::dynamic_pointer_cast<RMSNorm>(blocks["caption_norm"])->forward(ctx, context);
            sd::ggml_graph_cut::mark_graph_cut(x, "sana.prepare", "hidden");
            sd::ggml_graph_cut::mark_graph_cut(t, "sana.prepare", "time");
            sd::ggml_graph_cut::mark_graph_cut(embedded, "sana.prepare", "embedded");
            sd::ggml_graph_cut::mark_graph_cut(context, "sana.prepare", "context");
            for (int i = 0; i < config.layers; ++i) {
                x = std::dynamic_pointer_cast<TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)])
                        ->forward(ctx, x, t, context, mask, height, width);
                sd::ggml_graph_cut::mark_graph_cut(x, "sana.transformer_blocks." + std::to_string(i), "hidden");
            }
            auto table = ggml_add(g, params["scale_shift_table"], embedded);
            auto parts = ggml_ext_chunk(g, table, 2, 1);
            x          = modulated_norm(g, x, parts[0], parts[1]);
            x          = std::dynamic_pointer_cast<PaddedLinear>(blocks["proj_out"])->forward(ctx, x);
            return tokens_to_image(g, x, height, width);
        }
    };

    struct SanaRunner : public DiffusionModelRunner {
        SanaConfig config;
        SanaModel model;

        SanaRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(SanaConfig::detect_from_weights(weights, prefix)),
              model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "sana sprint"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& p) override {
            GGML_ASSERT(p.x && p.timesteps && p.context);
            GGML_ASSERT(p.x->shape()[3] == 1);
            auto extra     = std::get_if<FluxDiffusionExtra>(&p.extra);
            auto guidance  = extra && extra->guidance ? *extra->guidance : sd::Tensor<float>({1}, std::vector<float>{4.5f});
            auto get_graph = [&]() {
                auto gf   = new_graph_custom(32768);
                auto ctx  = get_context();
                auto mask = p.y ? make_input(*p.y) : nullptr;
                if (mask) {
                    mask = ggml_reshape_4d(compute_ctx, mask, mask->ne[0], 1, 1, 1);
                }
                auto out = model.forward(&ctx, make_input(*p.x), make_input(*p.timesteps),
                                         make_input(guidance), make_input(*p.context), mask);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), p.x->dim());
        }
    };
}  // namespace Sana

#endif  // __SD_MODEL_DIFFUSION_SANA_HPP__
