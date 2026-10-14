#ifndef __SD_MODEL_VAE_DC_AE_HPP__
#define __SD_MODEL_VAE_DC_AE_HPP__

#include <cinttypes>

#include "model/common/sana.hpp"
#include "model/vae/vae.hpp"

namespace Sana {
    struct DCAEConfig {
        int64_t latent_channels         = 32;
        int64_t head_dim                = 32;
        std::vector<int64_t> channels   = {128, 256, 512, 512, 1024, 1024};
        std::vector<int> layers         = {3, 3, 3, 3, 3, 3};
        std::vector<int> encoder_layers = {2, 2, 2, 3, 3, 3};
        bool has_encoder                = false;

        static DCAEConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            DCAEConfig config;
            config.has_encoder = weights.count(prefix + ".encoder.conv_in.weight") != 0;
            auto it            = weights.find(prefix + ".decoder.conv_in.weight");
            if (it != weights.end()) {
                config.latent_channels = it->second.ne[2];
                config.channels.back() = it->second.ne[3];
            }
            for (size_t i = 0; i < config.channels.size(); ++i) {
                const std::string stage = prefix + ".decoder.up_blocks." + std::to_string(i) + ".";
                int count               = 0;
                for (const auto& [name, tensor] : weights) {
                    if (!starts_with(name, stage)) {
                        continue;
                    }
                    count = std::max(count, atoi(name.substr(stage.size()).c_str()) + 1);
                    if (ends_with(name, ".conv1.weight")) {
                        config.channels[i] = tensor.ne[2];
                    } else if (ends_with(name, ".attn.to_q.weight")) {
                        config.channels[i] = tensor.ne[0];
                    }
                }
                if (count > 0) {
                    config.layers[i] = count - (i + 1 < config.channels.size() ? 1 : 0);
                }
                const std::string down = prefix + ".encoder.down_blocks." + std::to_string(i) + ".";
                count                  = 0;
                for (const auto& [name, tensor] : weights) {
                    if (starts_with(name, down)) {
                        count = std::max(count, atoi(name.substr(down.size()).c_str()) + 1);
                    }
                }
                if (count > 0) {
                    config.encoder_layers[i] = count - (i + 1 < config.channels.size() ? 1 : 0);
                }
            }
            if (it != weights.end()) {
                LOG_VERBOSE("dc-ae: latent_channels = %" PRId64 ", decoder_channels = %" PRId64,
                            config.latent_channels, config.channels.back());
            }
            return config;
        }
    };

    inline ggml_tensor* repeat_channels(ggml_context* ctx, ggml_tensor* x, int64_t repeats) {
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2], n = x->ne[3];
        x = ggml_reshape_4d(ctx, x, w * h, 1, c, n);
        x = ggml_repeat_4d(ctx, x, w * h, repeats, c, n);
        return ggml_reshape_4d(ctx, x, w, h, c * repeats, n);
    }

    inline ggml_tensor* pixel_shuffle(ggml_context* ctx, ggml_tensor* x) {
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2] / 4, n = x->ne[3];
        GGML_ASSERT(x->ne[2] == c * 4);
        x = ggml_reshape_4d(ctx, x, w, h, 2, 2 * c * n);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_4d(ctx, x, 2, w, h, 2 * c * n);
        x = ggml_reshape_4d(ctx, x, 2 * w, h, 2, c * n);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_4d(ctx, x, 2 * w, 2 * h, c, n);
    }

    inline ggml_tensor* pixel_unshuffle(ggml_context* ctx, ggml_tensor* x) {
        const int64_t w = x->ne[0] / 2, h = x->ne[1] / 2, c = x->ne[2], n = x->ne[3];
        GGML_ASSERT(x->ne[0] == 2 * w && x->ne[1] == 2 * h);
        x = ggml_reshape_4d(ctx, x, 2 * w, 2, h, c * n);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx, x, 2, w, h, 2 * c * n);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
        return ggml_reshape_4d(ctx, x, w, h, 4 * c, n);
    }

    inline ggml_tensor* average_channels(ggml_context* ctx, ggml_tensor* x, int64_t channels) {
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2], n = x->ne[3];
        GGML_ASSERT(c % channels == 0);
        x = ggml_reshape_4d(ctx, image_to_tokens(ctx, x), c / channels, channels, w * h, n);
        x = ggml_scale(ctx, ggml_sum_rows(ctx, x), static_cast<float>(channels) / c);
        return tokens_to_image(ctx, ggml_reshape_3d(ctx, x, channels, w * h, n), h, w);
    }

    class GroupedPointwise : public GGMLBlock {
        int64_t dim;
        int64_t groups;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, dim, groups);
        }

    public:
        GroupedPointwise(int64_t dim, int64_t groups) : dim(dim), groups(groups) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto g          = ctx->ggml_ctx;
            const int64_t h = x->ne[1], w = x->ne[0], n = x->ne[3];
            x = image_to_tokens(g, x);
            x = ggml_reshape_4d(g, x, dim, groups, h * w, n);
            x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
            x = ggml_mul_mat(g, params["weight"], x);
            x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
            return tokens_to_image(g, ggml_reshape_3d(g, x, dim * groups, h * w, n), h, w);
        }
    };

    class MultiscaleAttention : public GGMLBlock {
        int64_t channels;
        int64_t head_dim;

    public:
        MultiscaleAttention(int64_t channels, int64_t head_dim) : channels(channels), head_dim(head_dim) {
            for (const auto& name : {"to_q", "to_k", "to_v"}) {
                blocks[name] = std::make_shared<Linear>(channels, channels, false);
            }
            blocks["to_qkv_multiscale.0.proj_in"]  = std::make_shared<Conv2d_grouped>(3 * channels, 3 * channels, 3 * channels,
                                                                                      std::make_pair(5, 5), std::make_pair(1, 1),
                                                                                      std::make_pair(2, 2), std::make_pair(1, 1), false);
            blocks["to_qkv_multiscale.0.proj_out"] = std::make_shared<GroupedPointwise>(head_dim, 3 * channels / head_dim);
            blocks["to_out"]                       = std::make_shared<Linear>(channels * 2, channels, false);
            blocks["norm_out"]                     = std::make_shared<ChannelRMSNorm>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto g          = ctx->ggml_ctx;
            const int64_t h = x->ne[1], w = x->ne[0], n = x->ne[3];
            auto residual = x;
            auto tokens   = image_to_tokens(g, x);
            std::vector<ggml_tensor*> qkv;
            for (const auto& name : {"to_q", "to_k", "to_v"}) {
                qkv.push_back(std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, tokens));
            }
            auto projected      = tokens_to_image(g, ggml_ext_vec_concat(g, qkv, 0), h, w);
            auto scale          = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["to_qkv_multiscale.0.proj_in"])->forward(ctx, projected);
            scale               = std::dynamic_pointer_cast<GroupedPointwise>(blocks["to_qkv_multiscale.0.proj_out"])->forward(ctx, scale);
            projected           = ggml_concat(g, projected, scale, 2);
            const int64_t heads = 2 * channels / head_dim;
            projected           = ggml_reshape_4d(g, image_to_tokens(g, projected), 3 * head_dim, heads, h * w, n);
            projected           = ggml_cont(g, ggml_permute(g, projected, 0, 2, 1, 3));
            auto parts          = ggml_ext_chunk(g, projected, 3, 0);
            for (auto& part : parts) {
                part = ggml_cont(g, ggml_permute(g, part, 0, 2, 1, 3));
                part = ggml_reshape_3d(g, part, 2 * channels, h * w, n);
            }
            x = linear_attention(g, parts[0], parts[1], parts[2], heads);
            x = std::dynamic_pointer_cast<Linear>(blocks["to_out"])->forward(ctx, x);
            x = tokens_to_image(g, x, h, w);
            x = std::dynamic_pointer_cast<ChannelRMSNorm>(blocks["norm_out"])->forward(ctx, x);
            return ggml_add(g, residual, x);
        }
    };

    class DCAEResBlock : public GGMLBlock {
    public:
        DCAEResBlock(int64_t channels) {
            blocks["conv1"] = std::make_shared<Conv2d>(channels, channels, std::make_pair(3, 3),
                                                       std::make_pair(1, 1), std::make_pair(1, 1));
            blocks["conv2"] = std::make_shared<Conv2d>(channels, channels, std::make_pair(3, 3),
                                                       std::make_pair(1, 1), std::make_pair(1, 1), std::make_pair(1, 1), false);
            blocks["norm"]  = std::make_shared<ChannelRMSNorm>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto h = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"])->forward(ctx, x);
            h      = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"])->forward(ctx, ggml_silu(ctx->ggml_ctx, h));
            h      = std::dynamic_pointer_cast<ChannelRMSNorm>(blocks["norm"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    class EfficientViTBlock : public GGMLBlock {
    public:
        EfficientViTBlock(int64_t channels, int64_t head_dim) {
            blocks["attn"]     = std::make_shared<MultiscaleAttention>(channels, head_dim);
            blocks["conv_out"] = std::make_shared<GLUMBConv>(channels, channels * 4, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<MultiscaleAttention>(blocks["attn"])->forward(ctx, x);
            return std::dynamic_pointer_cast<GLUMBConv>(blocks["conv_out"])->forward(ctx, x);
        }
    };

    class DCAEUpBlock : public GGMLBlock {
        int64_t repeats;

    public:
        DCAEUpBlock(int64_t in_channels, int64_t out_channels) : repeats(out_channels * 4 / in_channels) {
            blocks["conv"] = std::make_shared<Conv2d>(in_channels, out_channels, std::make_pair(3, 3),
                                                      std::make_pair(1, 1), std::make_pair(1, 1));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto g        = ctx->ggml_ctx;
            auto residual = pixel_shuffle(g, repeat_channels(g, x, repeats));
            x             = ggml_upscale(g, x, 2, GGML_SCALE_MODE_NEAREST);
            x             = std::dynamic_pointer_cast<Conv2d>(blocks["conv"])->forward(ctx, x);
            return ggml_add(g, residual, x);
        }
    };

    class DCAEDecoder : public GGMLBlock {
        DCAEConfig config;

    public:
        DCAEDecoder(const DCAEConfig& config) : config(config) {
            blocks["conv_in"]  = std::make_shared<Conv2d>(config.latent_channels, config.channels.back(), std::make_pair(3, 3),
                                                          std::make_pair(1, 1), std::make_pair(1, 1));
            blocks["norm_out"] = std::make_shared<ChannelRMSNorm>(config.channels[0]);
            blocks["conv_out"] = std::make_shared<Conv2d>(config.channels[0], 3, std::make_pair(3, 3),
                                                          std::make_pair(1, 1), std::make_pair(1, 1));
            for (size_t i = 0; i < config.channels.size(); ++i) {
                const std::string stage = "up_blocks." + std::to_string(i) + ".";
                int start               = 0;
                if (i + 1 < config.channels.size()) {
                    blocks[stage + "0"] = std::make_shared<DCAEUpBlock>(config.channels[i + 1], config.channels[i]);
                    start               = 1;
                }
                for (int j = 0; j < config.layers[i]; ++j) {
                    if (i < 3) {
                        blocks[stage + std::to_string(j + start)] = std::make_shared<DCAEResBlock>(config.channels[i]);
                    } else {
                        blocks[stage + std::to_string(j + start)] = std::make_shared<EfficientViTBlock>(config.channels[i], config.head_dim);
                    }
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto g        = ctx->ggml_ctx;
            auto residual = repeat_channels(g, x, config.channels.back() / config.latent_channels);
            x             = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"])->forward(ctx, x);
            x             = ggml_add(g, x, residual);
            for (int i = static_cast<int>(config.channels.size()) - 1; i >= 0; --i) {
                const std::string stage = "up_blocks." + std::to_string(i) + ".";
                int start               = 0;
                if (i + 1 < static_cast<int>(config.channels.size())) {
                    x     = std::dynamic_pointer_cast<DCAEUpBlock>(blocks[stage + "0"])->forward(ctx, x);
                    start = 1;
                }
                for (int j = 0; j < config.layers[i]; ++j) {
                    auto block = blocks[stage + std::to_string(j + start)];
                    x          = i < 3 ? std::dynamic_pointer_cast<DCAEResBlock>(block)->forward(ctx, x)
                                       : std::dynamic_pointer_cast<EfficientViTBlock>(block)->forward(ctx, x);
                    sd::ggml_graph_cut::mark_graph_cut(x, "dc-ae.decoder." + stage + std::to_string(j + start), "hidden");
                }
            }
            x = std::dynamic_pointer_cast<ChannelRMSNorm>(blocks["norm_out"])->forward(ctx, x);
            return std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"])->forward(ctx, ggml_relu(g, x));
        }
    };

    class DCAEDownBlock : public GGMLBlock {
        int64_t out_channels;

    public:
        DCAEDownBlock(int64_t in_channels, int64_t out_channels) : out_channels(out_channels) {
            blocks["conv"] = std::make_shared<Conv2d>(in_channels, out_channels, std::make_pair(3, 3),
                                                      std::make_pair(2, 2), std::make_pair(1, 1));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto residual = average_channels(ctx->ggml_ctx, pixel_unshuffle(ctx->ggml_ctx, x), out_channels);
            return ggml_add(ctx->ggml_ctx, std::dynamic_pointer_cast<Conv2d>(blocks["conv"])->forward(ctx, x), residual);
        }
    };

    class DCAEEncoder : public GGMLBlock {
        DCAEConfig config;

    public:
        DCAEEncoder(const DCAEConfig& config) : config(config) {
            blocks["conv_in"]  = std::make_shared<Conv2d>(3, config.channels[0], std::make_pair(3, 3),
                                                          std::make_pair(1, 1), std::make_pair(1, 1));
            blocks["conv_out"] = std::make_shared<Conv2d>(config.channels.back(), config.latent_channels, std::make_pair(3, 3),
                                                          std::make_pair(1, 1), std::make_pair(1, 1));
            for (size_t i = 0; i < config.channels.size(); ++i) {
                const std::string stage = "down_blocks." + std::to_string(i) + ".";
                for (int j = 0; j < config.encoder_layers[i]; ++j) {
                    blocks[stage + std::to_string(j)] = i < 3
                                                            ? std::static_pointer_cast<GGMLBlock>(std::make_shared<DCAEResBlock>(config.channels[i]))
                                                            : std::static_pointer_cast<GGMLBlock>(std::make_shared<EfficientViTBlock>(config.channels[i], config.head_dim));
                }
                if (i + 1 < config.channels.size()) {
                    blocks[stage + std::to_string(config.encoder_layers[i])] = std::make_shared<DCAEDownBlock>(config.channels[i], config.channels[i + 1]);
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"])->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "dc-ae.encoder.conv_in", "hidden");
            for (size_t i = 0; i < config.channels.size(); ++i) {
                const std::string stage = "down_blocks." + std::to_string(i) + ".";
                for (int j = 0; j < config.encoder_layers[i]; ++j) {
                    auto block = blocks[stage + std::to_string(j)];
                    x          = i < 3 ? std::dynamic_pointer_cast<DCAEResBlock>(block)->forward(ctx, x)
                                       : std::dynamic_pointer_cast<EfficientViTBlock>(block)->forward(ctx, x);
                    sd::ggml_graph_cut::mark_graph_cut(x, "dc-ae.encoder." + stage + std::to_string(j), "hidden");
                }
                if (i + 1 < config.channels.size()) {
                    x = std::dynamic_pointer_cast<DCAEDownBlock>(blocks[stage + std::to_string(config.encoder_layers[i])])->forward(ctx, x);
                    sd::ggml_graph_cut::mark_graph_cut(x, "dc-ae.encoder." + stage + "downsample", "hidden");
                }
            }
            auto residual = average_channels(ctx->ggml_ctx, x, config.latent_channels);
            return ggml_add(ctx->ggml_ctx, std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"])->forward(ctx, x), residual);
        }
    };

    struct DCAERunner : public VAE {
        DCAEConfig config;
        DCAEDecoder decoder;
        DCAEEncoder encoder;

        DCAERunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : VAE(VERSION_SANA_SPRINT, backend, prefix, weight_manager),
              config(DCAEConfig::detect_from_weights(weights, prefix)),
              decoder(config),
              encoder(config) {
            decoder.init(params_ctx, weights, prefix + ".decoder");
            if (config.has_encoder) {
                encoder.init(params_ctx, weights, prefix + ".encoder");
            }
        }

        std::string get_desc() override { return "dc-ae"; }
        int get_encoder_output_channels(int) override { return static_cast<int>(config.latent_channels); }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            decoder.get_param_tensors(tensors, weight_prefix + ".decoder");
            if (config.has_encoder) {
                encoder.get_param_tensors(tensors, weight_prefix + ".encoder");
            }
        }

        sd::Tensor<float> _compute(int n_threads, const sd::Tensor<float>& z, bool decode_graph) override {
            if (!decode_graph && !config.has_encoder) {
                LOG_ERROR("Sana image-to-image requires a DC-AE GGUF with encoder weights");
                return {};
            }
            auto get_graph = [&]() {
                auto gf  = new_graph_custom(32768);
                auto ctx = get_context();
                auto out = decode_graph ? decoder.forward(&ctx, make_input(z)) : encoder.forward(&ctx, make_input(z));
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), z.dim());
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& output, std::shared_ptr<RNG>) override { return output; }
        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override { return latents / 0.41407f; }
        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override { return latents * 0.41407f; }
    };
}  // namespace Sana

#endif  // __SD_MODEL_VAE_DC_AE_HPP__
