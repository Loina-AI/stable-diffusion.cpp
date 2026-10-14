# Sana Sprint

[Sana Sprint 0.6B 1024px](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)
supports text-to-image and image-to-image generation with Gemma 2, the Sana linear-attention
transformer, DC-AE, and the SCM trigflow sampler. Use 1–4 steps;
the default two-step schedule uses angles `[1.57080, 1.3, 0]`.

Use the original
[Gemma 2 Q4_K_M GGUF](https://huggingface.co/BafS/gemma-2-2b-it-Q4_K_M-GGUF/blob/main/gemma-2-2b-it-q4_k_m.gguf)
with an external `tokenizer/tokenizer.json` from the Sana repository. Gemma
embeddings use the final normalized hidden state, the complex instruction
prefix, and the BOS token plus the last 299 tokens. Padding is masked in both
the text encoder and cross-attention.

```sh
./build/bin/sd-cli \
  --diffusion-model models/sana-sprint/source/transformer/diffusion_pytorch_model.safetensors \
  --llm models/sana-sprint/gemma-2-2b-it-q4_k_m.gguf \
  --tokenizer models/sana-sprint/tokenizer/tokenizer.json \
  --vae models/sana-sprint/source/vae/diffusion_pytorch_model.safetensors \
  --sampling-method scm --cfg-scale 1 --guidance 4.5 \
  --steps 2 --width 1024 --height 1024 \
  -p "a tiny astronaut hatching from an egg on the moon" -o sana.png
```

Image dimensions must be multiples of 32. `--guidance` controls the distilled
guidance embedding. Sana Sprint disables external CFG and ignores negative
prompts, matching Diffusers. The DC-AE uses 32 latent channels, 32× spatial compression, and a
scaling factor of `0.41407`. ControlNet and instruction-based editing are not implemented.
`--vae-tiling` reduces encoder and decoder memory, with the usual changes to global
attention at tile boundaries.

For image-to-image, add `-i input.png --strength 0.75` to the command. Use a
DC-AE file containing both encoder and decoder weights; a decoder-only file
cannot encode an input image. No additional denoiser or text encoder is needed.
Strength chooses the starting trigflow angle continuously, then uses the
remaining lower angles of the Sprint schedule. Lower strength preserves more
of the input; 1 starts from full noise, and 0 returns the resized input without
inference. This is ordinary img2img resynthesis, not an instruction-trained
editing model.

For custom schedules, pass angles through `--sigmas`, or set
`sana_max_timestep` and `sana_intermediate_timestep` in `--extra-sample-args`.
The intermediate angle is used only for two steps.

## Compact package

Download the transformer and VAE weights into `source/transformer/` and
`source/vae/`, the tokenizer into `tokenizer/`, and the Gemma GGUF into the
model directory. Then run:

```sh
python3 scripts/prepare_sana.py models/sana-sprint --sd-cli build/bin/sd-cli
models/sana-sprint/package/generate.sh -p "a red fox in a snowy forest" -o fox.png
```

The script uses Python's standard library and `sd-cli`. It extracts the
DC-AE encoder and decoder, converts them to FP16, and quantizes the transformer blocks'
linear and pointwise projections to Q8_0. Normalization stays in FP32;
timestep, guidance, caption and input/output projections retain their source
floating precision. BF16 linear weights use F32 arithmetic at runtime to
preserve activation precision. Depthwise convolutions retain their original
precision and use F16 convolution kernels. The source downloads
stay outside the distributable `package/` directory. `manifest.json` records
the measured file sizes; the total includes the CLI and tokenizer.

The specified Gemma file is about 1.709 GB. The default Q8_0 denoiser is
0.647 GB, and full FP16 DC-AE is 0.625 GB. Together with the tokenizer,
the model artifacts total 3.015 GB; the standalone CPU package is about
3.056 GB including its CLI. `--decoder-only` produces the smaller text-to-image
package (about 2.750 GB including the CLI).

For a smaller DC-AE, add `--vae-type q8_0`. Only its attention and pointwise
matrices become Q8; spatial/grouped convolutions and input/output layers remain
F16, and normalization remains F32. The full file is 399.5 MB, reducing model
artifacts to 2.790 GB (2.831 GB with the CLI). On 64px checks, encoder latent
relative RMS error was 2%, decoder PSNR against PyTorch was 55.9 dB, and full
image reconstruction PSNR relative to FP16 was 38.9 dB. FP16 remains the default
for input-image fidelity.

Use `--all-types` to also prepare BF16, Q8_0, Q6_K, Q5_K_M and Q4_K_M files
in `gguf/`, or `--dit-type` to choose a package variant. The K-quant recipes
use Q8_0 for the first two and last two blocks and cross-attention key/value
projections, and Q6_K for self-attention query/key/value and feed-forward
output projections. Remaining matrices use the variant's main K type.
The M recipes are specific to Sana.

K-quants need 256-column blocks. The converter adds zero columns to Sana's
1152/2880-column projections, and `Sana::PaddedLinear` pads their activations
to match. Unquantized outputs were identical before and after padding. These
K-quant GGUFs require a runtime containing the padded Sana implementation.
The model dimensions continue to describe the original, unpadded network.

The [published variants](https://huggingface.co/Ninnix96/Sana-Sprint-0.6B-1024px-GGUF)
include hashes, tensor types and diagnostic measurements. Q8_0 is the default
because it had the lowest numerical error and fastest CPU execution among
the quantized candidates, while adding only 68 MB over Q6_K. Those checks
used three small latent inputs with precomputed random embeddings; they do
not establish perceptual quality or mobile performance.

Pretrained checkpoint comparisons against PyTorch produced a denoiser RMS error of
0.00613 (cosine similarity 0.999982) and an FP16 decoder RMS error of 0.000148
(cosine similarity 0.9999998). The convolution kernels use FP16 arithmetic.
The FP16 encoder matched a 64×64 PyTorch reference with relative RMS error
0.000836. SCM matched Diffusers within `1.2e-7` for 1, 2, and 4 steps. A complete
128×128 CPU smoke test completed. Img2img at strength 0, 0.25, 0.75 and 1
passed CPU and Vulkan pipeline checks. A 1024px Vulkan render stopped because
available shared memory was below the runtime requirement; 1024px image
quality and mobile performance have not been validated.

For integration checks with precomputed embeddings, call `Sana::SanaRunner`
with `DiffusionParams.context` shaped `[2304, tokens, 1]`, `y` containing an
additive mask (`0` to keep, `-10000` to discard), and `FluxDiffusionExtra.guidance`
containing the unscaled guidance value. The runner applies its `0.1` scale.
The transformer timestep is `sin(angle) / (cos(angle) + sin(angle))`, rather
than the sampler angle itself. Decode raw latents separately with
`Sana::DCAERunner` after dividing by `0.41407`.
