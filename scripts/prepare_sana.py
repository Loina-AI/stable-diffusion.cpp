#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


DIT_TYPES = ("bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q5_0", "q4_0")


def tensor_type(name, variant, layers):
    if "norm" in name and name.endswith(".weight"):
        return "f32"
    block = re.search(r"transformer_blocks\.(\d+)\.", name)
    if not block or not name.endswith(".weight") or ".conv_depth." in name:
        return "bf16"
    if variant not in ("q6_k", "q5_k_m", "q4_k_m"):
        return variant
    index = int(block[1])
    if index < 2 or index >= layers - 2 or ".attn2.to_k." in name or ".attn2.to_v." in name:
        return "q8_0"
    if any(part in name for part in (".attn1.to_q.", ".attn1.to_k.", ".attn1.to_v.", ".ff.conv_point.")):
        return "q6_k"
    return variant.removesuffix("_m")


def prepare_denoiser(source, destination, variant, cli):
    with source.open("rb") as src:
        size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(size))
        layers = max(int(m[1]) for name in header if (m := re.search(r"transformer_blocks\.(\d+)\.", name))) + 1
        tensors, ranges, rules = {}, [], []
        offset = 0
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            kind = tensor_type(name, variant, layers)
            rules.append(re.escape(name) + "$=" + kind.replace("_k", "_K"))
            shape = list(tensor["shape"])
            if len(shape) == 4 and shape[2:] == [1, 1] and name.endswith((".conv_inverted.weight", ".conv_point.weight")):
                shape = shape[:2]
            begin, end = tensor["data_offsets"]
            row, padding = 0, 0
            if len(shape) == 2 and kind in ("q4_k", "q5_k", "q6_k"):
                width = {"BF16": 2, "F16": 2, "F32": 4}[tensor["dtype"]]
                row = shape[1] * width
                aligned = (shape[1] + 255) // 256 * 256
                padding = (aligned - shape[1]) * width
                shape[1] = aligned
            length = end - begin + (shape[0] * padding if padding else 0)
            tensors[name] = {"dtype": tensor["dtype"], "shape": shape, "data_offsets": [offset, offset + length]}
            ranges.append((8 + size + begin, end - begin, row, padding))
            offset += length
        with tempfile.TemporaryDirectory(prefix="sana-quant-", dir=destination.parent) as work:
            padded = Path(work) / "denoiser.safetensors"
            data = json.dumps(tensors, separators=(",", ":")).encode()
            data += b" " * (-len(data) % 8)
            with padded.open("wb") as dst:
                dst.write(struct.pack("<Q", len(data)))
                dst.write(data)
                for start, length, row, padding in ranges:
                    src.seek(start)
                    while length:
                        chunk_size = max(row, 1024 * 1024 // row * row) if padding else 1024 * 1024
                        chunk = src.read(min(length, chunk_size))
                        if not chunk:
                            raise ValueError("Truncated Sana weights")
                        if padding:
                            for i in range(0, len(chunk), row):
                                dst.write(chunk[i:i + row])
                                dst.write(bytes(padding))
                        else:
                            dst.write(chunk)
                        length -= len(chunk)
            subprocess.run([str(cli), "-M", "convert", "--diffusion-model", str(padded),
                            "--convert-name", "--type", "bf16", "--tensor-type-rules", ",".join(rules),
                            "--threads", "2", "-o", str(Path(work) / "denoiser.gguf")], check=True)
            os.replace(Path(work) / "denoiser.gguf", destination)


def extract_vae(source, destination, decoder_only=False):
    with source.open("rb") as src:
        size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(size))
        tensors = {}
        ranges = []
        offset = 0
        for name, tensor in header.items():
            if not name.startswith("decoder.") and (decoder_only or not name.startswith("encoder.")):
                continue
            begin, end = tensor["data_offsets"]
            shape = tensor["shape"]
            if len(shape) == 4 and shape[2:] == [1, 1]:
                if ".attn.to_qkv_multiscale." in name and name.endswith(".proj_out.weight"):
                    shape = [shape[0] // shape[1], shape[1], shape[1]]
                elif name.endswith((".conv_inverted.weight", ".conv_point.weight")):
                    shape = shape[:2]
            tensors["first_stage_model." + name] = {
                "dtype": tensor["dtype"], "shape": shape,
                "data_offsets": [offset, offset + end - begin],
            }
            ranges.append((8 + size + begin, end - begin))
            offset += end - begin
        if not tensors:
            raise ValueError("No DC-AE tensors found")
        data = json.dumps(tensors, separators=(",", ":")).encode()
        data += b" " * (-len(data) % 8)
        with destination.open("wb") as dst:
            dst.write(struct.pack("<Q", len(data)))
            dst.write(data)
            for start, length in ranges:
                src.seek(start)
                while length:
                    chunk = src.read(min(length, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Truncated DC-AE weights")
                    dst.write(chunk)
                    length -= len(chunk)


def prepare_vae(source, destination, variant, cli, decoder_only=False):
    with tempfile.TemporaryDirectory(prefix="sana-vae-", dir=destination.parent) as work:
        weights = Path(work) / "vae.safetensors"
        extract_vae(source, weights, decoder_only)
        rules = []
        if variant == "q8_0":
            with weights.open("rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            for name, tensor in header.items():
                if len(tensor["shape"]) == 2 and name.endswith(".weight"):
                    rules.append(re.escape(name) + "$=q8_0")
        command = [str(cli), "-M", "convert", "-m", str(weights), "--type", "f16", "--threads", "2"]
        if rules:
            command += ["--tensor-type-rules", ",".join(rules)]
        subprocess.run(command + ["-o", str(Path(work) / "vae.gguf")], check=True)
        os.replace(Path(work) / "vae.gguf", destination)


def copy_file(source, destination):
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description="Prepare a Sana Sprint text-to-image and img2img package")
    parser.add_argument("model_dir", type=Path, help="directory containing source/, tokenizer/, and the Gemma GGUF")
    parser.add_argument("--sd-cli", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dit-type", choices=DIT_TYPES, default="q8_0")
    parser.add_argument("--all-types", action="store_true", help="also prepare BF16, Q8_0, Q6_K, Q5_K_M and Q4_K_M in gguf/")
    parser.add_argument("--decoder-only", action="store_true", help="omit the DC-AE encoder (text-to-image only)")
    parser.add_argument("--vae-type", choices=("f16", "q8_0"), default="f16")
    args = parser.parse_args()
    source = args.model_dir
    output = args.output or source / "package"
    cli = args.sd_cli.resolve()
    dit = source / "source/transformer/diffusion_pytorch_model.safetensors"
    vae = source / "source/vae/diffusion_pytorch_model.safetensors"
    gemma = source / "gemma-2-2b-it-q4_k_m.gguf"
    tokenizer = source / "tokenizer/tokenizer.json"
    for path in (cli, dit, vae, gemma, tokenizer):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    output.mkdir(parents=True, exist_ok=True)
    dit_name = f"sana-sprint-{args.dit_type}.gguf"
    variants = DIT_TYPES[:5] if args.all_types else (args.dit_type,)
    gguf_dir = source / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    for variant in dict.fromkeys((*variants, args.dit_type)):
        prepare_denoiser(dit, gguf_dir / f"sana-sprint-{variant}.gguf", variant, cli)
    copy_file(gguf_dir / dit_name, output / dit_name)
    for old in output.glob("sana-sprint-*.gguf"):
        if old.name != dit_name:
            archived = gguf_dir / old.name
            if not archived.exists():
                copy_file(old, archived)
            old.unlink()
    vae_name = f"dc-ae{'-decoder' if args.decoder_only else ''}-{args.vae_type}.gguf"
    prepare_vae(vae, output / vae_name, args.vae_type, cli, args.decoder_only)
    for old in output.glob("dc-ae*.gguf"):
        if old.name != vae_name:
            archived = gguf_dir / old.name
            if not archived.exists():
                copy_file(old, archived)
            old.unlink()
    for src, name in ((gemma, gemma.name), (tokenizer, "tokenizer.json"), (cli, "sd-cli")):
        copy_file(src, output / name)
    launcher = output / "generate.sh"
    launcher.write_text(
        '#!/bin/sh\n'
        'directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'exec "$directory/sd-cli" \\\n'
        f'  --diffusion-model "$directory/{dit_name}" \\\n'
        f'  --llm "$directory/{gemma.name}" \\\n'
        '  --tokenizer "$directory/tokenizer.json" \\\n'
        f'  --vae "$directory/{vae_name}" \\\n'
        '  --sampling-method scm --cfg-scale 1 --guidance 4.5 \\\n'
        '  --steps 2 --width 1024 --height 1024 "$@"\n'
    )
    launcher.chmod(0o755)
    files = {p.name: p.stat().st_size for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest = {"model": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                "denoiser_type": args.dit_type, "files": files, "total_bytes": sum(files.values())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Package: {output} ({manifest['total_bytes'] / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
