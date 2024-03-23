# Convert paraformer model from PyTorch to ggml format
#
# Usage:
# python convert-pt-to-ggml.py -i damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/ -o .
# You need to download the model from
# https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary
#
# Also, you need to have the original models in damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404
# ├── am.mvn
# ├── configuration.json
# ├── config.yaml
# ├── decoding.yaml
# ├── example
# │   ├── asr_example.wav
# │   └── hotword.txt
# ├── fig
# │   └── struct.png
# ├── finetune.yaml
# ├── lm
# │   ├── lm.pb
# │   └── lm.yaml
# ├── model.pb
# ├── README.md
# ├── seg_dict
# └── tokens.txt
#
# This script loads the specified model and paraformer assets and saves them in ggml format.
# The output is a single binary file containing the following information:
#
#  - hparams
#  - tokenizer vocab
#  - model variables
#
# For each variable, write the following:
#
#  - Number of dimensions (int)
#  - Name length (int)
#  - Dimensions (int[n_dims])
#  - Name (char[name_length])
#  - Data (float[n_dims])
#

import argparse
import io
import struct
import sys
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", "-i", required=True)
    parse.add_argument("--output", "-o", default=".")
    parse.add_argument("--fp16", action="store_true")
    args = parse.parse_args()

    dir_in = Path(args.input)
    output = Path(args.output)

    # try to load PyTorch binary data
    try:
        model_bytes = open(dir_in / "model.pb", "rb").read()
        with io.BytesIO(model_bytes) as fp:
            checkpoint = torch.load(fp, map_location="cpu")
    except Exception:
        print("Error: failed to load PyTorch model file:", dir_in / "model.pb")
        sys.exit(1)

    hparams = {}

    with open(dir_in / "tokens.txt", "rb") as f:
        contents = f.read()
        tokens = {
            token: int(rank)
            for rank, token in enumerate(
                line.strip() for line in contents.splitlines() if line
            )
        }

    hparams["n_vocab"] = len(tokens)
    # encoder config
    hparams["n_encoder_hidden_state"] = 512
    hparams["n_encoder_linear_units"] = 2048
    hparams["n_encoder_attention_heads"] = 4
    hparams["n_encoder_layers"] = 50
    # decoder config
    hparams["n_decoder_hidden_state"] = 512
    hparams["n_encoder_0_norm_size"] = 560
    hparams["n_decoder_linear_units"] = 2048
    hparams["n_decoder_attention_heads"] = 4
    hparams["n_decoder_layers"] = 15
    hparams["fsmn_kernel_size"] = 11
    # predictor config
    hparams["n_predictor_dim"] = 512
    hparams["predictor_tail_threshold"] = 0.45

    print("hparams:", hparams)
    # output in the same directory as the model
    fname_out = output

    # use 16-bit or 32-bit floats
    use_f16 = args.fp16

    fout = fname_out.open("wb")

    fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
    fout.write(struct.pack("i", hparams["n_vocab"]))
    fout.write(struct.pack("i", hparams["n_encoder_hidden_state"]))
    fout.write(struct.pack("i", hparams["n_encoder_linear_units"]))
    fout.write(struct.pack("i", hparams["n_encoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["n_encoder_layers"]))
    fout.write(struct.pack("i", hparams["n_encoder_0_norm_size"]))
    fout.write(struct.pack("i", hparams["n_decoder_hidden_state"]))
    fout.write(struct.pack("i", hparams["n_decoder_linear_units"]))
    fout.write(struct.pack("i", hparams["n_decoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["n_decoder_layers"]))
    fout.write(struct.pack("i", hparams["fsmn_kernel_size"]))
    fout.write(struct.pack("i", hparams["n_predictor_dim"]))
    fout.write(struct.pack("f", hparams["predictor_tail_threshold"]))
    print(hparams)
    # write tokenizer
    fout.write(struct.pack("i", len(tokens)))
    print(f"tokens num: {len(tokens)}")

    for key in tokens:
        fout.write(struct.pack("B", len(key)))
        fout.write(key)

    for name in checkpoint.keys():
        data = checkpoint[name].squeeze().numpy()

        # reshape conv bias from [n] to [n, 1]
        if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
            data = data.reshape(data.shape[0], 1)
            print(f"  Reshaped variable: {name} to shape: ", data.shape)

        n_dims = len(data.shape)

        # looks like the paraformer models are in f32 by default
        # so we need to convert the small tensors to f32 until we fully support f16 in ggml
        # ftype == 0 -> float32, ftype == 1 -> float16
        if use_f16:
            ftype = 1
        else:
            ftype = 0

        if use_f16:
            data = data.astype(np.float16)
            ftype = 1
        else:
            data = data.astype(np.float32)
            ftype = 0

        str_ = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str_), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str_)

        # data
        print(
            "Processing variable: ",
            name,
            " with shape: ",
            data.shape,
            " with type: ",
            data.dtype,
        )
        data.tofile(fout)

    fout.close()

    print("Done. Output file: ", fname_out)
