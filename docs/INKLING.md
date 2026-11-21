# Inkling 2 Mini: a small-RAM engine that reads experts from the SSD

This page is for people who want to run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a
computer with little RAM and a fast NVMe SSD. It covers what runs, how memory is split, what was measured and what is not
done yet. **Status: experimental, CPU only, text only.**

> **On this page:** [Quick start](#quick-start) · [How it fits in little RAM](#how-it-fits-in-little-ram) ·
> [Sizes](#sizes) · [Measurements](#measurements) · [Checks](#checks) · [Options](#engine-options) ·
> [Limits and next steps](#limits-and-next-steps)

## Quick start

From a source checkout (CMake and a C++20 compiler; no GPU toolkit):

```sh
python terminal/tools/inkling/setup_inkling.py --plan          # what would happen; nothing is changed
python terminal/tools/inkling/setup_inkling.py --yes --start   # build, download, start the server
```

The script picks the size for your free RAM, builds `strata-inkling`, downloads Unsloth's GGUF (pinned revision, size
and SHA-256 checked), extracts the tokenizer and writes `~/strata-inkling/strata-inkling.json`. The server then answers on
`http://127.0.0.1:8080/v1` (OpenAI Chat Completions and Responses) and `/v1/messages` (Anthropic), as for the other models.
Choose a size with `--model UD-Q4_K_XL`, the context with `--context 16384`, the RAM with `--ram-budget 6`.

`strata launch` and the desktop app do not offer this model yet: the engine is not in the release packages.

## How it fits in little RAM

Inkling 2 Mini has 26.4B parameters in 24 layers. The middle 12 layers run three times per token, so a token goes
through 48 layer applications, and each picks 8 of its layer's 128 experts. The experts are 85-90% of every file:

| | UD-IQ1_S | UD-Q4_K_XL |
| --- | ---: | ---: |
| File | 7.8 GB | 17.6 GB |
| Dense weights (always in RAM) | 1.14 GiB | 1.69 GiB |
| Experts | 6.08 GiB | 14.71 GiB |
| One expert | about 2 MiB | 4.6 MiB (NVFP4), 8.1 MiB (Q8_0) |

So the engine keeps the dense weights in RAM and treats RAM as a cache for the experts:

- **Every expert fits:** they are read into RAM at start (`--experts resident`).
- **They do not:** the rest of the budget is an expert cache. A missing expert is read whole (2-8 MiB) with direct I/O
  by 8 reader threads, so the SSD gets large reads at a high queue depth and nothing is copied into the OS page cache.
  Experts in use are pinned; the others leave by a clock with use counts, so often-used experts stay. Each expert size
  has its own slots (UD-Q4_K_XL mixes 4.6 and 8.1 MiB experts), which fits 61% more experts in the same RAM than one
  slot size would.
- **The loop helps.** A core layer's three passes use the same experts, so the second and third passes often find
  them in RAM already. With 10% of UD-Q4_K_XL's experts in RAM, 61% of expert lookups were served from RAM; with 20%,
  81%; with 41%, 90%.

The token embedding table (246 MB) stays on the SSD too: one row is read per prompt token.

## Sizes

All ten of Unsloth's sizes load (the root files and the `Mainline_Support` files). The setup script recommends
**UD-Q4_K_XL** (the checkpoint's own NVFP4 experts) whenever at least a tenth of its experts fit in the RAM left over,
and streams the rest. With less, it offers the largest size whose experts all fit, or else the smallest.

| RAM free | What the setup script picks (16K context) |
| ---: | --- |
| 32 GiB | UD-Q4_K_XL, every expert in RAM |
| 12 GiB | UD-Q4_K_XL, 57% of the experts in RAM |
| 6 GiB | UD-Q4_K_XL, 16% of the experts in RAM |
| 4.5 GiB | UD-IQ1_S, 22% of the experts in RAM |

The attention caches take 32 KiB per context token (8 full-attention applications, f16) and a fixed 0.2 GiB for the
40 sliding-window applications, so 32K of context is 1.2 GiB.

## Measurements

> **Measured on:** a cloud VM with 4 vCPUs (Intel Xeon at 2.1 GHz, AVX-512 and AVX-VNNI), 15 GiB of RAM, no GPU, and a
> virtual disk that read 0.27 GB/s sequentially and 0.49 GB/s at 8 random 4.5 MiB reads in flight when cold (the host
> caches it after use: later runs read several times faster). strata-inkling at this branch, 4 threads; llama.cpp
> ggml-org/llama.cpp#32704 (42fd370cf) built for the same CPU. Single runs; greedy decoding.

### Everything in RAM (UD-IQ1_S)

| | strata-inkling | llama.cpp |
| --- | ---: | ---: |
| Decode, 16 tokens after a 512-token prompt | 4.30-4.43 tok/s | 4.78 tok/s (`llama-bench tg16`) |
| Prompt, 512 tokens | 8.6-8.9 tok/s | 18.0 tok/s (`llama-bench pp512`) |
| Through the server, a short chat turn | 4.7-5.1 tok/s | - |

Decode is within 10% of llama.cpp. The prompt is half as fast: the dense projections run here as row-by-token dot
products, where llama.cpp has repacked Q8_0 matrix kernels.

### Streaming from the SSD (UD-Q4_K_XL, 16.4 GiB of tensors on a 15 GiB computer)

A 32-token prompt, then 48 decoded tokens, default options. "From RAM" counts expert lookups (384 per token); "peak RAM"
is the engine's resident memory at the end (the attention caches hold little at this length).

| Expert cache | Experts in RAM | From RAM | Read from the SSD per token | Peak RAM | Decode on this VM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 GiB | 312 of 3,072 (10%) | 61% | 688 MiB | 2.87 GiB | 1.1-2.3 tok/s (5 runs) |
| 3 GiB | 624 (20%) | 81% | 342 MiB | 4.37 GiB | 2.6 tok/s (1 run) |
| 6 GiB | 1,248 (41%) | 90% | 179 MiB | 7.36 GiB | 3.1 tok/s (1 run) |
| 9.5 GiB | 1,976 (64%) | 95% | 97 MiB | 10.85 GiB | 1.9-3.2 tok/s (4 runs) |

The hit rates and the bytes read were the same in every run. The speed was not: this VM's disk is fast where its host
has cached the file and slow where it has not, so the same setting decoded at 1.1 or 2.3 tok/s. Without the time spent
waiting for reads, decoding took 0.27 s per token at every cache size (about 3.7 tok/s): that is this CPU's limit. On
another computer, a rough estimate per token is that compute time plus the MiB read divided by the SSD's speed: at
3 GB/s, 688 MiB is 0.24 s and 97 MiB is 0.03 s. That is an estimate, not a measurement. The engine does not yet compute
the experts it has while it reads the missing ones of the same layer (see [next steps](#limits-and-next-steps)).

**Lookahead (`--lookahead`, off by default).** Before a layer's attention, the engine can guess the layer's experts from
its input and read them while attention runs. The guess was right 52-53% of the time. With the disk warm it changed the
speed by -3% to +2% and always read more: +74% with a 9.5 GiB cache, +60% with 1.5 GiB (1.5 GiB: 2.14 tok/s on, 2.22 and
2.25 off). So it is off; it may help on a disk with high latency.

**Not measured yet:** a real NVMe SSD (3-7 GB/s), a computer with 8 or 16 GB of RAM, any GPU.

## Checks

- **Against llama.cpp** (the Mainline files; `--dump-logits` here, a libllama tool there): the logits of every position of
  four prompts (12, 105, 13 and 1,401 tokens; the last crosses the 1,024-token sliding window and the relative-position
  range). Mean KL divergence from llama.cpp without flash attention, next to llama.cpp's own two attention paths (flash
  attention on against off) on the same prompts:

  | Prompt | UD-IQ1_S (IQ and Q2_0 experts) | its llama.cpp FA vs no-FA | UD-IQ3_XXS (NVFP4, IQ3, IQ4_NL) | its llama.cpp FA vs no-FA |
  | --- | ---: | ---: | ---: | ---: |
  | 12 tokens | 0.0013 | 0.0011 | 0.0011 | 0.0007 |
  | 105 tokens (code) | 0.0006 | 0.0010 | 0.0003 | 0.0003 |
  | 13 tokens (chat tokens) | 0.011 | 0.013 | 0.016 | 0.004 |
  | 1,401 tokens, last 48 | 0.0011 | 0.0015 | 0.0010 | 0.0011 |

  The one outlier (UD-IQ3_XXS, chat tokens) is one position: in layer application 17 its 8th and 9th experts are 2.3%
  apart (0.1661 against 0.1624), the rounding of the two implementations picks a different 8th, and the layers after it
  follow. Every expert type's dot product equals llama.cpp's (they are ggml's, checked row by row against llama.cpp's
  build). Top-1 tokens agree except at such near ties.
- **The two layouts:** the root file and the `Mainline_Support` file of UD-IQ1_S give bit-identical logits.
- **`inkling_tiny_test`** (no model): a tiny random model with the full architecture, written in both layouts. A prompt in
  chunks, one token at a time and in one chunk; the two layouts; experts in RAM, memory-mapped and streamed through a
  4-slot cache; a rollback and a fresh run: all give the same logits, bit for bit.
- **`inkling_store_test`**: the slot cache, the RAM budget split, direct reads at unaligned offsets and the end of a
  file, and the SIMD Q2_0 kernels against a double-precision sum.
- **The tokenizer** (`terminal/tools/inkling/tokenizer_check.py`): 18 strings (case runs, contractions, digits, CJK,
  Arabic, emoji, combining marks, whitespace, the chat tokens) give llama.cpp's ids and decode back.
- **The protocol** (`terminal/tools/inkling/serve_gate.py`): a continued conversation and a rollback give a fresh engine's
  tokens, STOP, prompt-only requests, seeded sampling, a refused id.
- **The server** (`terminal/serve/test_inkling.py`): the chat template with tools and effort, the reply parser on thinking,
  answers, one and two tool calls and markers split across deltas.

## Engine options

`strata-inkling --model FILE --serve` is what the server starts; `--help` lists everything.

| Option | Effect |
| --- | --- |
| `--ctx N` | Context length (default 32768) |
| `--ram-budget GIB` | All the RAM the engine may use (default: what the OS reports free, less 1 GiB or 10%) |
| `--expert-cache GIB` | The expert cache directly, instead of the rest of the budget |
| `--experts auto\|resident\|mmap\|stream` | Where the experts live (`auto`: resident when they all fit, else stream) |
| `--io-threads N` | Parallel SSD reads (default 8) |
| `--no-direct-io` | Read through the OS page cache |
| `--lookahead` | Guess and read a layer's experts during its attention (see above) |
| `--threads N`, `--chunk N` | Compute threads (default all), prompt tokens per step (default 256) |
| `--bench --prompt-len N --gen N` | Prompt and decode speed with the expert cache's numbers |

## Limits and next steps

- **CPU only.** A GPU would hold the dense weights, the attention caches and the hottest experts: the next step.
- **Pictures:** not yet (the mmproj files are in the repository; the template marks images).
- **DSpark drafting:** not yet (Unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf`).
- **One conversation at a time**, with one rollback point at the end of the last prompt: a client that rewrites the last
  reply (for example without its thinking) continues from there; anything earlier is read again.
- **Prompt speed** is about half of llama.cpp's on the CPU (see above): the dense projections need matrix kernels.
- **Reads and compute do not overlap within a layer yet:** a layer waits for all its missing experts before it computes
  any. Computing the experts it has while the others load is the next step for the SSD path.
- Checked against llama.cpp: UD-IQ1_S and UD-IQ3_XXS. The other sizes use the same expert types (and BF16, Q3_K, Q4_K,
  Q5_K, Q5_0, Q8_0, MXFP4 from ggml) but were not compared token by token.

## Credits

Model: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines Lab (Apache 2.0);
GGUFs by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The forward pass follows llama.cpp's
`inkling2_loop` implementation ([#32694](https://github.com/ggml-org/llama.cpp/pull/32694),
[#32704](https://github.com/ggml-org/llama.cpp/pull/32704)); the expert dot products are ggml's.
