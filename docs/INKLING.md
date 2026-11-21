# Inkling 2 Mini: a small-RAM engine that reads experts from the SSD

This page is for people who want to run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a
computer with little RAM and a fast NVMe SSD. It covers what runs, how memory is split, what was measured and what is not
done yet. **Status: experimental, text only. Builds for the CPU, NVIDIA (CUDA) and AMD (HIP). Measured on an AMD
Radeon RX 9070 XT (HIP); the CUDA build uses the same code but was not compiled or run here.**

> **On this page:** [Quick start](#quick-start) · [How it fits in little RAM](#how-it-fits-in-little-ram) ·
> [With a GPU](#with-a-gpu) · [Sizes](#sizes) · [Measurements](#measurements) · [Checks](#checks) · [Options](#engine-options) ·
> [Limits and next steps](#limits-and-next-steps)

## Quick start

From a source checkout (CMake and a C++20 compiler; the CUDA toolkit or ROCm for a GPU build):

```sh
python terminal/tools/inkling/setup_inkling.py --plan          # what would happen; nothing is changed
python terminal/tools/inkling/setup_inkling.py --yes --start   # build, download, start the server
```

The script picks the size for your free RAM and builds `strata-inkling` for the GPU whose compiler it finds (`nvcc`: NVIDIA,
`hipcc`: AMD; `--backend cpu|cuda|hip` chooses). It downloads Unsloth's GGUF (pinned revision, size and SHA-256 checked),
extracts the tokenizer and writes `~/strata-inkling/strata-inkling.json`. The server then answers on
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

## With a GPU

With `--gpu auto` (the default in a GPU build) the GPU does two things:

- **The dense weights** (attention projections and the output head, 1.1 GiB of UD-Q4_K_XL) go from the file straight to
  VRAM and their products run there. They then take no RAM, which goes to the expert cache.
- **The most-used experts** fill the rest of the VRAM. This is a cache in front of the RAM cache: an expert is copied up
  from RAM the first time it is used while there is room, and once it is used twice when the cache is full. The experts
  that leave are picked by the same clock with use counts as in RAM. A layer runs the experts it has in VRAM on the GPU
  (ggml's `mul_mat_id`) while the CPU computes the others, and the outputs are summed in the same order as before.

`--vram-budget GIB` caps all the VRAM the engine uses (default: what is free, less 1 GiB). `--vram-budget 0` keeps
only the dense weights there. The attention over the cache, the routing and the small loop controllers stay on the CPU.

Every copy to and from the GPU goes through pinned host memory, and a call waits once. Decoding makes about 140 GPU
calls per token (three per layer application); on the RX 9070 XT each costs about 0.1 ms, which is now a large part of
the time per token.

Build it on its own:

```sh
cmake -S engine/src/inkling -B build-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release -DINKLING_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=native
cmake -S engine/src/inkling -B build-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release -DINKLING_GPU=hip -DGPU_TARGETS=gfx1201
cmake --build build-inkling --target strata-inkling          # (INKLING_GPU=off, the default: CPU only)
```

Use your card's gfx name for `GPU_TARGETS` (`rocm_agent_enumerator` prints it). The HIP build links ROCm's libraries
dynamically: ROCm has no static ones.

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
3 GB/s, 688 MiB is 0.24 s and 97 MiB is 0.03 s. That is an estimate, not a measurement.

**Reading while computing.** A layer computes the experts it has in RAM while its missing ones are read, and waits only
for those. In interleaved runs (the same settings, on and off with `STRATA_INKLING_NO_OVERLAP=1`, while another job used
the CPU) the time spent waiting for the SSD over 48 tokens fell from 9.5 to 6.2 s with a 1.5 GiB cache and from 2.8 to
1.3 s with 6 GiB. The table above was measured before this change.

**Lookahead (`--lookahead`, off by default).** Before a layer's attention, the engine can guess the layer's experts from
its input and read them while attention runs. The guess was right 52-53% of the time. With the disk warm it changed the
speed by -3% to +2% and always read more: +74% with a 9.5 GiB cache, +60% with 1.5 GiB (1.5 GiB: 2.14 tok/s on, 2.22 and
2.25 off). So it is off; it may help on a disk with high latency.

### A desktop with a GPU (UD-Q4_K_XL)

> **Measured on:** Intel Core with 8 performance and 8 efficiency cores (24 threads), 31 GiB of DDR5, AMD Radeon RX 9070
> XT (16 GB, gfx1201, ROCm 7), Kingston KC3000 NVMe SSD, Arch Linux. strata-inkling at this branch, 16 threads. Single
> runs, greedy decoding, except the server reply (temperature 0.7). The bench's decode tokens repeat, which makes the
> RAM cache's hit rate higher than in a real reply.

| Setup | Prompt | Decode |
| --- | ---: | ---: |
| CPU only, every expert in RAM (512-token prompt, 64 tokens) | 44.3 tok/s | 14.7 tok/s |
| GPU for the dense weights only (`--vram-budget 0`) | 53.4 tok/s | 13.9 tok/s |
| GPU for the dense weights and 13 GiB of experts (16-token prompt, 160 tokens) | - | 31.1 tok/s |
| 256K context, RAM capped at 14 GiB, GPU experts (2,048-token prompt, 160 tokens) | 62.7 tok/s | 21.3 tok/s |
| The same with RAM capped at 10 GiB | 60.1 tok/s | 21.2 tok/s |
| 256K context, 14 GiB of RAM, GPU experts: a 600-token reply through the server | - | 29.3 tok/s |

Before the experts went to VRAM, the same 256K runs decoded at 11.7 tok/s (14 GiB) and 5.3 tok/s (10 GiB, 736 MiB read
from the SSD per token). With 2,717 experts in VRAM (88% of them), 99% of the lookups found their expert there and the
SSD read 5-11 MiB per token. The peak RAM of the 10 GiB run was 2.5 GiB. The 600-token reply used 15 GiB of the card's
16 GB (with the desktop); set `--vram-budget` to keep room for other programs.

The default thread count is now the physical cores (16 here). All 24 hardware threads decoded at 7.3 tok/s against
14.5 with 16 (CPU only, every expert in RAM).

At 2,048 tokens of context the attention over the cache took 27 ms of the 47 ms per token; it runs on the CPU and grows
with the context.

**Not measured yet:** a computer with 8 or 16 GB of RAM, an NVIDIA GPU, a context filled past a few thousand tokens.

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
- **The GPU path** (`--dump-logits`, 835-token prompt, last 600 positions, against the CPU path): with ggml's CPU backend
  standing in for the GPU (`--gpu cpu --vram-budget 8`, 84% of the lookups through the GPU expert graph), the logits
  are the CPU path's (mean KL 0.0000). On the RX 9070 XT they differ: mean KL 0.14 (median 0.008), the same with the
  experts on the GPU or only the dense weights; the prompt's own next tokens were predicted 0.059 ± 0.030 nats worse
  with the dense weights only and 0.012 ± 0.034 worse with the experts too. Each kind of matrix on its own gives the
  same difference, so it is the GPU kernels' rounding (they quantize the activations to 8 bits per 32 values) carried
  through 48 routing choices, not one wrong kernel. A few positions differ a lot where the choice of the 8th expert is
  close.
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
| `--gpu auto\|off\|N` | The dense weights and the most-used experts on a GPU (`auto`: the first one in a GPU build; `cpu`: ggml's CPU backend, for tests) |
| `--vram-budget GIB` | All the VRAM the engine may use (default: what is free, less 1 GiB); `0`: the dense weights only |
| `--ctx N` | Context length (default 32768) |
| `--ram-budget GIB` | All the RAM the engine may use (default: what the OS reports free, less 1 GiB or 10%) |
| `--expert-cache GIB` | The expert cache directly, instead of the rest of the budget |
| `--experts auto\|resident\|mmap\|stream` | Where the experts live (`auto`: resident when they all fit, else stream) |
| `--io-threads N` | Parallel SSD reads (default 8) |
| `--no-direct-io` | Read through the OS page cache |
| `--lookahead` | Guess and read a layer's experts during its attention (see above) |
| `--threads N`, `--chunk N` | Compute threads (default: the physical cores), prompt tokens per step (default 256) |
| `--bench --prompt-len N --gen N` | Prompt and decode speed with the expert cache's numbers |

## Limits and next steps

- **GPU: the attention stays on the CPU.** It is the largest part of a token once the experts are in VRAM, and it grows
  with the context. The next step: the attention caches and each whole layer application on the GPU, one call per
  layer instead of three.
- **GPU: NVIDIA not tested.** The CUDA build uses the same ggml backend calls (and CUDA graphs) but was not compiled or
  run here.
- **GPU numbers differ from the CPU's** (see [Checks](#checks)): the same in kind as llama.cpp's GPU against its CPU,
  larger on this model because one rounding can change an expert choice.
- **The VRAM expert cache starts empty** and fills during the first replies (about 4.5 GiB copied in 160 tokens here).
- **Pictures:** not yet (the mmproj files are in the repository; the template marks images).
- **DSpark drafting:** not yet (Unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf`).
- **One conversation at a time**, with one rollback point at the end of the last prompt: a client that rewrites the last
  reply (for example without its thinking) continues from there; anything earlier is read again.
- **Prompt speed** is about half of llama.cpp's on the CPU (see above): the dense projections need matrix kernels.
- **Reads overlap compute only within a layer:** the next layer's experts are not known before its input is (the
  lookahead guess was not good enough, see above).
- Checked against llama.cpp: UD-IQ1_S and UD-IQ3_XXS. The other sizes use the same expert types (and BF16, Q3_K, Q4_K,
  Q5_K, Q5_0, Q8_0, MXFP4 from ggml) but were not compared token by token.

## Credits

Model: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines Lab (Apache 2.0);
GGUFs by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The forward pass follows llama.cpp's
`inkling2_loop` implementation ([#32694](https://github.com/ggml-org/llama.cpp/pull/32694),
[#32704](https://github.com/ggml-org/llama.cpp/pull/32704)); the expert dot products are ggml's.
