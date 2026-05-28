# HALLF: A Hardware-Aware Local LLM Framework on a Single RTX 3090

> Measured fine-tuning, quantized serving, context memory, agent learning, and the decision boundary for custom Rust-to-PTX kernels.

**Read the paper:** <https://axiomaabsurdo.github.io/hardware-aware-local-agent-framework/>
**Source:** [`main.qd`](main.qd) (Quarkdown 2.x, paged HTML)
**Status:** Working draft white paper for technical review
**Author:** Matias Ivan Mortara Silva

---

## What this paper argues

A local LLM stack on a single RTX 3090 should not be one tool's responsibility. HALLF separates the system into five lanes — *adaptation*, *serving*, *context memory*, *agent learning*, and *kernel research* — and uses **measurement, not intuition**, to decide where custom work is worth it.

The paper reports:

1. A QLoRA fine-tune of Mistral-7B-Instruct-v0.3 that works inside 5.12 GiB of VRAM.
2. An end-to-end serving comparison between bitsandbytes NF4 and vLLM + AWQ-Marlin on the same GPU.
3. An inspection-gated feasibility result for a proposed cuda-oxide AWQ kernel — stopped *before* a line of kernel code was written, because the toolchain lacked the required Ampere primitive.

## Headline numbers (RTX 3090, sm\_86, Mistral-7B)

### Decode throughput at batch size 1

| Input context | bitsandbytes NF4 | AWQ-Marlin | Speedup |
| ------------: | ---------------: | ---------: | ------: |
|           128 |       55.4 tok/s |  **130.4** |   2.35× |
|         2,048 |       18.8 tok/s |   22.4     |   1.19× |
|         4,096 |       11.5 tok/s |   12.3     |   1.07× |
|         8,192 |        6.11      |   6.10     |   1.00× |

### Total throughput at short context, scaling batch

| Batch | bitsandbytes NF4 | AWQ-Marlin | Speedup |
| ----: | ---------------: | ---------: | ------: |
|     1 |       53 tok/s   |   132      |   2.49× |
|     2 |       40         |   253      |   6.38× |
|     4 |       77         |   466      |   6.08× |
|     8 |      144         | **794**    |   5.53× |

### What this means

- **Short context, single request:** AWQ-Marlin is 2.4× faster.
- **Batched serving:** AWQ-Marlin is 5–6× faster.
- **Long context (8K):** both backends converge at ~6 tok/s — the linear-kernel optimization stops mattering. The next bottleneck lives in prefill / attention / KV cache, not in the GEMV.

## Why no custom CUDA kernel

The initial hypothesis: replace bitsandbytes' `kgemm_4bit_inference_naive` (27.17 % of measured decode time) with a fused Rust-to-PTX kernel via [cuda-oxide](https://github.com/quarkdown/cuda-oxide).

Two independent gates rejected this:

1. **Performance gate:** AWQ-Marlin already wins where the new kernel would compete.
2. **Capability gate:** cuda-oxide v0.1.0 exposes Hopper `wgmma` and Blackwell `tcgen05`, but **no Ampere `mma.sync` or WMMA abstraction**. A credible AWQ-Marlin competitor on sm\_86 requires tensor cores. Implementation halted at the inspection gate — decision `ABANDON_PROPOSED_KERNEL_TARGET`.

cuda-oxide stays in the framework, but only behind a profile-justified target *and* a confirmed primitive surface.

## Resulting framework

| Lane               | Technology               | Status                                     |
| ------------------ | ------------------------ | ------------------------------------------ |
| Adaptation         | QLoRA + bitsandbytes NF4 | Validated (canary loss 2.84 → 0.50)        |
| Production serving | vLLM + AWQ-Marlin        | Benchmarked, selected default              |
| Context memory     | OpenViking               | Specified; experimental validation pending |
| Agent learning     | OpenPipe ART (GRPO)      | Proposed research line                     |
| Kernel research    | cuda-oxide               | Execution validated; AWQ target abandoned  |
| Authoring          | Quarkdown                | This repo                                  |

## Repository layout

```text
main.qd                                 paged Quarkdown source
assets/                                 figures (SVG used by the build, PNG for social)
data/                                   raw CSVs behind the tables
  verified_initial_results.csv          QLoRA, profiler, cuda-oxide readiness
  context_scaling.csv                   §7.1 numbers
  batch_scaling.csv                     §7.2 numbers
  cuda_oxide_inspection_gate.csv        §8 primitive-support summary
source_reports/
  cuda_oxide_inspection_gate_result.txt original inspection record
.github/workflows/pages.yml             build + deploy to GitHub Pages
```

## Build locally

Requires [Quarkdown](https://github.com/iamgio/quarkdown) v2.1.2+ and a JDK 21+.

```bash
quarkdown c main.qd                 # HTML build → quarkdown-output/
quarkdown c main.qd -p -w           # live preview + watch
quarkdown c main.qd --pdf           # PDF build (needs Chrome/Chromium)
```

The GitHub Pages deployment uses `--out dist --out-name site` and injects OpenGraph / Twitter Card meta tags after the build; see [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## Evidence boundary

Numbers in this README come from local measurements on a single RTX 3090. The QLoRA, baseline-profiler, and cuda-oxide vector-add results are traceable to attached reports. The AWQ-Marlin scaling tables are author-supplied local summaries that should ship with their raw runs before external publication. OpenViking and OpenPipe ART are included as architectural choices and research lines — they have not been experimentally evaluated yet. No cuda-oxide LLM kernel was implemented; no performance ratio between a hypothetical cuda-oxide AWQ kernel and AWQ-Marlin is claimed.

## License

To be added before public submission.
