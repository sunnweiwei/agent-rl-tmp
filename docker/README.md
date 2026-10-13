# Images

| Image | Built from | Used by |
|---|---|---|
| `sunnweiwei/verl-qwen35:b200-verl0.8.0-vllm-c621af169-fla051-fused-intra-rl-sleepfix-lora64fix-20260719` | `b200-training/Dockerfile.fla-triton` | B200 recipes |
| `sunnweiwei/verl-qwen35:h100-verl0.8.0-vllm-main-c621af169-mtpaccept-prefixcache-20260614` | base image (verl 0.8 + vLLM c621af169, MTP and prefix-cache fixes) | H100 recipe |
| H100 image with the same vLLM patches as B200 | `h100-sampling/Dockerfile.fused-gdn` | optional for H100 |
| `agent-sandbox` | `sandbox/Dockerfile` | sandbox server |

vLLM patches applied on top of the base images (`b200-training/*.patch`):

- `vllm-fla051-fused-gdn.patch` — Qwen3.5/3.6 GatedDeltaNet prefill through
  FLA 0.5.1's fused KKT + solve_tril + recompute_w_u kernel.
- `vllm-rl-sleep-wake.patch` — CUDA synchronization around CuMem sleep/wake and
  a scheduler gate during partial wake. Colocated RL sleeps vLLM, wakes weights
  for the LoRA sync, then wakes the KV cache; without the barriers an in-flight
  CUDA graph can touch unmapped memory (Xid 31).
- `vllm-lora-int64-row-index.patch` — 64-bit row offsets in the LoRA expand
  kernel; int32 overflows for wide projections (27B gate/up) above ~61K packed
  tokens.

`scripts/preflight_image.sh` asserts the first two at container start when the
recipe sets `REQUIRE_VLLM_FLA051_FUSED_GDN` / `REQUIRE_VLLM_RL_SLEEP_WAKE_FIX`.

Build from the repository root, e.g.
`docker build -f docker/b200-training/Dockerfile.fla-triton -t <tag> .`
