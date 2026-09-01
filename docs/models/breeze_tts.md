# BreezeTTS 2

BreezeTTS 2 is a GGUF TTS family for instruction-conditioned speech and
prompt-audio voice cloning. The default package is Q8_0.

## Quick Start

Download the default Q8_0 package:

```bash
python3 tools/model_manager_v2.py install breeze_tts_2_q8_0 --models-root models
```

Voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Please read this line in a clear and natural voice." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option instruction="Speak clearly and naturally." \
  --out breeze_tts_clone.wav
```

Voice design:

```bash
audiocpp_cli \
  --task vdes \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Welcome to the local voice demo." \
  --request-option instruction="A warm female narrator with calm pacing and studio clarity." \
  --out breeze_tts_design.wav
```

## Model

| Field | Value |
|---|---|
| Family | `breeze_tts` |
| Default GGUF | `models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` |
| Tasks | `tts`, `clon`, `vdes` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Voice input | Optional for `tts`; required for `clon` |

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required for `clon` | Prompt/reference speaker audio. |
| `--reference-text` / `--request-option reference_text=<text>` | text | empty | Transcript for prompt audio when cloning. |
| `--request-option instruction=<text>` | text | `Speak clearly and naturally.` | Voice or style instruction. |
| `--request-option text_chunk_size=<n>` | integer > 0 | `600` | Long-form text chunk size. |
| `--request-option text_chunk_mode=<mode>` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `--request-option max_tokens=<n>` | integer > 0 | `1500` | Maximum generated acoustic frames. |
| `--request-option guidance_scale=<f>` | float >= 0 | `1.0` | Classifier-free guidance scale. |
| `--request-option temperature=<f>` | float > 0 | `0.9` | Backbone sampling temperature. |
| `--request-option depth_temperature=<f>` | float > 0 | `0.9` | Depth decoder sampling temperature. |
| `--request-option top_k=<n>` | integer >= 0 | `50` | Top-k sampling limit; `0` disables top-k filtering. |
| `--request-option top_p=<f>` | `0..1` | `1.0` | Top-p sampling limit. |
| `--request-option seed=<n>` | integer >= 0 | `0` | Generation seed. |
| `--session-option breeze_tts.reference_cache_slots=<n>` | integer >= 0 | `1` | Prepared reference-audio cache slots. |
| `--session-option weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Weight storage type; quantized types convert at load time from the BF16 package. |

Quantized weight storage is the largest measured speedup and applies to CUDA
and HIP alike: `q8_0` cut the fixed 100-token regression case from RTF ~1.5 to
~0.95 on gfx1151 and from ~0.77 to ~0.56 on an RTX 2080 Ti, and `q4_k` reached
~0.84 / ~0.49 respectively, with no audible quality regression in the Chinese
voice-design regression cases. Activations stay on the bf16-rounded path
regardless of `weight_type`; counter to intuition, fp32 is the one
configuration known to be *worse* for this model (mispronunciations and
runaway repetition), because the model is trained and tuned in bf16.

## CUDA/HIP Seeded Parity

CUDA sampling derives its Philox TensorIterator layout from the device's SM
count and maximum threads per SM. The BreezeTTS vocabularies are small enough
that normal CUDA GPUs use the complete TensorIterator grid. HIP now uses the
same full-grid CUDA/PyTorch-compatible sampler by default instead of the legacy
`std::discrete_distribution` fallback, so equal logits and seeds select equal
acoustic codes.

The layout can also be pinned explicitly for strict cross-machine regression
testing. For an RTX 2080 Ti, the observed CUDA layout is `68x1024`:

```bash
ENGINE_TORCH_SAMPLING_POLICY=68x1024 audiocpp_cli ... --backend cuda ...
ENGINE_TORCH_SAMPLING_POLICY=68x1024 audiocpp_cli ... --backend hip ...
```

The pinned path uses the CUDA/PyTorch-compatible Philox categorical sampler on
the host for both backends. It makes the generated acoustic codes identical
when the model logits are otherwise identical. With `--log`, compare
`breeze_tts.generate.semantic_codes_hash` and
`breeze_tts.generate.codes_hash` before comparing waveforms.

An equal seed is not by itself a bit-exact cross-backend guarantee. CUDA and HIP
BF16 kernels can produce slightly different logits, and a small difference near
a sampling boundary can make a long generation diverge. Voice cloning can also
diverge earlier if the reference encoder quantizes the same waveform to
different codes; `breeze_tts.reference.codes_hash` distinguishes that case from
a sampler problem. Even when all generated codes match, the final WAV need not
be bit-identical because speech-decoder floating-point rounding can differ.

## ROCm Performance Notes

At the default `guidance_scale=1`, BreezeTTS skips the mathematically redundant
unconditional backbone prompt, prefill, and decode. The depth transformer stays
batched to preserve its established numerical path. On one gfx1151 regression
case this reduced autoregressive time by about 16% and end-to-end time by about
15%; the exact gain depends on the prompt and generated length. Other guidance
scales retain the full classifier-free-guidance path.

Profiling on gfx1151 showed that the main remaining ROCm cost is the many small
batch-1 and batch-2 BF16 matrix-vector kernels in autoregressive and depth
decode, rather than Philox sampling. Forcing those operations through
hipBLASLt was slower in the measured workload, while disabling HIP graphs or
forcing F16 weight storage did not materially close the gap. Treat such backend
overrides as experiments and benchmark the complete request before enabling
them by default.
