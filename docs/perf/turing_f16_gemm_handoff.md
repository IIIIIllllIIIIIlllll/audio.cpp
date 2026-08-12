# Turing（及老卡）FP16 GEMM 性能优化交接

> 写于 2026-08-12，基于 IndexTTS-2.5 性能对比工作中的实测发现。
> 目标读者：接手此优化任务的工程师。本文档包含问题定义、实测数据、代码入口、验证方法。

## 1. 问题定义

在 Turing 架构（sm_75，RTX 2080 Ti 22GB）上，**f16 权重相对 fp32 权重没有任何推理速度收益**——实测 f16 / fp32 / q8_0 三档 RTF 几乎相同（见 §2）。理论上 f16 GEMM 走 tensor core（HMMA）应有接近 2 倍于 fp32 的吞吐。

怀疑（未证实）的_root cause_：本引擎模型图中激活张量为 f32，f16 权重 × f32 激活的混合精度 mul_mat 在 ggml CUDA 后端的调度下没有落入 tensor core 路径（tensor core 仅在 f16×f16→f16 且开启 TENSOR_OP 时启用）。调度层如果能把 f32 激活就地转 f16 并走 `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`，Turing 即可吃到 HMMA 红利。

**注意**：这是张量库/内核层的工作（`external/ggml/`），与任何模型实现（`src/models/`）无关。不要在本分支改模型代码。

## 2. 实测数据（2026-08-12，RTX 2080 Ti，驱动 595.71.05，CUDA 13.2）

IndexTTS-2.5（0.8B GPT + DiT CFM + BigVGAN，采样 beams=3/top_p 0.8/top_k 30/temp 0.8/rep 10），RTF = 推理耗时 / 音频时长，warm 均值：

| 权重档 | zh 短句 RTF | en 短句 RTF | ja 短句 RTF | zh 长句 RTF | 显存峰值 |
|---|---|---|---|---|---|
| fp32（orig GGUF） | 0.625 | 0.735 | 0.695 | 0.549 | 11.5 GB |
| f16 GGUF（8-11 测量） | ~0.62 | ~0.73 | ~0.70 | ~0.55 | 9.6 GB |
| q8_0 GGUF（8-11 测量） | ~0.58 | ~0.68 | ~0.66 | ~0.51 | 8.8 GB |

三档速度基本打平 → f16 没有 tensor core 加速、显存带宽也不是瓶颈（2080 Ti 616 GB/s 下 4.5GB vs 7.9GB 权重无差异）。fp32 快是因为 GEMM 本身在 Turing 上就是全速 f32 CUDA core 路径。

参考：官方 PyTorch 同精度 fp32 采样 RTF 0.517–0.733（与我们打平）；官方默认 bf16 在 Turing 上 RTF 1.136–1.368（bf16 无硬件支持，负优化）。

**优化目标**：f16 档 RTF 降到 fp32 档的 50–60%（tensor core 理论 2x，考虑激活转换开销打个折），且生成音频内容不变（验收见 §5）。

## 3. 代码入口（external/ggml，fork 自带）

主调度：`external/ggml/src/ggml-cuda/ggml-cuda.cu`

- `ggml_cuda_mul_mat()`（~line 2704）：mul_mat 总调度。量化权重走 mmq/mmv 自定义内核；非量化（f16/f32/bf16）走 cuBLAS 路径
- f16×f16→f16 tensor core 路径示例（~line 1850–1865）：`cublasGemmEx(..., CUDA_R_16F, ..., CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`——注意它要求 **src1（激活）也是 f16**；激活为 f32 时不走这里
- `batched_mul_mat_traits<GGML_TYPE_F16>`（~line 2309）：`compute_type = CUBLAS_COMPUTE_16F`
- `need_compute_32f`（~line 2403）：CDNA / RDNA4 / Volta 被强制 f32 compute，**Turing 不在强制列表**（f16 compute 本来就允许，问题在激活 dtype）
- 环境开关（fork 新增）：`GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` / `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F`（互斥，启动时打印检测日志）
- HIP 后端独立目录：`external/ggml/src/ggml-hip/`（ROCm ≥ 6.5 走 hipBLASLt，见 `ggml_hipblaslt_gemm` 包装）

建议的切入点（按优先级）：

1. **f16 权重 + f32 激活的就地转换路径**：调度时把 src1 cast 成 f16（一次性开销），走 TENSOR_OP f16 GEMM，输出转回 f32。先在小基准（单个 1280×5120 类 GEMM 形状）上验证 tensor core 确实加速
2. 若担心 f16 激活精度：可只对内层 GEMM 做转换，logits/输出层（`GGML_PREC_F32` 标记处，如 mel_head）保持 f32 路径——`dst->op_params[0] == GGML_PREC_DEFAULT` 判断处已是现成的分流点
3. HIP 侧同理（gfx1151 有原生 f16/bf16 加速），但优先级低于 CUDA Turing

## 4. 构建与基准环境

测试服务器：`192.168.5.12`（用户 mark，RTX 2080 Ti 22GB + AMD gfx1151 核显）。
完整环境细节见仓库根目录 `INDEXTTS25_PERF_HANDOFF.md` 第 1 节（plink 用法、CUDA 版本选型、网络镜像等）。

CUDA 构建：

```bash
cd /home/mark/Workspace/audiocpp-25-refactor   # 或你自己的 checkout
cmake -S . -B build/linux-cuda -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON \
  -DENGINE_BUILD_WARMBENCH=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/linux-cuda -j 16 --target audiocpp_cli index_tts2_warm_bench
```

基准脚本（可复用，含 VRAM 采样）：

```bash
/home/mark/Workspace/indextts25/perf/perf_cpp.sh          # 旧版（family index_tts2_5）
/home/mark/Workspace/indextts25/perf/perf_cpp_fp32.sh     # 新版（family index_tts2）
# 用法：perf_cpp_fp32.sh 内 TAG/MODEL/WB 三个变量改掉即可跑任意权重档
```

用例文件：`/home/mark/Workspace/indextts25/perf/perf_cases.json`（zh/en/ja 短句 + zh 长句，seed 固定）。
模型权重：`/home/mark/Workspace/indextts25/refactor/staging-{orig,f16,q8_0,bf16}/index-tts2_5-*.gguf`。
官方 Python 对照：`/home/mark/Workspace/indextts25/perf/perf_official.py`（bf16/fp32 × 采样/贪心四象限已齐，结果在 `official_results.json` / `official_fp32_sampling.json`）。

## 5. 验收标准

优化合入前必须通过：

1. **性能**：f16 档 zh_long RTF ≤ 0.33（fp32 档 0.549 的 60%）
2. **数值正确性**：贪心模式（`do_sample=false num_beams=1 seed=42`）下，优化后的 f16 输出与优化前的 f16 输出 **codes 序列一致**（用音频时长 + ASR 双重核对；逐样本对比不需要——GEMM 重排必有浮点噪声）
3. **内容正确性**：zh/en/ja 三条 ASR 逐字一致（qwen3_asr 验收命令见 INDEXTTS25_PERF_HANDOFF.md §4）
4. **回归**：fp32 档 RTF 不得劣化 >2%；q8_0 档不受影响

## 6. 已知陷阱

- **不要动** `dst->op_params[0] == GGML_PREC_F32` 标记的 GEMM（mel_head logits 等），那些是精度敏感点
- HIP 后端在 `ggml-hip/` 独立目录，改了 `ggml-cuda/` 不自动传导
- ROCm < 6.5 与 ≥ 6.5 的 hipBLASLt 数据类型枚举编号不同（fork 已做映射，新增调用点要用同一个包装）
- plink 断开后远程进程变孤儿继续跑：长任务一律 `nohup ... > log 2>&1 &`
