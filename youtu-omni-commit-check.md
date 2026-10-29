# svlys 三个 commit 冗余检查进度

检查对象（作者 jasonysliu / svlys，2026-10-19）：

| commit | 标题 | 规模 |
|---|---|---|
| `e145eb759` | metal: honor GGML_PREC_F32 in mul_mat | 1 文件，+6 / -2 |
| `1f0cdd6cf` | metal: read four vision RoPE sections | 1 文件，+15 / -6 |
| `ec2e6cde9` | mtmd: add Youtu-Parsing-Omni text, vision, and audio | 15 文件，+1644 / -15 |

状态说明：

- 已优化：确认有冗余或从头重写，已改为复用现有实现
- 保留：检查后认为必要、没有更好的现成实现
- 风险：不算冗余，但提 PR 前需要说明或另行处理
- 待确认：需要真实权重或维护者意见才能定

总进度：3 / 3 个 commit 已检查，冗余项全部处理完；视频已用真实 ffmpeg 跑通。剩余待办见第 5 节。

---

## 1. `e145eb759` metal: honor GGML_PREC_F32 in mul_mat

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 1.1 | 是否已有 F32 累加的 Metal 实现可复用 | 没有现成的 f32 累加 tiled kernel，回退到 mat-vec 是最小改法 | 保留 |
| 1.2 | 和其他后端的处理是否重复 | 只改 Metal，没有重复 | 保留 |
| 1.3 | 副作用 | 大 batch 时 mat-vec 偏慢；MiMoVL 也设置了 F32 精度，同样会变慢 | 风险 |

结论：没有冗余。建议单独提 ggml PR，并附性能数据。

## 2. `1f0cdd6cf` metal: read four vision RoPE sections

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 2.1 | 是否重复实现 | 是在补齐 Metal，使其和 CPU 参考实现（`ggml-cpu/ops.cpp` 已支持 4 个 section）一致，不是重复 | 保留 |
| 2.2 | 对现有 2-section 模型的影响 | qwen2vl 等模型 sect_0 == sect_1、sect_2 = sect_3 = 0，结果不变 | 保留 |
| 2.3 | 其他后端 | CUDA / Vulkan / SYCL / OpenCL 的 vision 内核只用 `sect_dims = s0 + s1`；Youtu 是 n_dims 64、sections {21, 21, 22, 0}，ic >= 42 时绕回 T 段，结果错误且不报错。WebGPU / CANN / OpenVINO / Hexagon / ET 已读 4 个 section。现有 vision 用例是 {n/4, n/4, 0, 0}（s0 + s1 == n_dims），测不出来。supports_op 的判断条件应为 `n_dims > s0 + s1`，不能用 `s2 != 0`（qwen2vl 传 {d/4, d/4, d/4, d/4}） | 风险 |

结论：没有冗余，是正确的修复。建议单独提 ggml PR，并补 `test-backend-ops` 用例。

## 3. `ec2e6cde9` mtmd: add Youtu-Parsing-Omni

### 3.1 转换脚本和 gguf-py

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 3.1.1 | `youtu_omni.py` 音频张量用硬编码字符串 `a.mm.*` | 改为走 `TensorNameMap`，复用 `TN_MM_NORM_PRE` / `TN_MM_AUDIO_MLP` / `conv2d` / `conv_out` | 已优化 |
| 3.1.2 | KV 字符串硬编码 | 改为 gguf-py 的 `Keys` 和 writer 方法 | 已优化 |
| 3.1.3 | `deepseek.py` kv_b 按 n_head 或 n_head_kv 拆分 | 文本侧走 DeepseekV2 是复用；拆分逻辑按 shape 判断，对原有 DeepSeek 配置结果不变 | 保留 |
| 3.1.4 | `base.py` 对 `TokenizersBackend` 特判 | 仓库固定的 transformers 4.57.6 没有该类，绕过确有必要；但这是共享代码，需维护者认可 | 待确认 |
| 3.1.5 | `get_model_architecture` 对 `None` config 兜底 | 一行改动，必要（`audio_config` 是 JSON null） | 保留 |

### 3.2 clip 图（`models/youtu-omni.cpp`、`qwen3a.cpp`）

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 3.2.1 | 音频 conv2d 前端拷贝 3 份，与 `qwen3a.cpp` 相同 | 抽成 `clip_graph::build_audio_conv2d_stem`，两边共用；qwen3a 输出逐位不变 | 已优化 |
| 3.2.2 | 2x2 patch merge 用自定义 `order` 输入 + get_rows | 改为 qwen2vl 的 permute，去掉 `order` 输入 | 已优化 |
| 3.2.3 | 通用 `build_ffn` 中加 `YOUTU_OMNI` 特判 | 移除，改为 override `build_mm`（同 MiMoVL），只作用于 `ff_down` | 已优化 |
| 3.2.4 | `youtu_joint_mel` 路径 | mtmd 从未设置，是死代码，已删除 | 已优化 |
| 3.2.5 | 音频尾部裁剪逐 chunk 切片再 concat | 保留部分本来就是前缀，改为一个 view | 已优化 |
| 3.2.6 | 图像 / 音频 / 组三条 build 路径 | 语义不同（融合层的 mask 和位置不同），不合并；抽出公共 helper | 已优化 |
| 3.2.7 | chunked MTHW RoPE 的 freq factors | 没有现成 helper，复用 `ggml_rope_multi` + factors 张量 | 保留 |

### 3.3 clip 加载与编码（`clip.cpp`、`clip-impl.h`、`clip-model.h`）

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 3.3.1 | RoPE 因子在 `clip_encode` 里算 3 次 | 合并为一份 | 已优化 |
| 3.3.2 | RGB 转 planar 写了 2 份 | 合并为 `clip_rgb_to_planar` | 已优化 |
| 3.3.3 | 融合层缺失时硬编码回退到 13/14/26/27 层 | 新模型不需要兼容旧文件，改为必须从 GGUF 读取 | 已优化 |
| 3.3.4 | 新增 `youtu_a_*` 模型字段 | 改为复用 `mm_norm_pre_w` / `mm_1_w` / `mm_2_w`；顺带修复 `clip_n_mmproj_embd` 在纯音频上下文中的空指针 | 已优化 |
| 3.3.5 | joint 专用的结构体字段 | 随死代码删除 | 已优化 |

### 3.4 mtmd 核心（`mtmd.cpp`、`mtmd.h`）

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 3.4.1 | 图像 / 音频 token 结构体中重复的分组字段 | 合并为 `mtmd_youtu_ref` | 已优化 |
| 3.4.2 | 图像和音频两套组编码分派 | 合并为 `mtmd_encode_youtu` | 已优化 |
| 3.4.3 | 批量编码会合并同尺寸图像 | 对行尾换行布局不成立，在 `can_batch_with` 中排除，删掉 chunk 级重复判断 | 已优化 |
| 3.4.4 | 组识别依赖 bitmap id 字符串（`+N` / `+aN`） | 没有更通用的现成机制 | 风险 |
| 3.4.5 | 新公共 API `mtmd_video_interleave_audio` | 上游可能会问为什么需要 | 风险 |

### 3.5 helper（`mtmd-helper.cpp`）

| # | 检查项 | 结论 | 状态 |
|---|---|---|---|
| 3.5.1 | 在 helper 里逐行 decode 换行，用了已弃用的 `get_nx/ny` | 改为在 mtmd 内部用 `llama_model_get_tok_embd_rows` 插入换行 embedding；上游 `-Werror` CI 下原写法会编译失败 | 已优化 |
| 3.5.2 | `extract_pcm` 重复了 ffmpeg 启动代码 | 抽成公共函数 | 已优化 |
| 3.5.3 | 采样率硬编码 16000 | 改为 `mtmd_get_audio_sample_rate` | 已优化 |
| 3.5.4 | 换行 embedding 直接取 `tok_embd` 行 | 在 embedding LoRA、Hadamard 旋转、`n_embd_inp != n_embd` 时可能和真正 decode "\n" 不同；本模型没有这些情况，真实权重下图像布局和 logits 与 HF 一致（KL 2e-9） | 已验证 |

---

## 4. 冗余之外发现的功能偏差（已按 HF/vLLM 修正）

这些不算冗余，但都是原 commit 和参考实现不一致的地方，已在同一分支修正，详见 `youtu-omni-report.md`。

| # | 偏差 | 状态 |
|---|---|---|
| 4.1 | 独立音频每 13 token 一段 mask、t 每段归零；参考实现是整块一个窗口 | 已修正 |
| 4.2 | 独立音频只包一层 begin/end；参考实现是每块单独包，多块时加时间戳 | 已修正 |
| 4.3 | 音频 mel 帧数多一帧（没有去掉最后一个 STFT 帧） | 已修正 |
| 4.4 | 视频每秒"帧 + 1 s 音频"；参考实现是 1.5 s / 29.5 s 分块（"帧、帧、2 s 音频"） | 已修正 |
| 4.5 | 视频只取前 128 秒；参考实现在整段视频上均匀取 128 帧 | 已修正 |
| 4.6 | 视频音频逐段算 mel；参考实现对整条音轨算 mel 再切片 | 已修正 |
| 4.7 | 视频帧没有单帧像素预算，且 `H*W` 后多一个 `\n` | 已修正 |
| 4.8 | 没有音轨的视频退化成独立图片；参考实现仍走融合组 | 已修正 |
| 4.9 | ffmpeg fps 滤镜每一帧晚约半个 step（29.97 fps 时偏差还会累积）；改为 `select` 精确取 `int(i * step)` 帧 | 已修正 |

## 5. 剩余待办

- [ ] 待确认：`base.py` 的 `TokenizersBackend` 绕过是否被维护者接受（3.1.4）
- [x] 换行 embedding：真实权重下与 HF 一致（3.5.4）
- [ ] 风险：Metal F32 回退的性能数据（1.3）
- [ ] 风险：CUDA / Vulkan / SYCL / OpenCL 的 4-section vision RoPE（2.3）。已在 `test-backend-ops` 加 {21, 21, 22, 0} 的 vision 用例（`test_rope` 新增可选 `sections` 参数），CPU grad 模式通过；后端修复留到 issue 里和维护者讨论
- [ ] 流程：上游没有 Youtu-Parsing-Omni 相关 issue / PR，也没有人报过后端 vision RoPE 只读 2 段的问题；前代 Youtu-VL 已在 #18479 合入。下一步由作者开 issue，再拆 Metal ggml PR（附性能数据）和 mtmd PR
- [ ] 风险：组识别依赖 bitmap id、新公共 API（3.4.4、3.4.5）
- [x] 有 ffmpeg 的机器上实跑视频：6 段合成视频帧序号、布局、e2e 全部通过（见 4.9）
- [x] 用真实权重对比 HF 输出：图像 / 音频 / 视频的 token 布局完全相同，logits top10 一致，KL <= 1e-8
- [ ] 待确认：视频预算常量 32768 / 1024 是否写进 GGUF
- [ ] 风险：很长视频的内存占用（音频不再有 128 s 上限）
- [ ] 风险：抽帧依赖 ffmpeg >= 5.1 的 `-fps_mode`
