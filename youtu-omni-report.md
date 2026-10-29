# Youtu-Parsing-Omni 支持：改动汇报

- 分支：`youtu-omni-split-cleanup`，基于 svlys 的 `ec2e6cde9`
- 状态：所有改动均未提交，18 个文件，+967 / -1133 行
- 参考实现：HF `tencent/Youtu-Parsing-Omni`（transformers 代码）和官方 vLLM 插件

## 1. 去冗余重构（针对 svlys 的 3 个 commit）

| 方向 | 改动 |
|---|---|
| 复用已有实现 | 音频 conv2d 前端和 qwen3a 共用 `build_audio_conv2d_stem`；2x2 patch merge 改为 qwen2vl 的 permute 写法；F32 精度改为 override `build_mm`（同 MiMoVL），移除通用 `build_ffn` 里的特判 |
| 删除死代码和重复 | 删除从未启用的 joint-mel 路径；RoPE 因子、RGB 转 planar 各只保留一份；音频尾部裁剪由 concat 循环改为一个 view |
| 规范化 | 张量名和 KV 走 gguf-py 的 `TensorNameMap` / `Keys`；融合层必须从 GGUF 读取，去掉硬编码回退 |
| mtmd 内部 | 两个 token 结构体中重复的分组字段合并为 `mtmd_youtu_ref`，编码分派合并为 `mtmd_encode_youtu`；行尾 `\n` embedding 改在 mtmd 内部插入，不再依赖已弃用的 `get_nx/ny` |
| helper | `extract_pcm` 复用 ffmpeg 启动代码，采样率从模型读取 |

验证：用随机初始化的微型模型对比，图像、音频、音视频组和 qwen3a 的输出都与原代码逐位一致。

## 2. 按 HF/vLLM 参考实现对齐

### 独立音频
- mel 与 `WhisperFeatureExtractor(padding="do_not_pad")` 一致：帧数为 n_samples/160，去掉最后一帧后再做 max-8 截断
- 按 2 s / 30 s 规则分块，计算方式照搬参考实现（"30 s" 一块实际是 2999 帧）
- 每块单独包 `<|audio_start|>…<|audio_end|>`；多于一块时，块前加 `HH:MM:SS` 时间戳
- 编码：去掉 13 token 分段 mask，整块是一个窗口，t = 0..n-1
- 顺带修复：`string_format` 返回的字符串末尾带 `\0`，会多生成一个 token；时间戳改用 `snprintf`

### 视频
- 采帧：最多 128 帧、不超过 1 fps，时间戳公式同 HF。原实现只取前 128 秒
- 抽帧用 ffmpeg `select='eq(n,floor(ceil(n/step)*step))'` 加 `-fps_mode passthrough`，取到的源帧序号和 HF 的 `int(i * step)` 完全相同。原先的 fps 滤镜每一帧都会晚约半个 step
- 分块：按 `_chunk_audio_video_frames` 的 1.5 s / 29.5 s 规则，1 fps 下是"帧、帧、2 s 音频"；单帧音频超过 29.5 s 时切开
- 提示词布局：`<|video_start|>`，每块依次是"[时间戳] 帧"和"[时间戳] 音频"，最后是 `<|video_end|>`；视频帧的 `H*W` 后面不加 `\n`
- mel：同一视频的音频片段拼回整条音轨，只算一次 mel 再切片，和 HF 的全局 max 截断一致
- 单帧像素预算：min(32^2 x 32768 / 帧数, 32^2 x 1024)
- 没有音轨的视频也走融合编码，所有帧放在同一组

### 已确认与参考实现一致、无需修改
RoPE 分段与因子、融合层、组内时间 cursor（帧 12.5、音频 1）、A->I 切组、merger eps、conv 前端

### 图像预处理
- 跟 HF `process_native_v2`（不是 vLLM 的 smart_resize）：按像素预算等比缩放，再在右侧和下方补 127 灰边到 32 的倍数；超出预算时向下取整并缩小内容
- 11 种尺寸 / 预算（含视频帧预算、放大、缩小、极端长宽比）与 HF 逐像素相同

## 3. 验证结果

| 项目 | 结果 |
|---|---|
| 构建 | 新增代码没有警告（仅剩 2 条 upstream 原有的 deprecated 警告） |
| 65 s 音频分块 | 帧数 2999/2999/502、时间戳、token 数与 Python 参考一致 |
| 视频布局 | 直接调用真实 `build_plan()`，9 个场景与 HF 逻辑的 Python 移植版逐项一致 |
| mel 切片 | 与整条音轨 mel 逐位相同 |
| qwen3a 回归 | mel 和编码器输出均逐位相同 |
| e2e（公开 API） | 图像、音频、长音频、音视频组、纯画面视频全部通过 |
| 真实 ffmpeg 视频 | 6 段合成视频（30 / 29.97 / 25 / 1 fps，10 s 到 3800 s，含无音轨）：帧序号与 HF 逐帧一致，布局与 Python 参考逐项一致，tokenize / encode / decode 通过；文件和内存缓冲两条输入路径结果相同；`llama-mtmd-cli` 实跑通过 |
| 真实权重对比 HF | HF f32（CPU）vs llama.cpp（mmproj f32、文本 bf16），用官方提示词，见下表 |

真实权重对比（输入：`test-1.jpeg` + document 提示词；`test-2.mp3` 转 16 kHz WAV + audio 提示词；8 s 960x540 带音轨视频 + natural_video 提示词）：

| 模态 | token 布局 | 编码器输出（逐行 cos） | 最后位置 logits |
|---|---|---|---|
| 图像 | 868 个 token 完全相同 | 均值 0.99948，最低 0.947；同一份像素输入时均值 0.999993，最低 0.9993 | top1 相同，top10 10/10，KL 2e-9 |
| 音频 | 588 个 token 完全相同 | 均值 1.000000，最低 0.99997 | top1 相同，top10 10/10，KL 6e-9 |
| 视频 | 4891 个 token 完全相同（4080 图像 + 105 音频） | 图像均值 0.99998，最低 0.984；音频最低 0.99998 | top1 相同，top10 10/10，KL 1e-8 |

- 图像的偏差来自 JPEG 解码：llama.cpp 用 stb_image，HF 用 PIL，5% 像素差 1-3 级；缩放本身逐位一致。这是所有视觉模型共有的差异
- 视频帧 ffmpeg 与 decord 解码逐像素相同；剩余的少量低 cos 行分散在各帧，是 f32 累加误差
- 行尾 `\n` 用 `tok_embd` 行插入（3.5.4）在本模型上与 HF 一致

## 4. 已知限制和风险

1. 真实权重只对比了最后一个位置的 logits，没有对比长文本生成；只测了 CPU
2. 抽帧需要 ffmpeg >= 5.1（`-fps_mode`）
3. 视频预算常量 32768 / 1024 写死在 `mtmd.cpp`，没有写进 GGUF
4. 音频长度不再有 128 s 上限，很长的视频内存占用较高（微型模型下 3800 s 视频峰值约 1.6 GB）
5. CUDA / Vulkan / SYCL / OpenCL 的 vision RoPE 仍然只读 2 个 section，这些后端上结果不正确，且不会报错；现有 `test-backend-ops` 的 vision 用例覆盖不到
6. Metal F32 回退路径在大 batch 时偏慢，MiMoVL 也受影响
7. 新增公共 API `mtmd_video_interleave_audio`；分组依赖 bitmap id（`+N` / `+aN`）
8. 改动规模较大，建议先开 issue 和维护者讨论，Metal 改动单独提 PR
