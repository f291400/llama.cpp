# Youtu-Parsing-Omni 实现导读

对照基线 `31288c9c`（`hexagon: FLASH_ATTN_EXT improvements...`）到当前工作区的净改动。这中间有 3 个已提交的 commit，加上尚未提交的整理和对齐。下面讲的是整理之后的行为，也就是现在代码实际在做什么。

相关材料：

- 官方模型卡：<https://huggingface.co/tencent/Youtu-Parsing-Omni>
- 改动汇报（验证数字、已知限制）：`youtu-omni-report.md`
- 冗余检查记录：`youtu-omni-commit-check.md`

## 1. 这批改动在做什么

Youtu-Parsing-Omni 是腾讯 5B 的全模态解析模型。输入可以是文档页、自然图、图表、音频，或带音轨的视频。输出是一份按任务提示词选定的 JSON。

llama.cpp 里原来没有这条路径。这次改动把它接进现有的两段式推理：

1. 文本解码器继续用已经支持的 DeepSeek-V2（MLA）图。权重类名是 `YoutuVITAForCausalLM`，转换时注册到 `DeepseekV2Model`。
2. 图像、音频、视频不进文本图。它们由 `tools/mtmd` 编成 embedding，再替换 prompt 里对应位置的 token。

所以这不是新的 `src/models/*.cpp` 文本架构。新代码集中在转换脚本、mmproj 计算图，以及 mtmd 的分词、抽帧和编码调度。

Metal 上另有两处配套：vision RoPE 要读满 4 个 section，以及 `GGML_PREC_F32` 的矩阵乘要真的用 f32 累加。SwiGLU 的 down 投影在 f16 累加时会溢出，和 MiMo-VL 是同一类问题。

## 2. 它是基于什么模型改的

官方说法和 llama.cpp 的接法要分开看。官方描述的是训练时的模型设计，代码里看到的是推理时复用了仓库里的哪一套实现。

### 2.1 官方模型本身

模型卡把结构写成两截：

- **Youtu-Omni-Encoder**：图像、音频、视频共用一个双向 Transformer。官方写明它从「一个预训练文本语言模型」初始化，用来加强读文字的能力。像素和 log-mel 各有一层很薄的 stem。位置编码是同一套 `(t, h, w)`。只有 stem 和 merger 按模态分开。
- **Youtu-LLM 解码器**：吃编码器吐出的 token，加上文本提示，生成 OmniSchema JSON。

和常见全模态模型的差别在编码器，不在「再挂一个视觉塔、再挂一个音频塔」。Qwen3-Omni 是视觉塔和音频塔各算各的，两路特征进语言模型之后才相遇。这里两路在编码器内部就在同一套注意力里相遇。融合层会让一帧看到同一时刻的声音。非融合层则让每一帧、每一段音频各自形成一个注意力窗口。

类名里的 `VITA` 是这条产品线的名字（`YoutuVITAForCausalLM`，`model_type = youtu_vita`）。2024 年那篇 VITA 论文是 Mixtral 8x7B 加独立视觉塔和音频塔，用来做交互。Parsing-Omni 没有沿用那套骨架。

技术报告在模型卡里标成 2026-10 发布，PDF 在官方仓库。公开材料没有写出编码器初始化用的是哪一个具体 checkpoint，只说是预训练文本模型。

### 2.2 在本仓库里分别接到了谁

| 部分 | 结构从哪来 | llama.cpp 里怎么接 |
|---|---|---|
| 文本解码器 | Youtu-LLM，MLA 布局和 DeepSeek-V2 相同 | `conversion/deepseek.py` 的 `DeepseekV2Model`，`arch = deepseek2`。前代 `YoutuForCausalLM`、`YoutuVLForConditionalGeneration` 已经走这条路径 |
| 编码器 block | Qwen2/Qwen3 那种 decoder block：RMSNorm、SwiGLU、GQA、`q_norm`/`k_norm`、独立 q/k/v/o，然后改成双向注意力 | 新图 `clip_graph_youtu_omni`，层名映射到已有的 vision encoder 张量 |
| 图像 stem 和 2x2 merge | 和 Qwen2-VL 同一套 patch conv，再按 2x2 合并块重排 | `build_image_tokens()` 直接用 qwen2vl 的 permute 写法 |
| 音频 stem | 和 Qwen3-Omni / Qwen3-ASR 相同：Whisper log-mel，100 帧一块，3 次 stride-2 conv，一块 13 个 token | 和 `qwen3a` 共用 `clip_graph::build_audio_conv2d_stem` |
| 图像动态分辨率 | Qwen-VL 一族的像素预算思路 | 对齐的是 HF `process_native_v2`：等比缩放后在右侧和下方补 127 灰边，对齐到 32 的倍数 |
| 视频融合、三分段 RoPE、行尾换行 | 这个模型自己的设计 | 新逻辑，没有现成模型可以整个套用 |

前代 **Youtu-VL**（仓库里已有的 `youtuvl`）文本侧同样是 DeepSeek-V2，视觉侧是另一套塔：window attention、不规则的 full-attention 层号，预处理也不同。Parsing-Omni 换掉了那座视觉塔，并加上音频和音视频融合。benchmark 表里的 **Youtu-Parsing（2.5B）** 是上一代文档解析模型，没有这条共享全模态编码器。

文本侧有一处和原版 DeepSeek 配置不一致，转换时单独处理了。DeepSeek 把 `num_key_value_heads` 写成 MLA 的头数。Youtu 把它写成 GQA 的 KV 头数。`kv_b_proj` 因此按张量形状决定按 `n_head` 还是 `n_head_kv` 切开，原有 DeepSeek 权重的切分结果不变。

### 2.3 用一张图看继承关系

```text
DeepSeek-V2 (MLA)
    |
    +-- Youtu-LLM 文本解码器
            |
            +-- Youtu-VL          视觉塔另做（window attention）
            |
            +-- Youtu-Parsing-Omni
                    |
                    +-- 解码器：沿用 Youtu-LLM / DeepSeek-V2
                    +-- 编码器：从预训练文本 LM 初始化的双向 Transformer
                         |-- 图像前端：Qwen2-VL 式 patch + 2x2 merge
                         |-- 音频前端：Qwen3-Omni / Qwen3-ASR 式 conv2d stem
                         +-- 融合层和 (t, h, w) RoPE：本模型新增

Qwen3-Omni：视觉塔和音频塔分开，在 LLM 里汇合。前端相似，编码器不是同一个。
2024 VITA：Mixtral + 双塔。只共享产品线名字。
```

## 3. 一次推理怎么走

mtmd 把用户输入拆成文本 chunk、图像 chunk、音频 chunk。文本 chunk 交给 `llama` 解码。图像和音频 chunk 先变成 embedding，再按 chunk 在序列里的位置填进去。

```text
输入
  |-- 文本 --------------------------> llama decode (DeepSeek-V2)
  |-- 图片
  |     预处理 (像素预算, 灰边)
  |     <|vision_start|> H*W \n
  |     每行图像 embedding 后面接一个 "\n" 的词表向量
  |     <|vision_end|>
  |-- 音频
  |     Whisper mel, 按 2 s / 30 s 切块
  |     每块 <|audio_start|> ... <|audio_end|>
  |     多于一块时, 块前加 HH:MM:SS
  +-- 视频 (ffmpeg 抽帧 + 抽音轨)
        <|video_start|>
        按 1.5 s / 29.5 s 打成若干组
        每组: [时间戳] 帧 ... [时间戳] 音频 ...
        <|video_end|>
        同一组的帧和音频进同一张计算图
```

视频能合成一组，靠的是 bitmap id。helper 给同一文件的帧写成 `{hash}+0`、`{hash}+1`，音频片段写成 `{hash}+a0`、`{hash}+a1`。`mtmd_youtu_index_pairs()` 用这个前缀把它们收成一组。`video_group_attention` 为真时，组只在「音频后面又出现帧」（A 到 I）处切开。没有音轨的视频仍然是一组，只是组里全是帧。

编码时，一组里的第一个 chunk 把整组送进 `clip_graph_youtu_omni::build_group()`。图的输出是「先全部视觉 embedding，再全部音频 embedding」。`mtmd_encode_youtu()` 按事件把它们拆开，后面的 chunk 先放进 `youtu_stash`，轮到那个 chunk 时再取出。调用方仍然按 prompt 顺序逐个 chunk 取 embedding。

## 4. 编码器

权重在 HF 里都在 `model.omni_model.*`。转换时文本文件丢掉这一段（mmproj 负责写）。mmproj 里 Transformer 只存一份，张量名用 vision 的前缀。音频上下文加载时也读这套 `v.*` 权重，再额外带上 conv stem 和 audio merger。见 `clip.cpp` 里 `PROJECTOR_TYPE_YOUTU_OMNI` 对音频 modality 把 prefix 改成 `"v"` 的那段。

一层的计算和普通 decoder block 相同，只是注意力是双向的：

```text
RMSNorm -> GQA (q_norm, k_norm, RoPE) -> 残差
RMSNorm -> SwiGLU -> 残差
```

down 投影强制 f32 累加（`build_mm` 里看到 `ff_down_w` 就 `ggml_prec_set_acc(..., GGML_PREC_F32)`）。

### 4.1 图像 token

`build_image_tokens()`：

1. 输入是 planar RGB，`[nx, ny, 3]`。
2. patch conv，stride = patch size（配置里是 16）。
3. 用和 Qwen2-VL 一样的 permute，把 token 排成 2x2 merge 顺序：按 2x2 块扫描，块内是 `(0,0), (1,0), (0,1), (1,1)` 这种相邻 patch。
4. 可选的 patch bias。

过完 Transformer 之后，`build_vision_merger()` 把每 4 个 patch 拼成一个向量，做 RMSNorm，再过一个无 bias 的 GELU MLP。`spatial_merge_size = 2`，所以输出 token 数是 `(H/32) * (W/32)`。

独立图像的 RoPE：`t = 0`，`h`、`w` 是 patch 坐标，`freq_scale = 1`。整张图一个双向窗口。

prompt 布局和 Qwen-VL 不同。Qwen-VL 是 `<|vision_start|>` 后直接跟 image pad。这里是：

```text
<|vision_start|>{高度}*{宽度}\n
(第 0 行 embedding) \n
(第 1 行 embedding) \n
...
<|vision_end|>
```

`H*W` 是缩放并补边之后的像素尺寸。行尾的 `\n` 不经过视觉塔，而是从文本模型的 `tok_embd` 取出这个 token 的行，插到每一行 embedding 后面。视频帧走同一套编码，但 `H*W` 后面不加这个 `\n`，帧和帧之间用时间戳文本隔开。

预处理 `process_native_v2`（`mtmd_image_preprocessor_youtu_omni`）：

- 像素数高于预算时按面积等比缩小，低于 `min_pixels` 时放大。
- 右边和下边补 `(127, 127, 127)`，使高宽都是 `patch_size * merge`（32）的倍数。
- 补边后的画布如果超出预算，改为向下取整，再把内容缩小进画布。
- 缩放算法是 bicubic。

视频里每一帧还有单独预算：`min(32^2 * 32768 / 帧数, 32^2 * 1024)`。帧越多，单帧越小。这两个常数目前写在 `mtmd.cpp`，没有进 GGUF。

### 4.2 音频 token

前端和 `qwen3a` 是同一函数 `build_audio_conv2d_stem()`：

1. log-mel，128 bin，16 kHz，FFT 400，hop 160。帧数规则对齐 `WhisperFeatureExtractor(padding="do_not_pad")`：`n_samples / 160`，丢掉 STFT 的最后一帧，再做 max-8 截断。
2. 时间维补到 100 的倍数。每 100 帧是一块（配置里的 `n_window = 50`，窗口长度是它的 2 倍）。
3. 三次 conv2d，stride 2，padding 1，GELU。100 帧变成 13 个时间步，mel 维也被压小。
4. `linear_proj` 投到编码器隐藏维。
5. 补齐的帧只影响最后一块的长度。有效 token 是一个前缀，用一次 `ggml_view` 裁掉，对应参考实现里的 `padded_mask_after_cnn`。长度公式在 `clip_youtu_omni_audio_tokens()`。

独立音频整块是一个双向窗口。时间坐标 `t = 0 .. n-1`，`h = w = 0`。早期实现曾按 13 个 token 分段并让 `t` 每段归零，那和 HF 不一致，已经去掉。

分块规则照搬参考实现，包括浮点舍入：最短 2 秒、最长 30 秒。因为帧数算法的关系，标称 30 秒的一块实际是 2999 帧。每块自己包 `<|audio_start|> ... <|audio_end|>`。多于一块时，块前加 `HH:MM:SS`。

merger 和视觉不同。`temporal_merge_size = 1`，所以不做时间合并：RMSNorm 之后一个无 bias 的 GELU MLP（`build_audio_merger`）。

### 4.3 视频：抽帧、分块、融合

抽帧在 `mtmd-helper.cpp`。最多 128 帧，并且不超过 1 fps。源帧序号是 `int(i * step)`，ffmpeg 用 `select='eq(n,floor(ceil(n/step)*step))'` 加 `-fps_mode passthrough` 去取。时间戳是 `源帧序号 / round(fps)` 秒。需要 ffmpeg >= 5.1。

每个帧拥有「这一帧到下一帧」之间的音频。分块规则：

- 累积音频达到 1.5 秒就收成一块。1 fps 时典型结果是两帧再跟大约 2 秒音频。
- 单帧自带的音频超过 29.5 秒时，先输出这一帧，再把音频按 29.5 秒切开。
- 没有音轨时，每一帧单独出现，仍然包在 `<|video_start|> ... <|video_end|>` 里，并走融合组编码。

同一段视频的音频先拼成整条音轨，只算一次 mel，再按片段切片。这样全局的 max-8 归一化和 HF 一致。如果每段单独算 mel，截断阈值会不同。

一组的计算图 `build_group()`：

1. 按事件顺序做 stem，拼成一条 token 序列。
2. 融合层：不使用 block mask，位置用共享时间轴 `positions`。
3. 其他层：block-diagonal mask，每个事件只看自己，位置用 `positions_local`。
4. 编码结束后把视觉 token 和音频 token 分开，各自过自己的 merger，再按「先视觉、后音频」接回去。

`clip.vision.fusion_layer` 是长度为层数的 0/1 数组，来自配置 `video_fusion_layer_freq`。`None` 表示每层都融合。整数 `N` 表示 `i % N == 0` 的层融合。转换时必须写成 GGUF 数组，加载时长度不对会直接报错。

### 4.4 RoPE 为什么要一份 freq_factors

head 的旋转维是 `head_dim / 2`。它被切成三段，长度大约各占 1/3，对应 `(t, h, w)`。`head_dim = 128` 时旋转维是 64，三段是 `{21, 21, 22, 0}`。

每一段使用自己的 theta 底数，也就是段内第 `k` 个频率是 `base^(-k / d_section)`。`ggml_rope_multi(VISION)` 会在段与段之间把 theta 归零，但指数仍然按整段 `n_dims` 来缩放。代码因此预计算一个 `freq_factors` 向量，把缩放乘回去：

```text
factor[k] = base ^ (k * (1/d_section - 2/n_rot))
```

独立图像和独立音频的 `freq_scale` 是 1。视频组是 0.5。原因是一帧在共享时间轴上占 `100/8 = 12.5` 个音频 token：hop 160 的 mel 大约每秒 100 帧，三次 stride-2 的理论步长是 8，所以一帧对应 12.5。ggml 的 RoPE 位置是 int32，组图里所有通道都存成「半步」，12.5 变成 25，再用 `freq_scale = 0.5` 还原。第 `i` 帧的时间是 `i * 25` 个半步。音频 token 在融合层上从当前 cursor 连续编号，在非融合层上从 0 重新编号。

这和 Qwen2-VL 的 M-RoPE 不是同一组 section。Qwen2-VL 图里常见的是四段、并且把 h/w 复制到多个 section。Youtu 是三段，第三段非零。只把 `sect_0 + sect_1` 当成旋转维的后端，在第 42 维之后会绕回 T 段，结果是错的，而且不会报错。CPU 参考实现读四段。这次 Metal kernel 也改成读四段。CUDA、Vulkan、SYCL、OpenCL 还没有改。

## 5. 代码地图

从「用户丢进一张图或一段视频」往下看，按这个顺序读。

| 文件 | 读什么 |
|---|---|
| `conversion/__init__.py` | `YoutuVITAForCausalLM`：文本映射到 `deepseek`，mmproj 映射到 `youtu_omni` |
| `conversion/deepseek.py` | 丢掉 `model.omni_model.*`；`kv_b` 按形状切分 |
| `conversion/youtu_omni.py` | 从 `omni_config` 读视觉和音频配置；写 fusion 层；patch embedding 从 Linear 还原成 conv kernel |
| `conversion/base.py` | `audio_config` 是 JSON null，取配置时要当成空字典。tokenizer 类名 `TokenizersBackend` 在仓库固定的 transformers 里没有，退回 `PreTrainedTokenizerFast` |
| `gguf-py/gguf/tensor_mapping.py` | `model.omni_model.*` 到现有 GGUF 张量名。音频 merger 复用 `mm_norm_pre` / `mm_1` / `mm_2` |
| `tools/mtmd/models/youtu-omni.cpp` | 计算图本身：stem、RoPE、融合 mask、两个 merger |
| `tools/mtmd/models/qwen3a.cpp` | 音频 conv stem 的另一侧调用方，回归时应对齐 |
| `tools/mtmd/clip-impl.h` | `clip_youtu_omni_audio_tokens`、半步位置、`clip_youtu_event` |
| `tools/mtmd/clip.cpp` | 加载 GGUF、填 positions / mask / rope factors、RGB 转 planar |
| `tools/mtmd/mtmd.cpp` | 特殊 token、像素预算、分块包裹、组索引、stash、行尾 `\n` |
| `tools/mtmd/mtmd-image.cpp` | `process_native_v2` |
| `tools/mtmd/mtmd-audio.cpp` | mel 和 2 s / 30 s 分块；视频音轨的 `preprocess_pieces` |
| `tools/mtmd/mtmd-helper.cpp` | ffmpeg 抽帧、抽 PCM、`build_plan()` |
| `ggml/src/ggml-metal/kernels/rope.metal` | 四段 vision RoPE |

公共 API 增加了 `mtmd_video_interleave_audio`。打开之后，helper 不按普通视频那样只吐帧，而是吐出 `build_plan()` 里的时间戳文本、帧和音频片段。

## 6. 读代码时容易混的几点

这些都是和参考实现对齐之后的行为。早期 commit 里有过另一套做法，工作区里已经改掉。

- 独立音频是整块一个窗口。13 只是 conv 把 100 帧变成多少个 token，不是注意力窗口的大小。
- 视频时间轴上的 12.5 是 `100/8`，是理论下采样率。conv 因为 padding，满 100 帧实际产出 13 个 token。融合层用 12.5 对齐帧和声音，token 个数仍按 13 算。
- 组图的 `freq_scale` 是 0.5，位置数组里的数是真实坐标的两倍。单独看 positions 会以为时间被放大了。
- 一组编码的输出顺序是「所有帧的 merger 结果，再所有音频的 merger 结果」，不是事件交错的顺序。stash 再按 prompt 顺序交回去。
- 行尾 `\n` 取的是词表 embedding 行。这个模型没有 embedding LoRA，也没有 `n_embd_inp != n_embd`，所以和先 decode 再查表一致。换一个有这些特性的模型时，这个捷径不能照搬。
- 图像预处理对齐的是 HF `process_native_v2`。vLLM 插件里的 `smart_resize` 是另一套，两边像素不会相同。
- JPEG 用 stb_image 解码，HF 用 PIL。同一张图会有少量像素差 1 到 3。缩放本身可以逐像素对齐。比编码器时要先排除解码差异。

## 7. 现在还不能假设成立的事

- CUDA、Vulkan、SYCL、OpenCL 的 vision RoPE 仍然只使用前两段。这些后端上 Youtu 的位置编码是错的。
- 视频的 32768 和 1024 token 预算写死在 `mtmd.cpp`。
- 音频不再截断到 128 秒。很长的视频会把整条 mel 和分组图都放在内存里。
- 抽帧依赖 ffmpeg，并且需要 `-fps_mode`（ffmpeg 5.1 及以上）。
- 真实权重对比过图像、音频、短视频的 token 布局和最后一个位置的 logits，对比平台是 CPU。长文本生成还没有和 HF 对过。
