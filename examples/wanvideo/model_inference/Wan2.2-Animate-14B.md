# Wan2.2-Animate-14B 推理脚本与 Pipeline 详解

本文基于 `examples/wanvideo/model_inference/Wan2.2-Animate-14B.py` 以及 `diffsynth/pipelines/wan_video.py` 的实现，对 **Wan2.2-Animate-14B** 的完整推理链路进行逐步拆解，重点解释：

- 脚本做了什么（Animate / Replace 两段示例）
- `WanVideoPipeline.from_pretrained(...)` 如何下载与组装模型
- `pipe(...)`（即 `WanVideoPipeline.__call__`）内部的 **Pipeline Units** 如何处理输入
- Animate 条件（pose / face）在 `model_fn_wan_video` 中如何注入到 DiT
- 为什么示例里要对条件视频做 `[:81-4]` 的截断，以及最终输出帧数为何可能是 `num_frames - 4`

---

## 1. 脚本概览：两段推理（Animate / Replace）

脚本 `Wan2.2-Animate-14B.py` 分为两部分：

1) **Animate**

- 输入：`input_image` + `animate_pose_video` + `animate_face_video`
- 输出：生成一段人物动画视频，动作跟随 pose，面部运动跟随 face 条件

2) **Replace（带 LoRA + Inpaint/Mask）**

- 在 Animate 的基础上额外输入：`animate_inpaint_video` + `animate_mask_video`
- 并加载 `relighting_lora.ckpt`，对 `pipe.dit` 打补丁
- 输出：在背景/遮罩约束下进行“替换/重绘”类的生成（更接近视频 inpaint/replace）

两段推理最终分别保存为：

- `video_1_Wan2.2-Animate-14B.mp4`
- `video_2_Wan2.2-Animate-14B.mp4`

---

## 2. `from_pretrained(...)`：模型下载、重定向与组装

脚本使用：

```py
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[...],
    tokenizer_config=ModelConfig(...),
)
```

### 2.1 ModelConfig 的意义

`ModelConfig(model_id=..., origin_file_pattern=...)` 会：

1. 默认用 ModelScope 下载（可用 `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` 切到 HF）。
2. 默认下载到 `./models/<model_id>/...`（可用 `DIFFSYNTH_MODEL_BASE_PATH` 改基路径）。
3. `origin_file_pattern` 可以是：
   - 单文件名（如 `Wan2.1_VAE.pth`）
   - 通配模式（如 `diffusion_pytorch_model*.safetensors`，可能匹配多个分片文件）
   - 目录前缀（末尾带 `/`，如 `google/umt5-xxl/`）

### 2.2 常见文件重定向（redirect_common_files）

`WanVideoPipeline.from_pretrained(..., redirect_common_files=True)` 会把一些常用的 `.pth` 文件 **重定向到** DiffSynth 提供的 safetensors 转换版，避免在不同模型间重复下载同名权重文件。重定向逻辑在 `diffsynth/pipelines/wan_video.py` 的 `redirect_dict` 中，例如：

- `models_t5_umt5-xxl-enc-bf16.pth` → `DiffSynth-Studio/Wan-Series-Converted-Safetensors/...safetensors`
- `Wan2.1_VAE.pth` → `DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors`

### 2.3 组件装配（fetch_model）

`download_and_load_models(...)` 会用权重文件的“hash 签名”识别模型类型并加载，随后 `from_pretrained` 按名称取回：

- `pipe.dit`（核心扩散模型 DiT；某些 I2V 模型会返回 `dit + dit2` 两个阶段模型）
- `pipe.text_encoder`（UMT5）
- `pipe.vae`（视频 VAE，空间下采样 8x；时间压缩比约 4）
- `pipe.image_encoder`（CLIP 图像编码器；用于 clip_feature 条件）
- `pipe.animate_adapter`（Animate 模块：pose 注入 + face motion 注入）
- 其他：motion controller / VACE / VAP / S2V 音频等（本脚本不使用或不触发）

同时会根据 `vae.upsampling_factor` 调整 `height_division_factor / width_division_factor`，用于输入分辨率对齐。

---

## 3. `pipe(...)` 总体流程：Units → Denoise Loop → Decode

入口是 `WanVideoPipeline.__call__`，核心结构可概括为：

1. **Scheduler 初始化**（FlowMatch）
2. 构造三组输入字典：
   - `inputs_shared`：共享输入（图像/视频/形状/seed 等）
   - `inputs_posi`：正向分支（prompt、face_pixel_values 等）
   - `inputs_nega`：负向分支（negative_prompt、face_pixel_values 等）
3. 依次执行 `self.units`（Pipeline Units），持续更新三组字典
4. 执行 denoising 迭代：`model_fn_wan_video(...)` → `scheduler.step(...)`
5. （Animate / VACE 特殊逻辑）对 latents 做裁剪
6. VAE 解码得到视频帧列表，并保存

---

## 4. 关键形状与“为什么是 4n+1”

### 4.1 `num_frames` 的约束：必须是 `4n + 1`

`WanVideoPipeline` 初始化时设置了：

- `time_division_factor = 4`
- `time_division_remainder = 1`

`WanVideoUnit_ShapeChecker` 会把不满足条件的 `num_frames` 向上取整到 `4n+1`。

因此示例用 `num_frames=81`（= 4×20 + 1）。

### 4.2 VAE 的时间轴映射（非常关键）

`WanVideoVAE.decode` 的时间长度关系是：

- 设 latent 时间长度为 `T_lat`
- 解码后输出帧数 `T_out = 4 * T_lat - 3`

推导得到：

- 想要输出 `T_out` 帧，需要 `T_lat = (T_out + 3) / 4`
- 当 `num_frames=81` 时，`T_lat = (81 + 3) / 4 = 21`

这也是 `WanVideoUnit_NoiseInitializer` 用的长度：

```py
length = (num_frames - 1) // 4 + 1   # 81 -> 21
```

---

## 5. 本脚本触发的 Pipeline Units（逐个解释）

`WanVideoPipeline.units` 很长，但本脚本主要触发以下几个（顺序即执行顺序）：

### 5.1 `WanVideoUnit_ShapeChecker`

- 输入：`height, width, num_frames`
- 行为：把 `height/width` 对齐到 `height_division_factor/width_division_factor` 的整数倍；把 `num_frames` 对齐到 `4n+1`

### 5.2 `WanVideoUnit_NoiseInitializer`

- 输入：`height, width, num_frames, seed, rand_device`
- 输出：`noise`（初始高斯噪声 latent）
- 形状：
  - `z_dim = pipe.vae.model.z_dim`（Wan VAE 默认 16）
  - 空间 latent：`H_lat = height / vae.upsampling_factor`，`W_lat = width / vae.upsampling_factor`（upsampling_factor=8）
  - 时间 latent：`T_lat = (num_frames - 1)//4 + 1`（81 -> 21）
  - `noise.shape = (1, z_dim, T_lat, H_lat, W_lat)`

### 5.3 `WanVideoUnit_PromptEmbedder`（seperate_cfg）

- 正向：`prompt` → `context`
- 负向：`negative_prompt` → `context`
- 使用 `pipe.tokenizer` + `pipe.text_encoder` 得到文本上下文向量
- 注意：
  - 当 `cfg_scale=1` 时，负向分支不会单独计算（runner 会复用正向输出）

### 5.4 `WanVideoUnit_InputVideoEmbedder`

- 本脚本 `input_video=None`，因此：
  - `latents = noise`
- 若 `input_video` 不为空，会先 VAE encode 得到 `input_latents`，并按 `denoising_strength` 加噪作为初始 `latents`（v2v）

### 5.5 `WanVideoUnit_ImageEmbedderVAE`（I2V 条件：生成 `y`）

> 只要 `input_image` 不为空且 `pipe.dit.require_vae_embedding=True`，就会触发。

- 输入：`input_image, num_frames, height, width, tiled, tile_size, tile_stride`
- 行为：
  1. 把 `input_image` resize 到 `(width, height)`，并预处理到 `[-1,1]`
  2. 构造一个长度为 `num_frames` 的“视频张量”：
     - 第 0 帧为 `input_image`
     - 后续帧全部为 0（若提供 `end_image`，则最后一帧为 `end_image`）
  3. 通过 VAE encode 得到 `y_latents`
  4. 构造同时间长度的 mask（4 通道形式），与 `y_latents` 在通道维拼接
- 输出：`y`，形状约为：
  - `y.shape = (1, 4 + z_dim, T_lat, H_lat, W_lat)`

### 5.6 `WanVideoUnit_ImageEmbedderCLIP`（生成 `clip_feature`）

> 只要 `input_image` 不为空且 `pipe.dit.require_clip_embedding=True`，就会触发。

- 输入：`input_image, height, width`
- 输出：`clip_feature`
- 行为：CLIP 编码输入图像得到图像上下文，用于在 `model_fn` 中拼到 text context 上

### 5.7 `WanVideoUnit_AnimatePoseLatents`（pose 条件 → `pose_latents`）

- 输入：`animate_pose_video`
- 行为：`preprocess_video` → `vae.encode`
- 输出：`pose_latents`

这里对应脚本里非常关键的一句：

```py
animate_pose_video = VideoData(...).raw_data()[:81-4]
```

原因见第 6 节：**Animate 的 pose_latents 会加到 `x[:, :, 1:]` 上**，因此它的 latent 时间长度必须是 `T_lat - 1`。

### 5.8 `WanVideoUnit_AnimateFacePixelValues`（face 条件 → `face_pixel_values`）

- take-over unit：直接读写 `inputs_posi/inputs_nega`
- 正向：`face_pixel_values = preprocess_video(animate_face_video)`
- 负向：`face_pixel_values = -1` 的全零张量（用于 CFG；当 `cfg_scale=1` 时基本不参与）

face 条件不是先过 VAE，而是以像素域输入，在 `WanAnimateAdapter` 内部提取 motion 表征。

### 5.9 `WanVideoUnit_AnimateInpaint`（仅 Replace 段触发，覆盖 `y`）

Replace 段脚本额外传入：

- `animate_inpaint_video`
- `animate_mask_video`

该 unit 会重新构造 `y`（**覆盖 5.5 生成的 `y`**）：

- 先对 `animate_inpaint_video` 做 VAE encode 得到背景 latent（按时间轴）
- 再对 `input_image` 做 VAE encode 得到 reference latent（单帧）
- 把 `animate_mask_video` 预处理到 `[0,1]` 后取 `1 - mask`，并插值到 latent 分辨率，作为 mask 通道
- 最终把 `reference + background` 拼成一个时间长度为 `1 + T_bg_lat` 的 `y`

Mask 的直观含义：

- 该实现内部使用的是“**已知/保留区域 = 1**”的约定（类似 5.5 里第一帧 mask=1）
- 因此会做 `1 - mask`：通常你提供的 mask 若是“白色表示要替换/重绘区域”，那么 `1-mask` 就变成“白色区域=0（未知），黑色区域=1（保留）”

---

## 6. Animate 为什么要 `[:81-4]`？以及输出帧数为什么是 `num_frames-4`

### 6.1 `pose_latents` 对齐：需要 `T_lat - 1`

在 `model_fn_wan_video(...)` 中，Animate 的注入点是：

```py
# 在 patch embedding 后
if pose_latents is not None and face_pixel_values is not None:
    x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
```

而 `WanAnimateAdapter.after_patch_embedding` 的关键逻辑是：

```py
pose_latents = self.pose_patch_embedding(pose_latents)
x[:, :, 1:] += pose_latents
```

这意味着：

- `x` 的 latent 时间长度是 `T_lat`
- `pose_latents` 必须匹配 `x[:, :, 1:]`，即时间长度必须是 `T_lat - 1`

当 `num_frames=81` 时，`T_lat=21`，所以 `pose_latents` 需要 `20` 个 latent timestep。

而 `VAE.encode` 的时间映射与 decode 相反，满足：

- 若输入像素帧数为 `T_in`
- 得到的 latent 时间长度约为 `T_lat = (T_in + 3)/4`

要得到 `T_lat = 20`，需要 `T_in = 4*20 - 3 = 77` 帧，
这就是脚本里 `[:81-4]`（`81-4 = 77`）的来源。

同理：

- `animate_face_video`
- `animate_inpaint_video`
- `animate_mask_video`

示例都截到 77 帧，是为了在 latent 时间尺度上与 `x[:, :, 1:]` / mask 对齐。

### 6.2 为什么最终输出是 `num_frames - 4`

`WanVideoPipeline.__call__` 在 denoise 完成后有一段特殊逻辑（源码注释 `# VACE (TODO: remove it)`）：

```py
if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
    f = 1
    inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
```

即当进入 Animate（同时提供 pose + face）时，会把 latent 的第一个时间 slice 去掉：

- 原本 `T_lat = 21`（对应 81 帧）
- 裁剪后 `T_lat = 20`
- 解码输出帧数变为 `4*20 - 3 = 77 = 81 - 4`

因此：

- **你传入的 `num_frames=81` 更像是“内部计算长度”（用于对齐/注入），实际返回帧数是 `num_frames-4`。**

> 提示：若你希望最终视频是 81 帧，需要自行在外部做拼接/补帧，或修改该裁剪逻辑（不建议在不了解 VACE/Animate 兼容目的时直接删除）。

---

## 7. Denoising 主循环：`model_fn_wan_video` 里 Animate 条件如何生效

### 7.1 输入汇总

在 unit 阶段结束后，关键张量大致是：

- `latents`: `(1, z_dim, T_lat, H_lat, W_lat)`
- `context`: 文本上下文
- `y`: VAE 条件（I2V / Replace inpaint mask）
- `clip_feature`: CLIP 图像特征（可选，取决于模型）
- `pose_latents`: `(1, z_dim, T_lat-1, H_lat, W_lat)`（由 pose 视频编码得到）
- `face_pixel_values`: `(1, 3, T_in, H, W)`（由 face 视频预处理得到）

### 7.2 `model_fn_wan_video` 的关键路径（与 Animate 相关）

`model_fn_wan_video` 内部主要步骤：

1. 计算时间嵌入 `t_mod`，并做 `context = dit.text_embedding(context)`
2. 若 `y` 存在且 `dit.require_vae_embedding=True`：
   - `x = cat([latents, y], dim=1)`
3. 若 `clip_feature` 存在且 `dit.require_clip_embedding=True`：
   - `clip_embedding = dit.img_emb(clip_feature)`
   - `context = cat([clip_embedding, context], dim=1)`
4. `x = dit.patchify(x, ...)`
5. **Animate 注入（第一次）**：
   - `x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)`
6. 进入 DiT blocks 循环
7. **Animate 注入（按 block）**：
   - 每个 block 后调用 `animate_adapter.after_transformer_block(block_id, x, motion_vec)`
   - 该实现里是每 5 层注入一次 face adapter（残差形式）
8. `x = dit.head(x, t)`，再 `unpatchify` 回到 `(B, C, T_lat, H_lat, W_lat)` 作为 `noise_pred`

### 7.3 CFG 与本脚本的 `cfg_scale=1`

`WanVideoPipeline.__call__` 中：

- `cfg_scale == 1.0` → **不会**计算负向分支、也不会做 guidance 合并
- 本脚本两段推理都设为 `cfg_scale=1`，因此：
  - `negative_prompt` 对结果基本无影响
  - 推理速度更快、显存更省

如果你希望提示词更“强约束”，通常需要设置 `cfg_scale > 1`。

---

## 8. Replace 段额外步骤：LoRA 加载与注入

Replace 段多了以下逻辑：

1) 下载 LoRA：

```py
snapshot_download(
  "Wan-AI/Wan2.2-Animate-14B",
  allow_file_pattern="relighting_lora.ckpt",
  local_dir="models/Wan-AI/Wan2.2-Animate-14B",
)
```

2) 加载权重并打到 `pipe.dit`：

```py
lora_state_dict = load_state_dict(... )["state_dict"]
pipe.load_lora(pipe.dit, state_dict=lora_state_dict)
```

`BasePipeline.load_lora(...)` 会把 LoRA state dict 转成统一格式，并按当前 VRAM 管理配置选择：

- **fuse** 到 base model（直接改权重；默认常见路径）
- 或 **hotload**（仅当模块启用 VRAM 管理且支持 AutoWrappedLinear 的 LoRA 权重列表）

注意：

- 脚本只对 `pipe.dit` 加载 LoRA。若某模型存在 `dit2`（双阶段），可能需要同时对 `pipe.dit2` 也加载相同 LoRA（否则两阶段风格不一致）。

---

## 9. 运行与排错建议

### 9.1 运行脚本

在仓库根目录执行：

```bash
python examples/wanvideo/model_inference/Wan2.2-Animate-14B.py
```

首次运行会下载：

- 模型权重到 `./models/...`
- 示例数据到 `./data/examples/wan/animate/...`

### 9.2 常见问题

1) **shape 不匹配 / stack 报错**

- 你传入的 `animate_pose_video / animate_face_video` 每一帧分辨率必须一致
- 且最好与 `height/width` 一致（或你在读取时先 resize）

2) **帧数对不上**

- Animate/Replace 的条件视频推荐长度为：`num_frames - 4`
- 并留意最终输出可能是：`num_frames - 4`（第 6.2 节）

3) **显存不够**

- 保持 `tiled=True`（脚本已开启）
- 调小 `height/width` 或 `num_frames`
- 调小 `num_inference_steps`
- 或使用 `examples/wanvideo/model_inference_low_vram/` 下的脚本版本

---

## 10. 速查：本脚本参数与默认值

- `torch_dtype=torch.bfloat16`，`device="cuda"`
- `num_frames=81`（内部长度；最终输出可能 77）
- `height=720`，`width=1280`
- `num_inference_steps=20`
- `cfg_scale=1`（关闭 CFG）
- `tiled=True`（VAE encode/decode 平铺以省显存）
- `seed=0`，`rand_device="cpu"`

