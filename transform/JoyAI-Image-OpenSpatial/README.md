# JoyAI-Image-OpenSpatial -> Pangu ML 转换说明

本文档给出可执行级别的映射规则，目标是将 `jdopensource/JoyAI-Image-OpenSpatial` 转成 `pangu_ml` 格式（规范见 `docs/pangu_ml_data_schema.md`）。

## 1) 源数据结构（Hugging Face）

- 数据集：`jdopensource/JoyAI-Image-OpenSpatial`
- `config=default`
- `split=train`
- `num_rows_total=2,352,448`（HF datasets-server 返回）

字段：

- `conversations`: `List[{"from": string, "value": string}]`
- `id`: `string`
- `data_source`: `string`
- `images`: `List[{"bytes": binary/base64-like string, "path": string|null}]`
- `type`: `string`
- `meta_info`: `string`（JSON 字符串）

## 2) 关键问题与落地决策

### A. 对话文本里的图像占位符 `<image>` 怎么处理？

观察到首条样本的 `conversations[0].value` 包含 `<image>`。转换脚本固定策略：

- 先将真实图像写入 Pangu 第一条 `user.content` 的 `type=image` 部件；
- 再将第一条 `user` 文本中的 `<image>` 去掉；
- 这样避免“同一语义被 image 部件和 `<image>` 文本重复表达”。

### B. Pangu 的 `id` 用什么？

`id` 可能在不同 `data_source` 下重复，因此脚本固定使用：

- `"{data_source}__{id}"`

另外，每条输出样本会新增与 `id` 并列的顶层字段：

- `category`: 由规则提取/推断出的子任务类别（如 `distance.single_view`）

### C. `images.path` 非空时要不要用？

不用。转换输出不会沿用源数据里的 `images.path` 作为 tar 内相对路径。

原因：

- `images.path` 是源数据组织细节，不适合作为目标格式稳定命名规则；
- 我们采用统一全局命名规则，保证跨分片/跨来源可控和一致；
- 当前数据里 `bytes` 可直接解码，不需要依赖 path 回退机制。

### D. 图像命名规则和 tar 内目录层级

`{id}.jpg` 对多图不够，且可能冲突。脚本固定采用全局 flat 命名；**编码与扩展名**由 Pillow 识别的源格式决定：

- **Pillow 识别为 PNG**（`format == PNG`）：写入 **PNG**（`optimize=True`），`relative_path = …_{idx:02d}.png`，jsonl 中 `format` 为 `image/png`（尽量保留透明通道）。
- **其它格式**（JPEG、WebP、GIF 等）：转为 **JPEG quality=95**，`relative_path = …_{idx:02d}.jpg`，`format` 为 `image/jpeg`。
- 不按 sample 建子目录，避免 tar 内大量小目录。
- `width/height` 固定从解码后的图像解析，不使用 `meta_info` 尺寸字段。

### E. 会不会有“对话轮次不合法”的样本？

脚本强校验：

- 角色映射：`human -> user`，`gpt -> assistant`
- 轮次必须严格交替，且第一轮必须是 `user`
- 不合法样本会跳过并计数到 `skipped_samples`

### F. 子任务 category 如何提取？

由于源数据没有显式 `category` 字段，脚本采用 **OpenSpatial 管线对齐的问句匹配策略**：

- 对第一轮 `human` 问句做归一化（去 `<image>`、统一空白、小写）；
- 优先匹配由 annotation 代码最终产出的问句形态（而不是只看 prompt template 原文）；
- 仅输出“主任务 + 视图范围”的粗粒度 category，不输出细分子类；
- 主任务集合：`grounding_3d`、`correspondence`、`distance`、`position`、`size`、`depth`、`counting`、`3d_scene_caption`；
- 视图范围后缀：
  - `.single_view`（单图）
  - `.multi_view`（多图）
- 若无法命中明确规则，回退为：
  - `unknown.single_view`
  - `unknown.multi_view`

`metadata.json` 中 `category_field_detected` 会标记为 `inferred_subtask_from_prompt`，表示 category 来自规则推断而非源字段。

## 3) 第一条真实样本 + 手工映射示例

以下原始内容来自 `rows(offset=0,length=1)`，仅对 `images[0].bytes` 做了截断展示。

### 3.1 原始样本（节选）

```json
{
  "id": "a1d03f7f-cabf-482e-8fd0-67f2dfd7464f",
  "data_source": "arkitscenes",
  "type": "unknow",
  "conversations": [
    {
      "from": "human",
      "value": "Here are the detailed camera parameters for the image.\n                Camera intrinsic parameters: Horizontal fov, hfov=62.26, and vertical fov, vfov=48.74. Image width=256 and height=192. We do not consider distortion parameters here. \n                Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location.\n                We take the camera coordinate system as the world coordinate system.\n\n                3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size,  pitch, yaw, roll]\n                * x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. z_center is the depth of the object in space.\n                * x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.\n                * pitch, yaw, roll: Euler angles representing rotations around the X, Y, and Z axes, respectively. Euler angles are expressed in radians.\n                * The rotation order of Euler angles is zxy.\n\n                Output a json list where each entry contains the object name in \"label\" and its 3D bounding box in \"bbox_3d\". <image> Find the 3D bounding box that encapsulates the table in this visual representation."
    },
    {
      "from": "gpt",
      "value": "[{'bbox_3d': [-0.85, 0.32, 2.12, 0.56, 0.82, 0.73, -2.75, 0.1, -2.4], 'label': 'table'}]"
    }
  ],
  "images": [
    {
      "bytes": "iVBORw0KGgoAAAANSUhEUgAAAQAAAADACAIAAABkjyoxAAEAAElEQVR4nJz9...<omitted>...==",
      "path": null
    }
  ],
  "meta_info": "[{\"resized_width\": -1, \"resized_height\": -1, \"width\": 256, \"height\": 192}]"
}
```

### 3.2 映射后的 Pangu 样本（手工展开示例）

> 说明：图像路径采用固定 flat 规则；下例源图为 PNG，故为 `…_00.png` 且 `format=image/png`（若为 JPEG 源则对应 `.jpg` / `image/jpeg`）。

```json
{
  "meta_prompt": [""],
  "id": "arkitscenes__a1d03f7f-cabf-482e-8fd0-67f2dfd7464f",
  "category": "grounding_3d.single_view",
  "data": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": {
            "type": "relative_path",
            "format": "image/png",
            "relative_path": "arkitscenes__a1d03f7f-cabf-482e-8fd0-67f2dfd7464f_00.png",
            "width": 256,
            "height": 192
          }
        },
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "Here are the detailed camera parameters for the image. Camera intrinsic parameters: Horizontal fov, hfov=62.26, and vertical fov, vfov=48.74. Image width=256 and height=192. ... Output a json list where each entry contains the object name in \"label\" and its 3D bounding box in \"bbox_3d\". Find the 3D bounding box that encapsulates the table in this visual representation."
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "[{'bbox_3d': [-0.85, 0.32, 2.12, 0.56, 0.82, 0.73, -2.75, 0.1, -2.4], 'label': 'table'}]"
          }
        }
      ]
    }
  ]
}
```

## 4) category 字段与提取策略

- 源数据字段层面没有稳定显式 `category`；
- 脚本先尝试读取显式字段（`category/sub_category/ability/task/label` 及 `meta_info` 同名键）；
- 若无显式字段，则使用第 2 节 F 的问句匹配策略推断 category；
- 最终 `metadata.json` 输出 `category_distribution`（样本量与占比）。

## 5) 脚本使用

脚本：`transform/JoyAI-Image-OpenSpatial/convert_to_pangu_ml.py`

```bash
python transform/JoyAI-Image-OpenSpatial/convert_to_pangu_ml.py \
  --parquet-dir /path/to/parquet_dir \
  --output-root /path/to/output_root \
  --max-samples 2000
```

参数：

- `--parquet-dir`：输入 parquet 文件夹（必填）
- `--output-root`：输出根目录（必填）
- `--max-samples`：小批量验证条数上限（可选）
- `--shard-size`：每分片样本数，默认 `8192`
- `--batch-size`：读取 parquet 的 batch 大小，默认 `256`

输出：

- `output_root/images/data_XXXXXX.tar`
- `output_root/jsonl/data_XXXXXX.jsonl`
- `output_root/metadata.json`

`metadata.json` 会包含：

- 数据集名
- `dataset_total_samples`：本次跑完整个 parquet 迭代后的总行数（与主循环统计一致）；若因 `--max-samples` 提前结束则为 `null`，并带 `stopped_by_max_samples: true`
- 实际扫描量/成功转换量/跳过量
- 各 `data_source` 的样本量与占比
- 是否检测到 category、各 category 的样本量与占比（若存在）
