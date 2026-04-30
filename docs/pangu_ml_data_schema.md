# Pangu ML 训练数据格式规范

本文档是 **HW_pangu_ml** 仓库内「任意数据源 → Pangu ML 格式」转换的 **唯一规范依据**，供实现脚本的人类与自动化 Agent 共同遵循。实现代码放在 `transform/<数据源名称>/` 下；跨数据源的全局说明放在本目录 `docs/`。

---

## 1. 仓库布局（与本文档的关系）

| 路径 | 用途 |
|------|------|
| `docs/` | 全局规范与说明（含本文档）。 |
| `transform/<数据源>/` | 某一种来源数据的转换脚本与辅助配置；**一种数据源一个子目录**，子目录名建议与来源一致（如 `coco`、`internal_qa`）。 |

Agent 在新增或修改转换逻辑时：**先满足本文档的 MUST 规则**，再在对应 `transform/<数据源>/` 中实现。

---

## 2. 磁盘目录与分片命名（MUST）

```
{root}/
├── images/
│   └── data_{shard_id:06d}.tar    # 该分片内所有图像，按 tar 内相对路径存储
└── jsonl/
    └── data_{shard_id:06d}.jsonl  # 与该分片 tar 使用相同 shard_id，一一对应
```

| 规则 | 说明 |
|------|------|
| **分片 id 对齐** | 同一 `shard_id` 的 `data_XXXXXX.jsonl` 与 `data_XXXXXX.tar` 属于同一分片；JSONL 中每条样本里图像部件的 `relative_path` **必须**能在对应 tar 内解析到同一文件。 |
| **图像路径** | JSONL 里 `image.relative_path` 与 tar **内**相对路径一致（不含 `images/` 或 tar 文件名前缀）。 |
| **tar 内组织** | 图像以文件形式打入 tar，路径由转换脚本约定，但须与 JSONL 中引用一致。 |

示例：`data_000000.jsonl` ↔ `data_000000.tar`。

---

## 3. JSONL 单行：顶层结构（MUST）

每行一个 JSON 对象，表示一条训练样本（可含多轮对话）。

| 字段 | 类型 / 约定 | 说明 |
|------|----------------|------|
| `meta_prompt` | 数组 | 保留为 `[""]`（仅一个元素，且为空字符串）。不得省略或改为其他内容。 |
| `data` | 数组 | 对话轮次，结构见第 4 节。 |
| `id` | 字符串 | 样本唯一标识；**仓库内命名规则由业务约定**（Agent 实现时须在 `transform/<数据源>/` 或 PR 说明中写清生成规则，保证稳定、可去重）。 |

---

## 4. `data` 数组：角色、顺序与部件（MUST）

### 4.1 角色与顺序

- **必须**严格交替：`user` → `assistant` → `user` → `assistant` → …
- **禁止**插入 `system` 角色或其它角色。
- **禁止**在 `content` 中使用本文档未列出的部件类型。

### 4.2 单条样本内的图像：只出现在最前一轮 user，且图在文前

1. **整段 `data` 范围（轮次）**  
   - `type: "image"` **只能**出现在 **`data` 数组里第一条** `role: "user"` 的 `content` 中。  
   - **第二条及以后**的任意 `user`、以及 **全部** `assistant` 的 `content` 中，**不得**再出现 `type: "image"`。  
   - 即：视觉输入只在样本最前一轮 user 集中给出；多轮追问只动文本，不重复插图。

2. **仅针对上述第一条 `user` 的 `content`（顺序）**  
   - 若该 `content` 里 **同时** 含有 `type: "image"` 与 `type: "text"`，则 **所有** `image` 须排在 **所有** `text` 之前（不允许先 `text` 后 `image`）。  
   - 该条里是否带图、带几张图、带几段 `text`，**根据实际情况进行处理**。

### 4.3 文本与图像部件

- `type: "text"` 的 `string` 为 UTF-8 文本；语义上可为 question、answer、说明等，**由数据源约定**。
- `type: "image"` 的 `relative_path` 等字段见第 5 节。

### 4.4 多轮对话

在 `data` 末尾按顺序追加 `(user, assistant)` 对。除 **4.2 第 1 点**（仅首条 `user` 可有图、后续轮次无图）与 **4.2 第 2 点**（首条 `user` 内图在文前）外，各轮 `text` 条数等由数据源约定。

---

## 5. `content` 部件 JSON 形状（MUST）

### 5.1 文本部件

```json
{
  "type": "text",
  "text": {
    "type": "string",
    "format": "utf-8",
    "string": "<question 或 answer 的纯文本>"
  }
}
```

### 5.2 图像部件

```json
{
  "type": "image",
  "image": {
    "type": "relative_path",
    "format": "image/jpeg",
    "relative_path": "<对应分片 tar 包内的相对路径>",
    "width": <整数>,
    "height": <整数>
  }
}
```

- `width` / `height` **必须**与实际图像像素一致（或与训练管线约定一致；若管线有固定 resize，须在数据源 README 中说明）。
- `format` 对应该图像编码；常见为 `image/jpeg`。

---

## 6. 完整示例

以下仅为 **常见** 写法；**多图 / 多段 text 仅允许写在第一条 `user` 里**，且仍须满足「图在文前」。后续轮次示例中不出现 `image`。

### 6.1 单轮（一组内一条 QA）

```json
{
  "meta_prompt": [""],
  "data": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": {
            "type": "relative_path",
            "format": "image/jpeg",
            "relative_path": "example/subdir/frame_001.jpg",
            "width": 640,
            "height": 426
          }
        },
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "<question>"
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
            "string": "<answer>"
          }
        }
      ]
    }
  ],
  "id": "<由转换脚本生成的样本 id>"
}
```

### 6.2 多轮

继续在 `data` 中追加 `user` / `assistant`。**第二条及以后的 `user` 与所有 `assistant` 的 `content` 中不得含 `image`**，与 **4.2** 一致。

### 6.3 首条 `user` 内多图、多段文本（示意）

**仅**适用于 `data` 里 **第一条** `user`。若同时含 `image` 与 `text`，则 **先写所有 image，再写所有 text**；张数、段数由数据源约定。示例：

```json
"content": [
  { "type": "image", "image": { "type": "relative_path", "format": "image/jpeg", "relative_path": "a/1.jpg", "width": 640, "height": 426 } },
  { "type": "image", "image": { "type": "relative_path", "format": "image/jpeg", "relative_path": "a/2.jpg", "width": 640, "height": 426 } },
  { "type": "text", "text": { "type": "string", "format": "utf-8", "string": "<text 片段>" } }
]
```

---

## 7. 分片与样本数约束（MUST）

- 每个数据分片（每个 `data_XXXXXX.jsonl` 及其配对的 tar）内，**样本条数尽量为 16 的倍数**（例如 8192），最末尾分片可例外。

---

## 8. Agent 自检清单（实现完成后逐项核对）

1. [ ] 目录结构为 `root/images/data_XXXXXX.tar` 与 `root/jsonl/data_XXXXXX.jsonl`，且 id 一致。
2. [ ] 每行 JSON 含 `meta_prompt`、`data`、`id`；`meta_prompt` 恰好为 `[""]`。
3. [ ] `data` 中角色严格 `user`/`assistant` 交替，无 `system`。
4. [ ] 整条样本内 **仅第一条** `user` 的 `content` 可含 `image`；后续任意 `user` / `assistant` 的 `content` 中 **无** `image`。
5. [ ] 第一条 `user` 若同时含 `image` 与 `text`，则所有 `image` 下标均小于任一 `text` 下标。
6. [ ] 所有 `relative_path` 在对应 tar 中存在且可读。
7. [ ] 每个 jsonl 分片内样本数为 16 的倍数。
8. [ ] 代码与 `id` 生成规则已写在 `transform/<数据源>/` 的 README 或注释中。

---

## 9. 修订说明

规范变更应更新本文档版本信息（可在 Git 提交信息中说明）；各 `transform/<数据源>/` 子项目应在 README 中注明「兼容本文档提交 `<commit 或日期>`」。
