# Phase 1：LiteCoA Infer 原型

## 1. 阶段目标

Phase 1 的目标是先在推理侧验证 LiteCoA 的交互链路，不修改 veRL、不修改 GRPO、不修改 `search_r1/llm_agent/generation.py`。

本阶段新增：

```text
infer_litecoa.py
```

该脚本用于验证以下流程：

```text
模型生成 <think>/<plan>/<search>
  -> 解析 <search> 中的多个 query
  -> batch 调用 retriever
  -> 格式化 <information>
  -> 拼回 prompt
  -> 模型继续生成 <answer> 或下一轮 <search>
```

## 2. 当前实现范围

`infer_litecoa.py` 已实现：

```text
1. LiteCoA prompt
2. <search>q1 || q2 || q3</search> 解析
3. 每轮最多 max_queries_per_turn 个 query
4. 空 query 过滤
5. 重复 query 过滤
6. 超过 query 数量上限时截断并给出 warning
7. retriever batch 请求
8. 多 query information formatter
9. 遇到 </search> 或 </answer> 停止本轮生成
10. 多轮 agent loop
11. 与 `infer.py` 风格一致的文件顶部脚本配置
12. dry-run parser 测试模式
```

本阶段没有修改：

```text
1. veRL 训练代码
2. GRPO rollout
3. search_r1/llm_agent/generation.py
4. reward
5. SFT 数据构造脚本
```

## 3. Parser Dry Run

不加载模型、不连接 retriever，只测试 parser 和 formatter：

```python
dry_run_parser = True
```

预期行为：

```text
<search>q1 || q2 || q3</search>
```

会被解析为：

```text
["q1", "q2", "q3"]
```

并格式化成：

```text
<information>
[Query] q1
Doc 1(Title: ...): ...

[Query] q2
Doc 1(Title: ...): ...
</information>
```

## 4. 真实推理运行方式

先启动本地 retriever，默认接口为：

```text
http://127.0.0.1:8000/retrieve
```

然后运行：

```bash
python infer_litecoa.py
```

运行前在 `infer_litecoa.py` 顶部修改配置。该脚本刻意保持和原始 `infer.py`
相近的 demo 风格：顶部配置、`StopOnSequence`、`get_queries`、`search`、
主 `while` loop 都在同一个文件中，方便后续和 Search-R1 原始推理逻辑对照。

| 配置项 | 作用 |
|---|---|
| `model_id` | Hugging Face 模型路径或本地模型路径 |
| `question` | 输入问题 |
| `retriever_url` | 检索服务接口 |
| `topk` | 每个 query 返回的文档数 |
| `max_turns` | 最多 search 轮数 |
| `max_queries_per_turn` | 每轮最多 query 数 |
| `max_new_tokens` | 每轮最大生成 token 数 |
| `temperature` | 采样温度 |
| `do_sample` | 是否采样；设为 `False` 时使用 greedy decoding |
| `dry_run_parser` | 只测试 parser 和 formatter |

## 5. Phase 1 验收结论

Phase 1 现在已经完成 LiteCoA 推理侧的最小闭环实现：

```text
prompt -> generation -> search parser -> batch retrieval ->
information formatter -> continued generation
```

下一阶段可以基于这个脚本复用 parser / formatter / retriever client 逻辑，进入 Phase 2：批量构造 LiteCoA SFT 数据。
