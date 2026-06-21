# Phase 4 LiteCoA Rollout Modification

## 1. 目标

Phase 4 第一刀改 Search-R1 的 agent rollout loop，让它支持 LiteCoA 的多 query search；同时补齐 LiteCoA prompt 数据处理和 answer reward 解析，使 GRPO 输入分布对齐 Phase 3 SFT。

本阶段不改：

```text
GRPO loss
advantage estimator
KL loss
VeRL 原生 rollout worker
tensor_helper
```

本阶段主要改：

```text
search_r1/llm_agent/generation.py
scripts/data_process/*search*.py
verl/utils/reward_score/qa_em.py
```

这个文件里的 `LLMGenerationManager` 是 Search-R1 插到 VeRL 训练流程里的 agent loop。VeRL 的 `RayPPOTrainer` 负责调用它，但不负责理解 `<search>` / `<information>` 的语义。

## 2. 原 Search-R1 rollout 逻辑

原逻辑是单 query search：

```text
model output
-> postprocess_predictions()
-> extract <search>...</search>
-> 把整个 search content 当成一个 query
-> batch_search([query])
-> 拼成 <information>retrieved docs</information>
-> 继续生成
```

如果模型输出：

```text
<search>q1 || q2 || q3</search>
```

旧代码会把它当成一个完整 query：

```text
"q1 || q2 || q3"
```

这不符合 LiteCoA。

## 3. 新 LiteCoA rollout 逻辑

现在每条样本的一次 `<search>` 可以包含多个 query：

```text
<search>q1 || q2 || q3</search>
```

解析规则：

```text
1. 用 "||" 分割 query。
2. 去掉首尾空格并压缩内部空白。
3. 丢弃空 query。
4. 丢弃包含 "<" 或 ">" 的 query。
5. 同一轮内 query 去重。
6. 每轮最多保留 max_queries_per_turn 个 query，默认 3。
```

batch 内检索方式：

```text
sample A: [q1, q2]
sample B: [q3]
sample C: [q4, q5, q6]

flat_queries = [q1, q2, q3, q4, q5, q6]
retriever.batch_search(flat_queries)
再按每条样本 query 数切回去
```

拼回 observation 的格式：

```text
<information>
[Query] q1
Doc 1(Title: ...)
Doc 2(Title: ...)

[Query] q2
Doc 1(Title: ...)
Doc 2(Title: ...)
</information>
```

该格式和 Phase 2 SFT 数据、Phase 3.5 vLLM eval 输出保持一致。

## 4. 修改文件

### `search_r1/llm_agent/generation.py`

新增配置字段：

```text
GenerationConfig.max_queries_per_turn = 3
```

新增方法：

```text
parse_search_queries()
_search_results2information()
```

修改方法：

```text
execute_predictions()
batch_search()
```

行为变化：

```text
1. `<search>single query</search>` 仍然兼容。
2. `<search>q1 || q2</search>` 会拆成多个 query。
3. batch_search() 现在返回 raw retrieval results，而不是提前拼成纯文本。
4. execute_predictions() 负责把每条样本的 query 和 retrieval result 拼成完整 `<information>` block。
```

### `verl/trainer/config/ppo_trainer.yaml`

新增：

```yaml
retriever:
  max_queries_per_turn: 3
```

### `verl/trainer/ppo/ray_trainer.py`

训练和验证创建 `GenerationConfig` 时透传：

```text
max_queries_per_turn = self.config.retriever.get('max_queries_per_turn', 3)
```

### `scripts/data_process/nq_search.py`

新增：

```text
template_type = litecoa
```

LiteCoA prompt 使用 Phase 3 SFT 最终训练时的单 user prompt，不使用 system role，也不加入 `<answer> Beijing </answer>` 示例。

### `scripts/data_process/qa_search_train_merge.py`

同样新增 `template_type = litecoa`，用于多 QA 数据集训练 parquet 构造。

### `scripts/data_process/qa_search_test_merge.py`

同样新增 `template_type = litecoa`，用于多 QA 数据集验证 parquet 构造。

### `verl/utils/reward_score/qa_em.py`

原 Search-R1 reward 解析依赖 prompt 里的 `<answer> Beijing </answer>` 示例，因此只有 `<answer>` 数量大于 1 时才取最后一个答案。LiteCoA prompt 为了对齐 SFT，不再放这个示例，所以现在改成：

```text
没有 <answer> -> None
有一个或多个 <answer> -> 取最后一个
```

这样既支持 LiteCoA 的单 answer 输出，也兼容原 Search-R1 prompt 中带示例 answer 的情况。

## 5. 不变部分

`run_llm_loop()` 的整体结构不变：

```text
for step in range(max_turns):
    generate
    parse action
    search if needed
    append information

final rollout:
    generate once more
    do_search=False
```

因此它天然支持：

```text
最后一轮 search 后，模型仍有一次基于 information 生成 answer 的机会。
```

`tensor_helper.py` 不需要改，因为它只处理：

```text
padding
attention_mask
position_ids
info_mask
sequence truncation
```

它不关心 query 格式。

## 6. 本地验证

本地做了不依赖真实 vLLM / retriever 的轻量逻辑验证：

```text
输入 1:
<search>q1 || q2 || q2 || q3 || q4</search>

结果:
q1, q2, q3 被保留
重复 q2 被去掉
q4 因 max_queries_per_turn=3 被截断
```

输出 observation：

```text
<information>
[Query] q1
Doc 1(Title: Title q1) Body for q1

[Query] q2
Doc 1(Title: Title q2) Body for q2

[Query] q3
Doc 1(Title: Title q3) Body for q3
</information>
```

同时验证：

```text
<search>single query</search>
```

仍然能正常走单 query 检索。

语法检查：

```bash
python -m py_compile search_r1/llm_agent/generation.py verl/trainer/ppo/ray_trainer.py
```

## 7. 下一步

Rollout 多 query 支持完成后，Phase 4 后续还需要：

```text
1. 改 LiteCoA prompt / data_process，生成 LiteCoA GRPO train/test parquet。
2. 确认 reward 的 <answer> 抽取仍兼容 LiteCoA prompt。
3. 新增 LiteCoA GRPO smoke 启动脚本。
4. 先跑 val_only smoke，再跑极小步数 GRPO。
```
