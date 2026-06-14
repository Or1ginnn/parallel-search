import re

import requests
import torch
import transformers


question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retriever_url = "http://127.0.0.1:8000/retrieve"
topk = 3
max_turns = 3
max_queries_per_turn = 3
max_new_tokens = 1024
temperature = 0.7
do_sample = True
timeout = 60

# Set this to True to test only parser/formatter without loading model or retriever.
dry_run_parser = False

question = question.strip()
if question[-1] != "?":
    question += "?"

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = "\n\n{output_text}{information}\n\n"

# Prepare the message
prompt = f"""You are a search-augmented reasoning agent. \
You can only use the following tags: <think>, <plan>, <search>, <information>, <answer>. \
You must conduct reasoning inside <think> and </think> before every plan, search, or answer. \
Use <plan> to decompose the question into searchable sub-questions. \
If you lack knowledge, call a search engine by <search> query </search>. \
You can put multiple independent queries in one search action with "||", for example <search> query1 || query2 </search>. \
Each search action can contain at most {max_queries_per_turn} queries. \
The search engine will return results between <information> and </information>. \
Do not generate <information> yourself. \
If the evidence is sufficient, provide the answer inside <answer> and </answer>, without detailed illustrations. \
Question: {question}\n"""


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [
            torch.as_tensor(target_id, device=input_ids.device)
            for target_id in self.target_ids
        ]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                return True

        return False


def get_queries(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return []

    queries = []
    seen = set()
    for item in matches[-1].split("||"):
        query = " ".join(item.strip().split())
        if not query:
            continue
        key = query.lower()
        if key in seen:
            print(f'[parser warning] duplicate query dropped: "{query}"')
            continue
        seen.add(key)
        queries.append(query)

    if len(queries) > max_queries_per_turn:
        print(
            f"[parser warning] query count {len(queries)} exceeds max "
            f"{max_queries_per_turn}; truncating"
        )
        queries = queries[:max_queries_per_turn]

    return queries


def has_answer(text):
    return re.search(r"<answer>.*?</answer>", text, re.DOTALL) is not None


def truncate_at_first_action(text):
    candidates = []
    for tag in ("search", "answer"):
        end = text.find(f"</{tag}>")
        if end >= 0:
            candidates.append(end + len(f"</{tag}>"))
    if not candidates:
        return text
    return text[: min(candidates)]


def search(queries):
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": True,
    }
    results = requests.post(retriever_url, json=payload, timeout=timeout).json()["result"]

    def _passages2string(query, retrieval_result):
        format_reference = f"[Query] {query}\n"
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    information = "<information>"
    for query, retrieval_result in zip(queries, results):
        information += "\n" + _passages2string(query, retrieval_result)
    information += "</information>"
    return information


def run_dry_parser_demo():
    sample = (
        "<think>Break the question into related facts.</think>\n"
        "<plan>Search the person, the contract, and the later role.</plan>\n"
        "<search>Mike Barnett contracts || CSKA Moscow general manager player || "
        "Mike Barnett hockey agent client</search>"
    )
    queries = get_queries(sample)
    fake_results = [
        [
            {"document": {"contents": f"Title {i}\nEvidence snippet for {query}."}}
            for i in range(1, 3)
        ]
        for query in queries
    ]

    print("sample:")
    print(sample)
    print("\nqueries:")
    print(queries)
    print("\nformatted information:")

    information = "<information>"
    for query, retrieval_result in zip(queries, fake_results):
        information += "\n"
        information += f"[Query] {query}\n"
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            information += f"Doc {idx + 1}(Title: {title}) {text}\n"
    information += "</information>"
    print(information)


if dry_run_parser:
    run_dry_parser_demo()
    raise SystemExit

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# Initialize the stopping criteria
target_sequences = [
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
    "</answer>",
    " </answer>",
    "</answer>\n",
    " </answer>\n",
    "</answer>\n\n",
    " </answer>\n\n",
]
stopping_criteria = transformers.StoppingCriteriaList(
    [StopOnSequence(target_sequences, tokenizer)]
)

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

cnt = 0
total_queries = 0

print("\n\n################# [Start LiteCoA Reasoning + Searching] ##################\n\n")
print(prompt)

# Encode the chat-formatted prompt and move it to the correct device
while cnt < max_turns:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "stopping_criteria": stopping_criteria,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    # Generate text with the stopping criteria
    outputs = model.generate(**generate_kwargs)
    generated_tokens = outputs[0][input_ids.shape[1] :]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    output_text = truncate_at_first_action(output_text)

    if has_answer(output_text) or outputs[0][-1].item() in curr_eos:
        print(output_text)
        break

    tmp_queries = get_queries(output_text)
    if tmp_queries:
        information = search(tmp_queries)
        total_queries += len(tmp_queries)
    else:
        information = ""
        print("[agent warning] no valid query found; stopping")

    search_text = curr_search_template.format(
        output_text=output_text,
        information=information,
    )
    prompt += search_text
    cnt += 1
    print(search_text)

    if not tmp_queries:
        break

print(f"\nsearch_turns={cnt}, total_queries={total_queries}")
