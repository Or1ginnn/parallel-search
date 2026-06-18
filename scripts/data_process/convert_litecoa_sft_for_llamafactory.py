import argparse
import json
import re
from pathlib import Path


INFER_LITECOA_PROMPT = """You are a search-augmented reasoning agent. \
You can only use the following tags: <think>, <plan>, <search>, <information>, <answer>. \
You must conduct reasoning inside <think> and </think> before every plan, search, or answer. \
Use <plan> to decompose the question into searchable sub-questions. \
If you lack knowledge, call a search engine by <search> query </search>. \
You can put multiple independent queries in one search action with "||", for example <search> query1 || query2 </search>. \
Each search action can contain at most {max_queries_per_turn} queries. \
The search engine will return results between <information> and </information>. \
Do not generate <information> yourself. \
If the evidence is sufficient, provide the answer inside <answer> and </answer>, without detailed illustrations. \
Question: {question}
"""

TAG_RE = re.compile(r"<(think|plan|search|information|answer)>.*?</\1>", re.DOTALL)


def ensure_question_mark(question):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def extract_question(record, user_msg):
    question = record.get("question")
    if question:
        return ensure_question_mark(question)

    content = user_msg["content"].strip()
    if content.startswith("Question:"):
        content = content[len("Question:") :].strip()
    return ensure_question_mark(content)


def split_trajectory(assistant_content):
    parts = TAG_RE.findall(assistant_content)
    spans = list(TAG_RE.finditer(assistant_content))
    if not spans:
        raise ValueError("assistant trajectory has no LiteCoA tags")

    messages = []
    current_assistant = []
    last_end = 0

    for match in spans:
        gap = assistant_content[last_end : match.start()]
        if gap.strip():
            raise ValueError(f"unexpected text outside tags: {gap[:80]!r}")

        tag = match.group(1)
        text = match.group(0)
        if tag == "information":
            if not current_assistant:
                raise ValueError("information appears before assistant action")
            messages.append(
                {"role": "assistant", "content": "\n".join(current_assistant).strip()}
            )
            current_assistant = []
            messages.append({"role": "user", "content": text.strip()})
        else:
            current_assistant.append(text.strip())

        last_end = match.end()

    tail = assistant_content[last_end:]
    if tail.strip():
        raise ValueError(f"unexpected trailing text outside tags: {tail[:80]!r}")
    if current_assistant:
        messages.append(
            {"role": "assistant", "content": "\n".join(current_assistant).strip()}
        )

    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("converted trajectory must end with assistant message")
    return messages


def convert_messages(record, max_queries_per_turn):
    messages = record["messages"]
    if len(messages) != 3:
        raise ValueError(f"expected 3 messages, got {len(messages)}")

    system_msg, user_msg, assistant_msg = messages
    if system_msg.get("role") != "system":
        raise ValueError("first message must be system")
    if user_msg.get("role") != "user":
        raise ValueError("second message must be user")
    if assistant_msg.get("role") != "assistant":
        raise ValueError("third message must be assistant")

    question = extract_question(record, user_msg)
    user_content = INFER_LITECOA_PROMPT.format(
        max_queries_per_turn=max_queries_per_turn,
        question=question,
    )
    return [{"role": "user", "content": user_content}] + split_trajectory(
        assistant_msg["content"]
    )


def convert_record(record, keep_metadata, max_queries_per_turn):
    converted = {"messages": convert_messages(record, max_queries_per_turn)}
    if keep_metadata:
        for key, value in record.items():
            if key != "messages":
                converted[key] = value
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_llamafactory.jsonl",
    )
    parser.add_argument("--keep_metadata", action="store_true")
    parser.add_argument("--max_queries_per_turn", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                converted = convert_record(
                    record,
                    keep_metadata=args.keep_metadata,
                    max_queries_per_turn=args.max_queries_per_turn,
                )
            except Exception as exc:
                raise ValueError(f"failed to convert line {line_no}: {exc}") from exc
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"converted {count} records")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
