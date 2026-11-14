"""Record real, deterministic local responses with and without the RAG pipeline."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from nba_rag.config import DATA, LLM_ID, ROOT
from nba_rag.engine import NBAEngine
import torch

QUESTIONS = [
    "Who won the regular-season MVP in 2024-25?",
    "Who won Rookie of the Year in 2024-25?",
    "Who won the 2023 NBA championship and Finals MVP?",
]


@torch.inference_mode()
def model_only(engine, question):
    messages = [
        {"role": "system", "content": "You are an NBA history assistant. Answer concisely and factually. If you do not know, say you do not know. Distinguish regular-season MVP from Finals MVP."},
        {"role": "user", "content": question},
    ]
    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    started = time.monotonic()
    output = engine.model.generate(**inputs, max_new_tokens=240, do_sample=False,
                                   repetition_penalty=1.08, pad_token_id=engine.tokenizer.eos_token_id)
    answer = engine.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return {"answer": answer, "seconds": round(time.monotonic() - started, 1),
            "messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'comparison.json')
    args = parser.parse_args()
    engine = NBAEngine()
    engine.load()
    results = []
    for question in QUESTIONS:
        baseline = model_only(engine, question)
        rag = engine.ask(question)
        result = {"question": question, "model_only": baseline, "rag": rag}
        results.append(result)
        print(json.dumps({"question": question, "model_only": baseline['answer'],
                          "rag": rag['answer'], "rag_mode": rag['answer_mode']}, ensure_ascii=False), flush=True)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "model": LLM_ID,
              "platform": platform.platform(), "torch": torch.__version__,
              "settings": {"do_sample": False, "max_new_tokens": 240, "repetition_penalty": 1.08, "device": "cpu"},
              "corpus_sha256": hashlib.sha256((DATA / 'documents.json').read_bytes()).hexdigest(),
              "method": "Same checkpoint, questions and decoding settings. Baseline receives no documents. RAG uses the production retrieval, prompt, abstention and verified-record fallback; it is an application comparison, not an isolated retrieval benchmark.",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved real responses to {args.output}', flush=True)


if __name__ == '__main__':
    main()
