"""Render saved real responses; optionally capture the report with Chromium."""
import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(report):
    esc = lambda value: html.escape(str(value))
    cards = []
    for i, result in enumerate(report['results'], 1):
        rag = result['rag']
        sources = ''.join(f'<a href="{esc(p["url"])}">[{p["citation"]}] {esc(p["title"])}</a><p>{esc(p["text"].split(chr(10),1)[-1])}</p>' for p in rag['sources'])
        mode = 'Verified archive records (extractive fallback)' if rag['answer_mode'] == 'verified_records' else 'Model-generated answer with retrieved context'
        cards.append(f'''<section class="example"><div class="question"><span>0{i}</span><h2>{esc(result['question'])}</h2></div>
<div class="columns"><article class="baseline"><div class="label">MODEL ONLY <span>No documents</span></div><div class="answer">{esc(result['model_only']['answer'])}</div><div class="timing">{result['model_only']['seconds']}s · no retrievable source citations</div></article>
<article class="rag"><div class="label">WITH RAG <span>Official NBA evidence</span></div><div class="answer">{esc(rag['answer'])}</div><div class="timing">{rag['seconds']}s · {esc(mode)}</div><div class="evidence">{sources}</div></article></div></section>''')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CourtSide — real local model comparison</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f7ef;color:#252a20;font:16px/1.6 Arial,sans-serif}main{max-width:1280px;margin:auto;padding:48px 48px 32px}.brand{font-weight:800;font-size:24px;letter-spacing:-1px}.brand b{background:#d6ee83;border-radius:50%;padding:5px 10px;margin-right:8px}.eyebrow{font-size:11px;letter-spacing:2px;color:#727d60;margin-top:35px}h1{font-size:48px;line-height:1.1;letter-spacing:-2px;margin:16px 0}header p{color:#6d7761;max-width:860px}.chips{display:flex;gap:10px;margin:24px 0 32px}.chips span{font-size:11px;border:1px solid #d8dfcc;border-radius:20px;padding:5px 12px;background:#eff2e6}.example{border:1px solid #dce1d1;border-radius:12px;background:white;overflow:hidden;margin:22px 0}.question{display:flex;gap:16px;align-items:center;padding:18px 24px;border-bottom:1px solid #e4e8dc}.question>span{font:13px monospace;color:#7b8867}h2{font-size:20px;line-height:1.4;margin:0}.columns{display:grid;grid-template-columns:1fr 1fr}article{padding:24px}.baseline{background:#fbf7f1;border-right:1px solid #e2e5da}.rag{background:#f3f7e9}.label{font-size:11px;letter-spacing:1.4px;font-weight:bold;color:#946c35;display:flex;justify-content:space-between;align-items:center}.label span{font-size:10px;font-weight:normal;letter-spacing:0;border:1px solid #e3d6c1;border-radius:20px;padding:3px 9px}.rag .label{color:#526e26}.rag .label span{border-color:#cddcb5}.answer{white-space:pre-wrap;overflow-wrap:anywhere;margin:18px 0;line-height:1.75;font-size:16px}.timing{color:#8a8f7e;font-size:11px}.evidence{border-top:1px solid #dce5ce;margin-top:18px;padding-top:14px}.evidence a{font-size:12px;color:#526e26}.evidence p{font-size:12px;color:#68765a;margin:4px 0 12px}footer{color:#78816c;font-size:12px;margin-top:26px;line-height:1.8}footer strong{color:#4d5a3f}@media(max-width:700px){main{padding:24px 18px}.columns{grid-template-columns:1fr}.baseline{border-right:0;border-bottom:1px solid #e2e5da}h1{font-size:36px}.chips{flex-wrap:wrap}}
</style></head><body><main><header><div class="brand"><b>◉</b>CourtSide.</div><div class="eyebrow">LOCAL MODELS · REAL RESPONSES · TRACEABLE EVIDENCE</div><h1>Same model. Same questions.<br>Now give it the records.</h1><p>Exact responses from Qwen2.5-0.5B-Instruct on CPU. The baseline uses model memory; the RAG app retrieves official NBA history before answering.</p><div class="chips"><span>No paid API</span><span>Deterministic decoding</span><span>Saved responses, not a mockup</span></div></header>''' + ''.join(cards) + f'''<footer><strong>How to read this:</strong> This is one recorded run on three selected questions, not an accuracy benchmark. Both paths use the same checkpoint and decoding settings. The RAG application also includes season/category filtering and a verified-record fallback, labeled above when used.<br>Recorded: {esc(report['recorded_at'])} · Model: {esc(report['model'])}<br>Settings, baseline prompts, timings and source records are saved in comparison.json. These screenshots are browser captures of this saved-response report.</footer></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    report = json.loads((ROOT / 'docs/comparison.json').read_text())
    page_path = ROOT / 'docs/comparison.html'
    page_path.write_text(render(report))
    if args.capture:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1)
            page.goto(page_path.as_uri())
            page.screenshot(path=str(ROOT / 'docs/images/rag-comparison.png'), full_page=True)
            for i, card in enumerate(page.locator('.example').all(), 1):
                card.screenshot(path=str(ROOT / f'docs/images/example-{i}.png'))
            browser.close()
    print('Rendered', page_path)


if __name__ == '__main__':
    main()
