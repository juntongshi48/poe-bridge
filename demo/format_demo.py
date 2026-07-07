"""Phase 2: render the JSON traces from poe_bridge_demo.py into a plain-text
decoding illustration matching the layout in DEMO.md."""

import argparse
import json
import os

from transformers import AutoTokenizer


LABEL_WIDTH = 18  # column width for "DLM draft:", "RS toward PoE:", etc.

IGNORED_TOKENS = [
    'Ġ',
    'ĠĠ',
    'Ċ',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces_dir", type=str, default="demo_traces")
    p.add_argument("--naive_file", type=str, default="naive_speculative.json")
    p.add_argument("--poe_file", type=str, default="poe_bridge.json")
    p.add_argument("--output_file", type=str, default="demo_traces/formatted.txt")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Limit number of steps shown per method (None = all).")
    p.add_argument("--width", type=int, default=120,
                   help="Soft wrap width for long token rows.")
    return p.parse_args()


# ---------- token rendering ----------

def decode_one(tok_id, tokenizer):
    if tok_id == tokenizer.eos_token_id:
        return "<EOS>"
    tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
    s = tokenizer.convert_tokens_to_string([tok])
    s = s.strip()
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    if not s:
        # fallback for whitespace-only / unprintable tokens
        s = repr(tok)
    return s


def box(tok_id, tokenizer, marker=""):
    return f"[{decode_one(tok_id, tokenizer)}{marker}]"


def correction_box(rej_id, corr_id, tokenizer):
    return f"[{decode_one(rej_id, tokenizer)}✗ → {decode_one(corr_id, tokenizer)}★]"


def is_ignored(tok_id, tokenizer):
    """True if the raw BPE form of the token is in IGNORED_TOKENS (whitespace-only filler)."""
    tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
    return tok in IGNORED_TOKENS


# ---------- row builders (return a list of box-strings) ----------

def draft_boxes(cand, tokenizer, tail_after_reject=2):
    """Show accepted prefix + rejected token + up to `tail_after_reject` more, then '...'.
    If all tokens were accepted, show the full draft. Skips IGNORED_TOKENS."""
    draft = cand["dlm_draft_tokens"]
    orig_n = cand["orig_num_accept"]
    if orig_n >= len(draft):
        return [box(t, tokenizer) for t in draft if not is_ignored(t, tokenizer)]
    end = min(orig_n + 1 + tail_after_reject, len(draft))
    parts = [box(t, tokenizer) for t in draft[:end] if not is_ignored(t, tokenizer)]
    if end < len(draft):
        parts.append("...")
    return parts


def rs_boxes(cand, tokenizer):
    draft = cand["dlm_draft_tokens"]
    accepted = cand["accepted_tokens"]
    orig_n = cand["orig_num_accept"]
    was_corr = cand["was_corrected"]

    parts = []
    for i in range(min(orig_n, len(draft))):
        if not is_ignored(draft[i], tokenizer):
            parts.append(box(draft[i], tokenizer, marker="✓"))
    if orig_n < len(draft):
        rej_tok = draft[orig_n]
        if was_corr and orig_n < len(accepted):
            # Always render the correction box, even if either side is in IGNORED_TOKENS,
            # since the correction step itself is meaningful.
            parts.append(correction_box(rej_tok, accepted[orig_n], tokenizer))
        elif not is_ignored(rej_tok, tokenizer):
            parts.append(box(rej_tok, tokenizer, marker="✗"))
        if orig_n + 1 < len(draft):
            parts.append("[discarded...]")
    return parts


def append_boxes(cand, tokenizer):
    return [box(t, tokenizer) for t in cand["accepted_tokens"] if not is_ignored(t, tokenizer)]


# ---------- line assembly with soft wrap ----------

def wrap_boxes(prefix, boxes, width):
    """Join boxes with spaces under `prefix` (left-padded to LABEL_WIDTH),
    soft-wrapping to `width` columns; continuation lines are indented."""
    indent = " " * LABEL_WIDTH
    label = prefix.ljust(LABEL_WIDTH)
    if not boxes:
        return [label]
    lines, cur = [], label
    for b in boxes:
        candidate = cur + (" " if cur not in (label, indent) else "") + b
        if len(candidate) > width and cur not in (label, indent):
            lines.append(cur)
            cur = indent + b
        else:
            cur = candidate
    lines.append(cur)
    return lines


def k_row(k, boxes, width):
    return wrap_boxes(f"k={k+1}", boxes, width)


# ---------- step renderers ----------

def render_step_naive(step, tokenizer, width):
    cand = step["candidates"][0]
    out = [f"Step {step['step'] + 1}"]
    out += wrap_boxes("DLM draft:", draft_boxes(cand, tokenizer), width)
    out += wrap_boxes("RS toward AR:", rs_boxes(cand, tokenizer), width)
    out += wrap_boxes("Append:", append_boxes(cand, tokenizer), width)
    return "\n".join(out)


def render_step_poe(step, tokenizer, width):
    cands = step["candidates"]
    sel_k = step["selected_k"]

    out = [f"Step {step['step'] + 1}"]

    out.append("Parallel DLM drafts:")
    for c in cands:
        out += k_row(c["k"], draft_boxes(c, tokenizer), width)
    out.append("")

    out.append("RS toward PoE:")
    for c in cands:
        out += k_row(c["k"], rs_boxes(c, tokenizer), width)
    out.append("")

    out.append("IS toward AR:")
    is_parts = []
    for c in cands:
        tag = " selected" if c["selected"] else ""
        is_parts.append(f"k={c['k']+1}: {c['is_weight']:.2f}{tag}")
    out.append(" " * LABEL_WIDTH + "    ".join(is_parts))
    out.append("")

    sel_cand = cands[sel_k]
    out += wrap_boxes("Append:", append_boxes(sel_cand, tokenizer), width)
    return "\n".join(out)


# ---------- top-level rendering ----------

def header_box(text, width):
    inner = max(width - 2, len(text) + 2)
    top = "┌" + "─" * inner + "┐"
    mid = "│ " + text.ljust(inner - 1) + "│"
    bot = "└" + "─" * inner + "┘"
    return "\n".join([top, mid, bot])


def render_run(result, tokenizer, *, section_letter, section_title, render_step_fn, args):
    out = []
    out.append(f"({section_letter}) {section_title}")
    out.append("─" * args.width)
    steps = result["trace"]
    if args.max_steps is not None:
        steps = steps[:args.max_steps]
    for i, step in enumerate(steps):
        if len(step["candidates"][0]["dlm_draft_tokens"]) == 0:
            # The final step might corresponds to the generation of EOS, so skip it.
            break
        out.append(render_step_fn(step, tokenizer, args.width))
        out.append("")
    if args.max_steps is not None and len(result["trace"]) > args.max_steps:
        out.append(f"[... {len(result['trace']) - args.max_steps} more steps omitted ...]\n")
    final = result["final_text"]
    out.append(f"Final output:     \"{final}\"")
    out.append("")
    out.append(f"Stats: {result['num_tokens_generated']} tokens in "
               f"{result['num_forward_evals']} forward passes "
               f"({result['elapsed_sec']:.2f}s)")
    return "\n".join(out)


def main():
    args = parse_args()

    naive_path = os.path.join(args.traces_dir, args.naive_file)
    poe_path = os.path.join(args.traces_dir, args.poe_file)
    with open(naive_path) as f:
        naive = json.load(f)
    with open(poe_path) as f:
        poe = json.load(f)

    # Both runs share the prompt + tokenizer; load tokenizer from one of them.
    tokenizer = AutoTokenizer.from_pretrained(naive["tokenizer_name"], trust_remote_code=True)

    sections = []
    sections.append(header_box(f"Prompt: \"{naive['prompt_text']}\"", args.width))
    sections.append("")
    sections.append(render_run(
        naive, tokenizer,
        section_letter="A",
        section_title="Direct DLM → AR correction (naive speculative, K=1, mw=0.0)",
        render_step_fn=render_step_naive, args=args,
    ))
    sections.append("")
    sections.append(render_run(
        poe, tokenizer,
        section_letter="B",
        section_title=f"PoE-Bridge correction (K={poe['config']['n_parallel_samples']}, "
                      f"mw={poe['config']['mixture_weight']})",
        render_step_fn=render_step_poe, args=args,
    ))

    output = "\n".join(sections) + "\n"

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(output)
    print(output)
    print(f"\n[wrote {args.output_file}]")


if __name__ == "__main__":
    main()
