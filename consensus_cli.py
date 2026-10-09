"""
Command-line front end for the consensus engine (the desktop app is consensus_app.py).

    python consensus_cli.py --prompt "Example - compare options" --rounds 2 --judge claude
    python consensus_cli.py --prompt-file my_prompt.txt --input AMOUNT=10000 --participants claude,codex,gemini
    python consensus_cli.py --prompt "Example - compare options" --mock --participants claude,codex,gemini --judge claude
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

from dotenv import load_dotenv

import consensus_core as core

load_dotenv(os.path.join(core.APP_DIR, ".env"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ask_wildcards(prompt_text: str, overrides: Dict[str, str]) -> Dict[str, str]:
    wildcards = core.find_wildcards(prompt_text)
    unknown = set(overrides) - {w.name for w in wildcards}
    if unknown:
        print(f"Warning: --input name(s) not found in the prompt: {', '.join(sorted(unknown))}")
    values: Dict[str, str] = {}
    pending = [w for w in wildcards if w.name not in overrides]
    if pending:
        print(f"The prompt has {len(pending)} value(s) to fill in.\n")
    interactive = sys.stdin.isatty()
    for w in wildcards:
        if w.name in overrides:
            values[w.name] = overrides[w.name]
            continue
        if not interactive:
            if w.default:
                values[w.name] = w.default
                continue
            sys.exit(f"No value for {w.name} and no terminal to ask on. Pass --input {w.name}=...")
        ctx = core.context_line(prompt_text, w.name)
        if ctx:
            print(f"  ...{ctx[:110]}")
        if w.choices:
            print("  choices: " + " | ".join(w.choices))
        label = f"{w.name} [{w.default}]" if w.default else w.name
        while True:
            try:
                answer = input(f"  {label}: ").strip() or w.default
            except EOFError:
                if w.default:
                    answer = w.default
                else:
                    sys.exit(f"\nNo value for {w.name} and no input available. Pass --input {w.name}=...")
            if answer:
                values[w.name] = answer
                print()
                break
            print("  A value is required.")
    return values


def print_table(rows: List[List[str]], headers: List[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def print_round(rr: core.RoundResult, labels: Dict[str, str]) -> None:
    keys = list(rr.answers)
    n = len(keys)
    title = "independent answers" if rr.index == 0 else "after revision"
    print("\n" + "=" * 78)
    print(f"ROUND {rr.index} ({title}): {core.round_summary(rr)}")
    print("=" * 78)
    attr = rr.attributes[0] if rr.attributes else ""
    rows = []
    for r in rr.matrix:
        cells = [r.key, r.label[:28]]
        mark = "!= " if (attr and r.attr_agreement(attr) is False) else ""
        cells += [(mark + (r.text(k, attr) or "held") if attr else "held") if r.held[k] else "" for k in keys]
        rows.append(cells + [{"all": "ALL", "majority": f"{r.count}/{n} consensus", "minority": f"{r.count}/{n} argued"}[r.status()]])
    if attr:
        sums = [core._num(core.column_sum(rr.matrix, k, attr)) for k in keys]
        if any(sums):
            rows.append(["SUM", ""] + sums + [""])
    head = [labels[k] + (f" ({attr})" if attr else "") for k in keys]
    print_table(rows, ["Key", "Label"] + head + ["Status"])
    if attr:
        print("(!= marks an attribute whose values differ between the models)")
    if len(rr.attributes) > 1:
        print(f"(other attributes in the reports and CSV: {', '.join(rr.attributes[1:])})")
    if rr.index > 0:
        for k in keys:
            print(f"\n{labels[k]} changes: {' '.join(rr.answers[k].changes.split())[:600]}")


def print_judge(jr: core.JudgeResult) -> None:
    print("\n" + "=" * 78)
    print(f"JUDGE RULING by {jr.label}")
    print("=" * 78)
    rows = [[it.key, it.label[:28], "; ".join(f"{a.name}: {a.value}" for a in it.attributes)[:40], it.rationale[:60]]
            for it in jr.answer.items]
    print_table(rows, ["Key", "Label", "Attributes", "Rationale"])
    print("\nRulings: " + " ".join(jr.answer.changes.split())[:1500])


def parse_kv(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--input expects NAME=value, got '{s}'")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-LLM consensus (CLI).")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="Name of a saved prompt in the prompts/ folder")
    src.add_argument("--prompt-file", help="Path to a prompt text file")
    src.add_argument("--ask", help="A plain question: --rounds revisions, a judge, format planned automatically")
    parser.add_argument("--no-plan", action="store_true", help="Send the prompt as written, without planning the format")
    parser.add_argument("--pause", action="store_true",
                        help="Pause after each round to type a note for the models, skip to the judge, or stop")
    parser.add_argument("--merge-synonyms", action="store_true",
                        help="After each round, merge differently-named items that mean the same thing (one extra short call)")
    parser.add_argument("--input", action="append", type=parse_kv, default=[], metavar="NAME=value",
                        help="Fill a {{NAME}} wildcard without being asked")
    parser.add_argument("--participants", default="claude,codex",
                        help="Comma-separated subset of claude,codex,gemini (default claude,codex)")
    parser.add_argument("--via", action="append", type=parse_kv, default=[], metavar="PARTICIPANT=cli|api",
                        help="Route per participant: cli = vendor tool on your subscription (default for claude and "
                             "codex), api = vendor API with a key from .env (gemini is always api)")
    parser.add_argument("--mock", action="store_true", help="Canned answers instead of real models, for trying the tool")
    parser.add_argument("--no-search", action="store_true",
                        help="Models answer from memory without web search: much faster, fine for opinion or advice questions")
    parser.add_argument("--model", action="append", type=parse_kv, default=[], metavar="PARTICIPANT=model",
                        help="Model override, e.g. --model claude=opus --model codex=gpt-5.5")
    parser.add_argument("--rounds", type=int, default=2, help="Revision rounds after the independent answers (default 2)")
    parser.add_argument("--no-stop-early", action="store_true", help="Run every round even after full agreement")
    parser.add_argument("--judge", choices=["none", "claude", "codex", "gemini"], default="none",
                        help="Participant that issues the final ruling")
    parser.add_argument("--out-dir", help="Output folder (default: results/<timestamp>/)")
    parser.add_argument("--show-reports", action="store_true", help="Print the full final reports too")
    args = parser.parse_args()

    if args.ask:
        raw = args.ask
        args.no_plan = False
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            raw = f.read()
    else:
        names = core.list_prompts()
        if not names:
            sys.exit("No saved prompts. Save one in the app or pass --prompt-file.")
        name = args.prompt
        if not name:
            print("Saved prompts:")
            for i, n in enumerate(names, 1):
                print(f"  {i}. {n}")
            try:
                name = names[int(input("Pick a number: ").strip()) - 1]
            except (ValueError, IndexError):
                sys.exit("Invalid choice.")
        if name not in names:
            sys.exit(f"No saved prompt named '{name}'. Available: {', '.join(names)}")
        raw = core.load_prompt(name)

    values = ask_wildcards(raw, dict(args.input))
    prompt = core.with_date_header(core.fill_wildcards(raw, values))

    chosen = [p.strip() for p in args.participants.split(",") if p.strip()]
    bad = [p for p in chosen if p not in core.PARTICIPANT_KEYS]
    if bad:
        sys.exit(f"Unknown participant(s): {', '.join(bad)}")
    cfg = core.ParticipantConfig(enabled={k: k in chosen for k in core.PARTICIPANT_KEYS}, mock=args.mock,
                                 web_search=not args.no_search)
    for k, m in args.model:
        cfg.models[k] = m
    for k, v in args.via:
        if v not in core.VIA_OPTIONS.get(k, []):
            sys.exit(f"--via {k}={v} is not possible; {k} supports: {', '.join(core.VIA_OPTIONS.get(k, []))}")
        cfg.via[k] = v
    participants = core.build_participants(cfg)
    labels = {p.key: p.label for p in participants}
    enabled = [k for k in core.PARTICIPANT_KEYS if k in chosen]   # participants are built in this order
    judge_key = None
    if args.ask and args.judge == "none":
        args.judge = enabled[0]
    if args.judge != "none":
        if args.judge not in enabled:
            sys.exit("The judge must be one of the participants.")
        judge_key = participants[enabled.index(args.judge)].key
    quick = core.make_planner(cfg) if (not args.no_plan or args.merge_synonyms) else None
    planner = None if args.no_plan else quick

    def gate(rr: core.RoundResult, next_step: str) -> core.GateDecision:
        print(f"\nPaused before {next_step}. Type a note for the models and press Enter, or press Enter for none.")
        extra = "  'judge' = skip to the judge," if (judge_key and next_step != "judge") else ""
        print(f"  Commands:{extra}  'stop' = stop here.")
        try:
            text = input("  > ").strip()
        except EOFError:
            return core.GateDecision("continue")
        if text.lower() == "stop":
            return core.GateDecision("stop")
        if text.lower() == "judge" and judge_key:
            return core.GateDecision("judge")
        return core.GateDecision("continue", text)

    procs = core.ProcessRegistry()
    try:
        result = core.run_consensus(prompt, values, participants, rounds=args.rounds, stop_early=not args.no_stop_early,
                                    judge_key=judge_key, planner=planner, merge_synonyms=args.merge_synonyms,
                                    merger=quick, out_dir=args.out_dir, procs=procs,
                                    log=lambda m: print(f"  {m}"),
                                    on_round=lambda rr: print_round(rr, labels),
                                    on_judge=print_judge,
                                    gate=gate if args.pause else None)
    except KeyboardInterrupt:
        procs.cancel()                      # kill the model tools instead of waiting for them
        sys.exit("\nInterrupted. Running model calls were stopped; finished rounds are kept in the results folder.")
    if result.rounds and args.show_reports:
        for k, ans in result.final.answers.items():
            print("\n" + "#" * 78 + f"\n{labels[k]} FINAL REPORT\n" + "#" * 78)
            print(core.answer_as_text(ans))
        if result.judge:
            print("\n" + "#" * 78 + "\nJUDGE REPORT\n" + "#" * 78)
            print(core.answer_as_text(result.judge.answer))
    print(f"\nEverything saved under {result.out_dir}")
    print("  consolidated.md, final_table.csv, run.json, round<N>_<participant>.md, judge.md")


if __name__ == "__main__":
    main()
