"""
Consensus - desktop app.

Write or pick a prompt, fill in its {{wildcards}}, choose participants, revision rounds
and a judge, and watch the models argue toward an answer. Results, reports and past
runs are browsable inside the app; the program instructions given to the models are
editable in the Instructions tab.

Launch:  python consensus_app.py   (or double-click "Consensus.pyw")
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, Optional

from dotenv import load_dotenv

import consensus_core as core

load_dotenv(os.path.join(core.APP_DIR, ".env"))

SETTINGS_PATH = os.path.join(core.APP_DIR, "app_settings.json")
PARTICIPANT_LABELS = {"claude": "Claude", "codex": "OpenAI", "gemini": "Gemini"}

WILDCARD_HELP = """Wildcards let one saved prompt ask you for values every time you run it.

{{AMOUNT}}
    An input box. You must fill it in before running.

{{HORIZON|10 years}}
    An input box pre-filled with "10 years". Press Run to accept it or type over it.

{{RISK|low|medium|high}}
    A dropdown with three choices. "low" is selected by default. You can also type
    something that is not in the list.

Rules:
  - Names are letters, digits, underscores and spaces. They are case-sensitive.
  - Use the same name twice and it is asked once and filled in everywhere.
  - Values you enter are remembered for the next run of any prompt that uses the same name.
  - Today's date is always added to the top of the prompt automatically.
"""


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def num(x: Optional[float]) -> str:
    return core._num(x)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(core.APP_NAME)
        self.minsize(1100, 740)
        self.settings = load_settings()
        self.geometry(self.settings.get("geometry", "1300x860"))

        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.procs: Optional[core.ProcessRegistry] = None
        self.run_result: Optional[core.RunResult] = None    # what the Results tab is showing
        self.live_result: Optional[core.RunResult] = None   # the run in progress (or the last one finished)
        self.current_prompt_name: Optional[str] = None
        self.input_vars: Dict[str, tk.StringVar] = {}
        self._wildcard_job: Optional[str] = None
        self.reports_dir = ""
        self.instructions = core.load_instructions()

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=24)
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_run = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_reports = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)
        self.tab_instructions = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)
        for tab, text in ((self.tab_run, "Run"), (self.tab_results, "Results"), (self.tab_reports, "Reports"),
                          (self.tab_history, "History"), (self.tab_instructions, "Instructions"),
                          (self.tab_settings, "Settings"), (self.tab_help, "Help")):
            self.notebook.add(tab, text=f"  {text}  ")
        self._build_help_tab()

        self._build_run_tab()
        self._build_results_tab()
        self._build_reports_tab()
        self._build_history_tab()
        self._build_instructions_tab()
        self._build_settings_tab()
        self.apply_model_config_to_run_tab()

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", padx=8, pady=(0, 6))

        self.refresh_prompt_list()
        last = self.settings.get("last_prompt")
        if last and last in core.list_prompts():
            self.open_prompt(last)
        self.refresh_history()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll_queue)

    # ------------------------------------------------------------------ Run tab
    def _build_run_tab(self) -> None:
        ask = ttk.LabelFrame(self.tab_run, text="Ask a question", padding=6)
        ask.pack(fill="x", padx=6, pady=(6, 2))
        self.ask_entry = ttk.Entry(ask, font=("Segoe UI", 11))
        self.ask_entry.pack(side="left", fill="x", expand=True)
        self.ask_entry.insert(0, self.settings.get("last_question", ""))
        self.ask_entry.bind("<Return>", lambda e: self.ask_question())
        self.ask_btn = ttk.Button(ask, text="Ask", style="Run.TButton", command=self.ask_question)
        self.ask_btn.pack(side="left", padx=(8, 0))
        ttk.Label(ask, text="Plain question, no wildcards. Uses the participants and revision rounds set below, "
                            "always with a judge, and plans the answer format for you. For elaborate prompts use "
                            "the library.",
                  foreground="#555", wraplength=520, justify="left").pack(side="left", padx=(10, 0))

        paned = ttk.PanedWindow(self.tab_run, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=0)
        ttk.Label(left, text="Saved prompts", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.prompt_list = tk.Listbox(left, width=28, exportselection=False, activestyle="none")
        self.prompt_list.pack(fill="both", expand=True, pady=4)
        self.prompt_list.bind("<<ListboxSelect>>", self.on_prompt_selected)
        btns = ttk.Frame(left)
        btns.pack(fill="x")
        for text, cmd in (("New", self.new_prompt), ("Save", self.save_current_prompt),
                          ("Save as", self.save_prompt_as), ("Delete", self.delete_current_prompt)):
            ttk.Button(btns, text=text, command=cmd, width=8).pack(side="left", padx=1, pady=2)

        right = ttk.Frame(paned, padding=6)
        paned.add(right, weight=1)

        head = ttk.Frame(right)
        head.pack(fill="x")
        self.prompt_title = tk.StringVar(value="Untitled prompt")
        ttk.Label(head, textvariable=self.prompt_title, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Wildcard guide", command=self.show_wildcard_help).pack(side="right")
        legend = ttk.Frame(right)
        legend.pack(fill="x", pady=(2, 2))
        ttk.Label(legend, text="Legend:", foreground="#555").pack(side="left")
        for code, meaning in (("{{NAME}}", "ask for a value"), ("{{NAME|default}}", "ask, pre-filled"),
                              ("{{NAME|a|b|c}}", "dropdown of choices")):
            ttk.Label(legend, text=code, font=("Consolas", 9), foreground="#1a5fb4").pack(side="left", padx=(8, 2))
            ttk.Label(legend, text=meaning, foreground="#555").pack(side="left")

        editor_frame = ttk.Frame(right)
        editor_frame.pack(fill="both", expand=True, pady=(2, 6))
        self.editor = tk.Text(editor_frame, wrap="word", undo=True, font=("Consolas", 10), height=12)
        sb = ttk.Scrollbar(editor_frame, command=self.editor.yview)
        self.editor.configure(yscrollcommand=sb.set)
        self.editor.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.editor.bind("<<Modified>>", self.on_editor_modified)

        self.inputs_frame = ttk.LabelFrame(right, text="Inputs found in the prompt", padding=6)
        self.inputs_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(self.inputs_frame, text="No {{wildcards}} in this prompt.", foreground="#777").pack(anchor="w")

        s = self.settings
        settings = ttk.LabelFrame(right, text="Run settings", padding=6)
        settings.pack(fill="x", pady=(0, 6))

        ttk.Label(settings, text="Participants").grid(row=0, column=0, sticky="nw")
        pf = ttk.Frame(settings)
        pf.grid(row=0, column=1, columnspan=3, sticky="w")
        self.use_vars: Dict[str, tk.BooleanVar] = {}
        self.via_vars: Dict[str, tk.StringVar] = {}
        self.via_combos: Dict[str, ttk.Combobox] = {}
        self.use_checks: Dict[str, ttk.Checkbutton] = {}
        self.route_hints: Dict[str, tk.StringVar] = {}
        defaults_on = {"claude": True, "codex": True, "gemini": False}
        ttk.Label(pf, text="via", foreground="#555").grid(row=0, column=1, sticky="w", padx=(10, 2))
        for i, key in enumerate(core.PARTICIPANT_KEYS, start=1):
            self.use_vars[key] = tk.BooleanVar(value=bool(s.get(f"use_{key}", defaults_on[key])))
            self.via_vars[key] = tk.StringVar(value=s.get(f"via_{key}", ""))
            cb = ttk.Checkbutton(pf, text=PARTICIPANT_LABELS[key], variable=self.use_vars[key], width=12,
                                 command=self.refresh_judge_choices)
            cb.grid(row=i, column=0, sticky="w", pady=1)
            self.use_checks[key] = cb
            combo = ttk.Combobox(pf, state="readonly", width=18)
            combo.bind("<<ComboboxSelected>>", lambda e, k=key: self.on_via_changed(k))
            combo.grid(row=i, column=1, sticky="w", padx=(10, 2), pady=1)
            self.via_combos[key] = combo
            hv = tk.StringVar(value="")
            self.route_hints[key] = hv
            ttk.Label(pf, textvariable=hv, foreground="#555").grid(row=i, column=2, sticky="w", padx=(10, 2))
        ttk.Label(pf, text="Routes and default models are set up once in the Settings tab.", foreground="#555").grid(
            row=1, column=3, sticky="w", padx=(14, 0))
        self.mock_var = tk.BooleanVar(value=bool(s.get("mock", False)))
        ttk.Checkbutton(pf, text="Use mock models instead (canned answers, no real calls, for trying the app)",
                        variable=self.mock_var).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

        ttk.Label(settings, text="Revision rounds").grid(row=2, column=0, sticky="w", pady=(6, 0))
        rf = ttk.Frame(settings)
        rf.grid(row=2, column=1, columnspan=3, sticky="w", pady=(6, 0))
        self.rounds_var = tk.IntVar(value=int(s.get("rounds", 2)))
        ttk.Spinbox(rf, from_=0, to=10, textvariable=self.rounds_var, width=5).pack(side="left")
        self.stop_early_var = tk.BooleanVar(value=bool(s.get("stop_early", True)))
        ttk.Checkbutton(rf, text="Stop early once everyone holds the same items",
                        variable=self.stop_early_var).pack(side="left", padx=(12, 0))
        ttk.Label(rf, text="   Round 0 is independent; each revision round every model argues the others' answers.",
                  foreground="#555").pack(side="left")

        ttk.Label(settings, text="Judge").grid(row=3, column=0, sticky="w", pady=(6, 0))
        jf = ttk.Frame(settings)
        jf.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))
        self.judge_var = tk.StringVar(value=s.get("judge", "none"))
        self.judge_combo = ttk.Combobox(jf, state="readonly", width=18)
        self.judge_combo.pack(side="left")
        ttk.Label(jf, text="   After the last round the judge reads every final answer and rules on each argued item.",
                  foreground="#555").pack(side="left")
        self.refresh_judge_choices()

        ttk.Label(settings, text="Format").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.plan_var = tk.BooleanVar(value=bool(s.get("plan", True)))
        ttk.Checkbutton(settings, text="Plan the answer format automatically before round 0 (what each item is, "
                                       "which attributes, which units)", variable=self.plan_var
                        ).grid(row=4, column=1, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(settings, text="Web search").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.search_var = tk.BooleanVar(value=bool(s.get("web_search", True)))
        ttk.Checkbutton(settings, text="Let the models search the web (needed for prices, news, anything current; "
                                       "turn off for opinion or advice questions to get answers in seconds)",
                        variable=self.search_var).grid(row=7, column=1, columnspan=3, sticky="w", pady=(6, 0))
        self.merge_var = tk.BooleanVar(value=bool(s.get("merge_synonyms", False)))
        ttk.Checkbutton(settings, text="Also merge differently-named items that mean the same thing after each "
                                       "round (one short extra call per round)", variable=self.merge_var
                        ).grid(row=5, column=1, columnspan=3, sticky="w")
        ttk.Label(settings, text="Pacing").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.pause_var = tk.BooleanVar(value=bool(s.get("pause", False)))
        ttk.Checkbutton(settings, text="Pause after each round so I can review the results and comment before "
                                       "the next step", variable=self.pause_var
                        ).grid(row=6, column=1, columnspan=3, sticky="w", pady=(6, 0))

        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=(0, 4))
        self.run_btn = ttk.Button(actions, text="Run", style="Run.TButton", command=self.start_run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(actions, text="Open results folder", command=self.open_results_folder).pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.pack(side="right")

        log_frame = ttk.LabelFrame(right, text="Progress", padding=4)
        log_frame.pack(fill="both")
        self.log = tk.Text(log_frame, height=5, state="disabled", font=("Consolas", 9), wrap="word")
        lsb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        self.log.pack(side="left", fill="both", expand=True)
        lsb.pack(side="right", fill="y")

        self.note_frame = ttk.LabelFrame(right, text="Your note for the next step", padding=4)
        self.note_frame.pack(fill="x", pady=(4, 0))
        self.note_box = tk.Text(self.note_frame, height=3, font=("Segoe UI", 10), wrap="word", undo=True)
        self.note_box.pack(fill="x")
        gate_row = ttk.Frame(self.note_frame)
        gate_row.pack(fill="x", pady=(4, 0))
        self.gate_continue_btn = ttk.Button(gate_row, text="Continue", command=lambda: self.resolve_gate("continue"),
                                            state="disabled")
        self.gate_continue_btn.pack(side="left")
        self.gate_judge_btn = ttk.Button(gate_row, text="Skip to judge", command=lambda: self.resolve_gate("judge"),
                                         state="disabled")
        self.gate_judge_btn.pack(side="left", padx=6)
        self.gate_stop_btn = ttk.Button(gate_row, text="Stop here", command=lambda: self.resolve_gate("stop"),
                                        state="disabled")
        self.gate_stop_btn.pack(side="left")
        self.gate_hint = tk.StringVar(value="Type a note while the models work; it is sent to every model with the "
                                            "next round. With pausing on, the run waits here after each round.")
        ttk.Label(gate_row, textvariable=self.gate_hint, foreground="#555", wraplength=620,
                  justify="left").pack(side="left", padx=(12, 0))
        self.gate_state: Optional[dict] = None
        self.gate_next_step = ""

    def refresh_judge_choices(self) -> None:
        names = ["none"] + [PARTICIPANT_LABELS[k] for k in core.PARTICIPANT_KEYS if self.use_vars[k].get()]
        keys = ["none"] + [k for k in core.PARTICIPANT_KEYS if self.use_vars[k].get()]
        self._judge_keys = keys
        self.judge_combo["values"] = names
        current = self.judge_var.get()
        self.judge_combo.current(keys.index(current) if current in keys else 0)
        self.judge_combo.bind("<<ComboboxSelected>>", lambda e: self.judge_var.set(self._judge_keys[self.judge_combo.current()]))
        self.judge_var.set(keys[self.judge_combo.current()])

    def show_wildcard_help(self) -> None:
        win = tk.Toplevel(self)
        win.title("Wildcard guide")
        win.geometry("640x420")
        t = tk.Text(win, wrap="word", font=("Consolas", 10), padx=10, pady=10)
        t.insert("1.0", WILDCARD_HELP)
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ------------------------------------------------------------- Results tab
    def _build_results_tab(self) -> None:
        top = ttk.Frame(self.tab_results, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="View:").pack(side="left")
        self.round_combo = ttk.Combobox(top, state="readonly", width=52)
        self.round_combo.pack(side="left", padx=6)
        self.round_combo.bind("<<ComboboxSelected>>", lambda e: self.show_view(self.round_combo.current()))
        ttk.Label(top, text="Attribute:").pack(side="left", padx=(10, 0))
        self.attr_combo = ttk.Combobox(top, state="readonly", width=16)
        self.attr_combo.pack(side="left", padx=6)
        self.attr_combo.bind("<<ComboboxSelected>>", lambda e: self.show_view(self.round_combo.current()))
        self.results_run = tk.StringVar(value="No run loaded")
        ttk.Label(top, textvariable=self.results_run, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        self.results_summary = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.results_summary).pack(side="left", padx=12)
        ttk.Button(top, text="Export CSV...", command=self.export_csv).pack(side="right")
        self.live_btn = ttk.Button(top, text="Show current run", command=self.show_live_run, state="disabled")
        self.live_btn.pack(side="right", padx=6)

        legend = ttk.Frame(self.tab_results, padding=(6, 0))
        legend.pack(fill="x")
        for color, text in (("#e8f5e9", "held by everyone"), ("#fff9c4", "majority (consensus)"),
                            ("#ffe0b2", "minority (argued)"), ("#e3f2fd", "judge included"), ("#eeeeee", "judge excluded")):
            tk.Label(legend, text="  ", bg=color, relief="solid", bd=1).pack(side="left", padx=(8, 2))
            ttk.Label(legend, text=text, foreground="#444").pack(side="left")
        ttk.Label(legend, text="   ≠ = the models give different values for this attribute "
                               "(free-text attributes are shown but not scored)", foreground="#444").pack(side="left")

        paned = ttk.PanedWindow(self.tab_results, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        tree_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=3)
        self.matrix = ttk.Treeview(tree_frame, show="headings", selectmode="browse")
        tsb = ttk.Scrollbar(tree_frame, command=self.matrix.yview)
        self.matrix.configure(yscrollcommand=tsb.set)
        self.matrix.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")
        for tag, color in (("all", "#e8f5e9"), ("majority", "#fff9c4"), ("minority", "#ffe0b2"),
                           ("included", "#e3f2fd"), ("excluded", "#eeeeee")):
            self.matrix.tag_configure(tag, background=color)
        self.matrix.bind("<<TreeviewSelect>>", self.on_matrix_select)

        notes_frame = ttk.LabelFrame(paned, text="Details", padding=4)
        paned.add(notes_frame, weight=2)
        self.notes = tk.Text(notes_frame, wrap="word", state="disabled", font=("Segoe UI", 10))
        nsb = ttk.Scrollbar(notes_frame, command=self.notes.yview)
        self.notes.configure(yscrollcommand=nsb.set)
        self.notes.pack(side="left", fill="both", expand=True)
        nsb.pack(side="right", fill="y")
        self.notes.tag_configure("h", font=("Segoe UI", 10, "bold"))

    # ------------------------------------------------------------- Reports tab
    def _build_reports_tab(self) -> None:
        top = ttk.Frame(self.tab_reports, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Report:").pack(side="left")
        self.report_combo = ttk.Combobox(top, state="readonly", width=60)
        self.report_combo.pack(side="left", padx=6)
        self.report_combo.bind("<<ComboboxSelected>>", lambda e: self.show_report())
        ttk.Button(top, text="Open in editor", command=self.open_report_external).pack(side="left")
        frame = ttk.Frame(self.tab_reports)
        frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.report_view = tk.Text(frame, wrap="word", state="disabled", font=("Segoe UI", 10))
        rsb = ttk.Scrollbar(frame, command=self.report_view.yview)
        self.report_view.configure(yscrollcommand=rsb.set)
        self.report_view.pack(side="left", fill="both", expand=True)
        rsb.pack(side="right", fill="y")

    # ------------------------------------------------------------- History tab
    def _build_history_tab(self) -> None:
        frame = ttk.Frame(self.tab_history, padding=6)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Past runs (newest first). Double-click one to browse its results and reports.").pack(anchor="w")
        self.history_list = tk.Listbox(frame, exportselection=False, activestyle="none", font=("Consolas", 10))
        self.history_list.pack(fill="both", expand=True, pady=4)
        self.history_list.bind("<Double-Button-1>", lambda e: self.load_history_run())
        btns = ttk.Frame(frame)
        btns.pack(fill="x")
        ttk.Button(btns, text="Load", command=self.load_history_run).pack(side="left")
        ttk.Button(btns, text="Open folder", command=self.open_history_folder).pack(side="left", padx=6)
        ttk.Button(btns, text="Refresh", command=self.refresh_history).pack(side="left")
        ttk.Button(btns, text="Delete run", command=self.delete_history_run).pack(side="right")

    # -------------------------------------------------------- Instructions tab
    def _build_instructions_tab(self) -> None:
        frame = ttk.Frame(self.tab_instructions, padding=6)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Program instructions sent to the models alongside your prompt. Edit and Save to "
                              "change how they behave; Reset restores the defaults.", foreground="#444").pack(anchor="w")
        self.instruction_boxes: Dict[str, tk.Text] = {}
        titles = {"plan": "Format planning: the answer template decided before round 0",
                  "base": "Every call: how to answer and format the result",
                  "revision": "Revision rounds: how to treat the other models' answers",
                  "judge": "Judge pass: how to rule on argued items",
                  "reconcile": "Name merging (optional): how to spot items that mean the same thing"}
        for key in ("plan", "base", "revision", "judge", "reconcile"):
            lf = ttk.LabelFrame(frame, text=titles[key], padding=4)
            lf.pack(fill="both", expand=True, pady=4)
            t = tk.Text(lf, wrap="word", height=4, font=("Segoe UI", 10), undo=True)
            t.pack(fill="both", expand=True)
            t.insert("1.0", self.instructions[key])
            self.instruction_boxes[key] = t
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Save instructions", command=self.save_instructions).pack(side="left")
        ttk.Button(btns, text="Reset to defaults", command=self.reset_instructions).pack(side="left", padx=6)

    # ---------------------------------------------------------------- Help tab
    def _build_help_tab(self) -> None:
        frame = ttk.Frame(self.tab_help, padding=8)
        frame.pack(fill="both", expand=True)
        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, text="Getting started: what to set up before the first run, and what to do when something "
                            "goes wrong.", foreground="#444").pack(side="left")
        ttk.Button(top, text="Open as a file", command=lambda: os.startfile(os.path.join(core.APP_DIR, "GETTING_STARTED.md"))
                   ).pack(side="right")
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 10), padx=12, pady=8)
        sb = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side="left", fill="both", expand=True, pady=(6, 0))
        sb.pack(side="right", fill="y", pady=(6, 0))
        text.tag_configure("h1", font=("Segoe UI", 14, "bold"), spacing3=6)
        text.tag_configure("h2", font=("Segoe UI", 11, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("code", font=("Consolas", 10), background="#f3f3f3")
        try:
            with open(os.path.join(core.APP_DIR, "GETTING_STARTED.md"), encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            lines = ["GETTING_STARTED.md is missing from the app folder."]
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                text.insert("end", "    " + line + "\n", "code")
            elif line.startswith("# "):
                text.insert("end", line[2:] + "\n", "h1")
            elif line.startswith("## "):
                text.insert("end", line[3:] + "\n", "h2")
            elif line.startswith("|"):
                cells = [c.strip().replace("`", "").replace("**", "") for c in line.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue
                text.insert("end", "    " + "   |   ".join(cells) + "\n", "code")
            else:
                # strip markdown emphasis and inline code markers for plain reading
                plain = line.replace("**", "").replace("`", "")
                text.insert("end", plain + "\n")
        text.configure(state="disabled")

    # ------------------------------------------------------------ Settings tab
    ROUTE_LABELS = {"cli": "CLI (subscription)", "api": "API (key)"}
    API_KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "codex": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    CLI_TOOL = {"claude": "claude", "codex": "codex"}

    def model_config(self, key: str) -> dict:
        """Per-model setup from app settings: which routes are enabled and the default model per route."""
        defaults = {"cli": key in core.CLI_TOOL_KEYS, "api": key == "gemini", "cli_model": "", "api_model": ""}
        saved = self.settings.get("model_config", {}).get(key, {})
        return {**defaults, **saved}

    def enabled_routes(self, key: str) -> list:
        mc = self.model_config(key)
        return [r for r in core.VIA_OPTIONS[key] if mc.get(r)]

    def _build_settings_tab(self) -> None:
        outer = ttk.Frame(self.tab_settings, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Set up each model once. Tick the routes it may run through, give each route a key or "
                              "default model if needed, and press Test. On the Run tab you then pick a route per "
                              "participant from the ones you enabled here.",
                  foreground="#444", wraplength=900, justify="left").pack(anchor="w", pady=(0, 6))

        self.cfg_vars: Dict[str, Dict[str, tk.Variable]] = {}
        self.status_vars: Dict[str, Dict[str, tk.StringVar]] = {}
        self.test_btns: Dict[str, Dict[str, ttk.Button]] = {}
        self.key_entries: Dict[str, ttk.Entry] = {}
        env = core.read_env_file()

        for key in core.PARTICIPANT_KEYS:
            mc = self.model_config(key)
            card = ttk.LabelFrame(outer, text=PARTICIPANT_LABELS[key], padding=8)
            card.pack(fill="x", pady=4)
            self.cfg_vars[key] = {}
            self.status_vars[key] = {}
            self.test_btns[key] = {}
            row = 0
            # CLI route
            if key in self.CLI_TOOL:
                v = tk.BooleanVar(value=bool(mc["cli"]))
                self.cfg_vars[key]["cli"] = v
                ttk.Checkbutton(card, text=self.ROUTE_LABELS["cli"], variable=v, width=20,
                                command=self.on_config_changed).grid(row=row, column=0, sticky="w")
                ttk.Label(card, text="model:").grid(row=row, column=1, sticky="e", padx=(8, 2))
                mv = tk.StringVar(value=mc.get("cli_model", ""))
                self.cfg_vars[key]["cli_model"] = mv
                ttk.Entry(card, textvariable=mv, width=22).grid(row=row, column=2, sticky="w")
                sv = tk.StringVar(value=core.participant_status(key, "cli"))
                self.status_vars[key]["cli"] = sv
                ttk.Label(card, textvariable=sv, width=48).grid(row=row, column=3, sticky="w", padx=(12, 0))
                b = ttk.Button(card, text="Test", width=8, command=lambda k=key: self.test_participant(k, "cli"))
                b.grid(row=row, column=4, padx=6)
                self.test_btns[key]["cli"] = b
                ttk.Label(card, text=f"Uses the {self.CLI_TOOL[key]} tool signed in to your subscription, with web "
                                     "search. Blank model = the tool's default.", foreground="#555").grid(
                    row=row + 1, column=0, columnspan=5, sticky="w", pady=(0, 6))
                row += 2
            else:
                ttk.Label(card, text="Gemini's command-line tool no longer accepts personal accounts, so Gemini runs "
                                     "through the API only.", foreground="#555").grid(
                    row=row, column=0, columnspan=5, sticky="w", pady=(0, 6))
                row += 1
            # API route
            v = tk.BooleanVar(value=bool(mc["api"]))
            self.cfg_vars[key]["api"] = v
            ttk.Checkbutton(card, text=self.ROUTE_LABELS["api"], variable=v, width=20,
                            command=self.on_config_changed).grid(row=row, column=0, sticky="w")
            ttk.Label(card, text="model:").grid(row=row, column=1, sticky="e", padx=(8, 2))
            mv = tk.StringVar(value=mc.get("api_model", ""))
            self.cfg_vars[key]["api_model"] = mv
            ttk.Entry(card, textvariable=mv, width=22).grid(row=row, column=2, sticky="w")
            sv = tk.StringVar(value=core.participant_status(key, "api"))
            self.status_vars[key]["api"] = sv
            ttk.Label(card, textvariable=sv, width=48).grid(row=row, column=3, sticky="w", padx=(12, 0))
            b = ttk.Button(card, text="Test", width=8, command=lambda k=key: self.test_participant(k, "api"))
            b.grid(row=row, column=4, padx=6)
            self.test_btns[key]["api"] = b
            ttk.Label(card, text="API key:").grid(row=row + 1, column=1, sticky="e", padx=(8, 2))
            kv = tk.StringVar(value=env.get(self.API_KEY_ENV[key], os.getenv(self.API_KEY_ENV[key], "")))
            self.cfg_vars[key]["api_key"] = kv
            ke = ttk.Entry(card, textvariable=kv, width=52, show="•")
            ke.grid(row=row + 1, column=2, columnspan=2, sticky="w")
            self.key_entries[key] = ke
            ttk.Button(card, text="Show", width=6,
                       command=lambda e=ke: e.configure(show="" if e.cget("show") else "•")).grid(row=row + 1, column=4, padx=6)
            hint = {"claude": "Anthropic API key (console.anthropic.com), billed per token. Blank model = "
                              f"{core.API_DEFAULT_MODELS['claude']}. No web search.",
                    "codex": "OpenAI API key (platform.openai.com), billed per token. Blank model = "
                             f"{core.API_DEFAULT_MODELS['codex']}. No web search.",
                    "gemini": "Gemini API key (aistudio.google.com, free tier works). Blank model = "
                              f"{core.API_DEFAULT_MODELS['gemini']}, with fallback to older Flash models. "
                              "No web search on the free tier."}[key]
            ttk.Label(card, text=hint, foreground="#555", wraplength=900, justify="left").grid(
                row=row + 2, column=0, columnspan=5, sticky="w")

        glob = ttk.LabelFrame(outer, text="Global", padding=8)
        glob.pack(fill="x", pady=4)
        ttk.Label(glob, text="Per-call timeout (seconds)").grid(row=0, column=0, sticky="w")
        self.timeout_var = tk.StringVar(value=env.get("CLI_TIMEOUT_SECONDS", os.getenv("CLI_TIMEOUT_SECONDS", "")))
        ttk.Entry(glob, textvariable=self.timeout_var, width=10).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(glob, text="How long to wait for one model call. Research with web search can take 20 minutes or "
                             "more. Blank = 2400.", foreground="#555").grid(row=0, column=2, sticky="w")

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Save settings", command=self.save_config).pack(side="left")
        ttk.Label(btns, text="Keys and the timeout go to the .env file next to the app; routes and default models to "
                             "app_settings.json. Takes effect immediately.", foreground="#555").pack(side="left", padx=10)

    def on_config_changed(self) -> None:
        """A route checkbox changed: keep the Run tab's choices in step (saved on Save)."""
        self.apply_model_config_to_run_tab()

    def save_config(self) -> None:
        t = self.timeout_var.get().strip()
        if t and (not t.isdigit() or int(t) < 30):
            messagebox.showwarning("Timeout", "The timeout must be a whole number of seconds, at least 30.")
            return
        updates = {self.API_KEY_ENV[k]: self.cfg_vars[k]["api_key"].get() for k in core.PARTICIPANT_KEYS}
        updates["CLI_TIMEOUT_SECONDS"] = t
        try:
            core.write_env_file(updates)
        except OSError as e:
            messagebox.showerror("Could not save", str(e))
            return
        mcfg = self.settings.setdefault("model_config", {})
        for k in core.PARTICIPANT_KEYS:
            vars_ = self.cfg_vars[k]
            mcfg[k] = {"cli": bool(vars_["cli"].get()) if "cli" in vars_ else False,
                       "api": bool(vars_["api"].get()),
                       "cli_model": vars_["cli_model"].get().strip() if "cli_model" in vars_ else "",
                       "api_model": vars_["api_model"].get().strip()}
        save_settings(self.settings)
        for k in core.PARTICIPANT_KEYS:
            for route, sv in self.status_vars[k].items():
                sv.set(core.participant_status(k, route))
        self.apply_model_config_to_run_tab()
        self.set_status("Settings saved")

    def test_participant(self, key: str, route: str) -> None:
        self.test_btns[key][route].configure(state="disabled")
        self.status_vars[key][route].set("Testing...")
        model = self.cfg_vars[key].get(f"{route}_model")
        model = model.get().strip() if model else ""
        # use the key as typed, even before Save
        env_name = self.API_KEY_ENV[key]
        typed_key = self.cfg_vars[key]["api_key"].get().strip()
        q = self.queue

        def worker() -> None:
            if route == "api" and typed_key:
                os.environ[env_name] = typed_key
            try:
                q.put(("test", (key, route, core.test_participant(key, model, route))))
            except Exception as e:  # noqa: BLE001
                q.put(("test", (key, route, f"FAILED: {str(e).strip().splitlines()[0][:150]}")))

        threading.Thread(target=worker, daemon=True).start()

    def save_instructions(self) -> None:
        self.instructions = {k: t.get("1.0", "end-1c").strip() or core.DEFAULT_INSTRUCTIONS[k]
                             for k, t in self.instruction_boxes.items()}
        core.save_instructions(self.instructions)
        self.set_status("Instructions saved")

    def reset_instructions(self) -> None:
        if not messagebox.askyesno("Reset", "Replace all three instruction texts with the defaults?"):
            return
        for k, t in self.instruction_boxes.items():
            t.delete("1.0", "end")
            t.insert("1.0", core.DEFAULT_INSTRUCTIONS[k])
        self.save_instructions()

    # ------------------------------------------------------- prompt library ops
    def refresh_prompt_list(self) -> None:
        self.prompt_list.delete(0, "end")
        names = core.list_prompts()
        for name in names:
            self.prompt_list.insert("end", name)
        if self.current_prompt_name in names:
            self.prompt_list.selection_set(names.index(self.current_prompt_name))

    def editor_text(self) -> str:
        return self.editor.get("1.0", "end-1c")

    def set_editor_text(self, text: str) -> None:
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self.editor.edit_modified(False)
        self.rebuild_inputs()

    def on_prompt_selected(self, _event=None) -> None:
        sel = self.prompt_list.curselection()
        if not sel:
            return
        name = self.prompt_list.get(sel[0])
        if name == self.current_prompt_name:
            return
        if not self.confirm_discard():
            self.refresh_prompt_list()
            return
        self.open_prompt(name)

    def open_prompt(self, name: str) -> None:
        try:
            text = core.load_prompt(name)
        except OSError as e:
            messagebox.showerror("Could not open prompt", str(e))
            return
        self.current_prompt_name = name
        self.prompt_title.set(name)
        self.set_editor_text(text)
        self.settings["last_prompt"] = name
        self.refresh_prompt_list()

    def confirm_discard(self) -> bool:
        """Before leaving an edited prompt: Yes = save it, No = drop the edits, Cancel = stay. Returns True to proceed."""
        if not self.editor.edit_modified():
            return True
        name = self.current_prompt_name or "this untitled prompt"
        choice = messagebox.askyesnocancel("Unsaved changes",
                                           f"Save your changes to {name}?\n\nYes = save, No = discard the changes, "
                                           "Cancel = stay here.")
        if choice is None:
            return False
        if choice:
            if self.current_prompt_name is None:
                self.save_prompt_as()
                return not self.editor.edit_modified()    # False if they cancelled the name dialog
            self.save_current_prompt()
        return True

    def new_prompt(self) -> None:
        if not self.confirm_discard():
            return
        self.current_prompt_name = None
        self.prompt_title.set("Untitled prompt")
        self.set_editor_text("")
        self.prompt_list.selection_clear(0, "end")

    def save_current_prompt(self) -> None:
        if self.current_prompt_name is None:
            self.save_prompt_as()
            return
        core.save_prompt(self.current_prompt_name, self.editor_text())
        self.editor.edit_modified(False)
        self.set_status(f"Saved prompt '{self.current_prompt_name}'")

    def save_prompt_as(self) -> None:
        name = simpledialog.askstring("Save prompt as", "Name for this prompt:",
                                      initialvalue=self.current_prompt_name or "", parent=self)
        if not name:
            return
        if name in core.list_prompts() and name != self.current_prompt_name:
            if not messagebox.askyesno("Overwrite?", f"A prompt named '{name}' exists. Overwrite it?"):
                return
        core.save_prompt(name, self.editor_text())
        self.current_prompt_name = name
        self.prompt_title.set(name)
        self.editor.edit_modified(False)
        self.settings["last_prompt"] = name
        self.refresh_prompt_list()
        self.set_status(f"Saved prompt '{name}'")

    def delete_current_prompt(self) -> None:
        sel = self.prompt_list.curselection()
        if not sel:
            return
        name = self.prompt_list.get(sel[0])
        if not messagebox.askyesno("Delete prompt", f"Delete the saved prompt '{name}'? This cannot be undone."):
            return
        core.delete_prompt(name)
        if name == self.current_prompt_name:
            self.current_prompt_name = None
            self.prompt_title.set("Untitled prompt (deleted from library)")
        self.refresh_prompt_list()

    # -------------------------------------------------------------- wildcards
    def on_editor_modified(self, _event=None) -> None:
        if not self.editor.edit_modified():
            return
        if self._wildcard_job:
            self.after_cancel(self._wildcard_job)
        self._wildcard_job = self.after(400, self.rebuild_inputs)
        self.editor.edit_modified(True)

    def rebuild_inputs(self) -> None:
        self._wildcard_job = None
        text = self.editor_text()
        wildcards = core.find_wildcards(text)
        remembered = self.settings.setdefault("inputs", {})
        for name, var in self.input_vars.items():
            if var.get():
                remembered[name] = var.get()
        for child in self.inputs_frame.winfo_children():
            child.destroy()
        self.input_vars = {}
        if not wildcards:
            ttk.Label(self.inputs_frame, text="No {{wildcards}} in this prompt.", foreground="#777").pack(anchor="w")
            return
        grid = ttk.Frame(self.inputs_frame)
        grid.pack(fill="x")
        for i, w in enumerate(wildcards):
            ttk.Label(grid, text=w.name, font=("Segoe UI", 9, "bold")).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=1)
            initial = remembered.get(w.name) if remembered.get(w.name) and (not w.choices or remembered[w.name] in w.choices) else w.default
            var = tk.StringVar(value=initial or "")
            self.input_vars[w.name] = var
            if w.choices:
                box = ttk.Combobox(grid, textvariable=var, values=w.choices, width=26)
                box.grid(row=i, column=1, sticky="w", pady=1)
            else:
                ttk.Entry(grid, textvariable=var, width=28).grid(row=i, column=1, sticky="w", pady=1)
            ctx = core.context_line(text, w.name)
            if len(ctx) > 110:
                ctx = ctx[:107] + "..."
            ttk.Label(grid, text=ctx, foreground="#555").grid(row=i, column=2, sticky="w", padx=10)

    # --------------------------------------------------------------- running
    def participant_config(self) -> core.ParticipantConfig:
        cfg = core.ParticipantConfig(mock=bool(self.mock_var.get()), web_search=bool(self.search_var.get()))
        for k in core.PARTICIPANT_KEYS:
            via = self.via_vars[k].get()
            cfg.enabled[k] = bool(self.use_vars[k].get()) and (bool(via) or cfg.mock)
            cfg.via[k] = via or core.VIA_OPTIONS[k][0]
            mc = self.model_config(k)
            cfg.models[k] = (mc.get(f"{via}_model") or "").strip() if via else ""
        return cfg

    def on_via_changed(self, key: str) -> None:
        routes = self.enabled_routes(key)
        idx = self.via_combos[key].current()
        self.via_vars[key].set(routes[idx] if 0 <= idx < len(routes) else "")

    def apply_model_config_to_run_tab(self) -> None:
        """Offer each participant only the routes enabled in Settings; disable it if there are none."""
        for key in core.PARTICIPANT_KEYS:
            routes = self.enabled_routes(key)
            combo = self.via_combos[key]
            combo["values"] = [self.ROUTE_LABELS[r] for r in routes]
            current = self.via_vars[key].get()
            if routes:
                if current not in routes:
                    current = routes[0]
                combo.current(routes.index(current))
                combo.configure(state="readonly")
                self.use_checks[key].configure(state="normal")
                self.route_hints[key].set("")
            else:
                combo.set("")
                combo.configure(state="disabled")
                self.use_vars[key].set(False)
                self.use_checks[key].configure(state="disabled")
                self.route_hints[key].set("not set up: enable a route in Settings")
                current = ""
            self.via_vars[key].set(current)
        self.refresh_judge_choices()

    def start_run(self) -> None:
        """Run the prompt in the editor with the settings as shown."""
        text = self.editor_text().strip()
        if not text:
            messagebox.showwarning("Empty prompt", "Write or pick a prompt first.")
            return
        values = {n: v.get().strip() for n, v in self.input_vars.items()}
        try:
            prompt = core.fill_wildcards(text, values)
        except ValueError as e:
            messagebox.showwarning("Missing input", str(e))
            return
        self.launch(prompt, values, rounds=max(0, self.rounds_value()), judge_ui=self.judge_var.get())

    def ask_question(self) -> None:
        """Run a one-off question with the rounds set below, a judge, and the format planned automatically."""
        question = self.ask_entry.get().strip()
        if not question:
            messagebox.showwarning("Empty question", "Type a question first.")
            return
        judge_ui = self.judge_var.get()
        if judge_ui == "none":
            judge_ui = next((k for k in core.PARTICIPANT_KEYS if self.use_vars[k].get()), "none")
        self.settings["last_question"] = question
        self.launch(question, {}, rounds=max(0, self.rounds_value()), judge_ui=judge_ui, force_plan=True)

    def launch(self, prompt: str, values: Dict[str, str], rounds: int, judge_ui: str, force_plan: bool = False) -> None:
        if self.procs is not None:
            messagebox.showinfo("Busy", "A run is already in progress. Stop it first.")
            return
        cfg = self.participant_config()
        try:
            participants = core.build_participants(cfg)
        except Exception as e:
            messagebox.showerror("Cannot start", str(e))
            return
        judge_key = None
        enabled_keys = [k for k in core.PARTICIPANT_KEYS if cfg.enabled.get(k)]
        if judge_ui != "none" and judge_ui in enabled_keys:
            # map the UI key (claude/codex/gemini) to the participant actually built
            judge_key = participants[enabled_keys.index(judge_ui)].key
        elif judge_ui != "none":
            self.append_log(f"Judge '{judge_ui}' is not an enabled participant; running without a judge.")
        planner = core.make_planner(cfg) if (force_plan or self.plan_var.get() or self.merge_var.get()) else None
        merge_synonyms = bool(self.merge_var.get())
        if merge_synonyms and not (force_plan or self.plan_var.get()):
            planner_for_run = None  # merging needs the planner participant but not the format step
        else:
            planner_for_run = planner

        self.persist_settings()
        self.clear_log()
        self.live_result = None
        self.live_btn.configure(state="disabled")
        self.procs = core.ProcessRegistry()
        self.run_btn.configure(state="disabled")
        self.ask_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(12)
        self.set_status("Running...")

        prompt = core.with_date_header(prompt)
        stop_early = self.stop_early_var.get()
        instructions = dict(self.instructions)
        procs, q = self.procs, self.queue
        labels = {p.key: p.label for p in participants}
        out_dir = os.path.join(core.RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.has_judge = judge_key is not None
        gate_state = {"event": threading.Event(), "decision": None}
        self.gate_state = gate_state

        def gate(rr: core.RoundResult, next_step: str) -> core.GateDecision:
            gate_state["event"].clear()
            gate_state["decision"] = None
            q.put(("gate", next_step))
            gate_state["event"].wait()
            return gate_state["decision"] or core.GateDecision("stop")

        def worker() -> None:
            try:
                q.put(("start", (labels, out_dir)))
                result = core.run_consensus(
                    prompt, values, participants, rounds=rounds, stop_early=stop_early, judge_key=judge_key,
                    planner=planner_for_run, merge_synonyms=merge_synonyms, merger=planner,
                    instructions=instructions, procs=procs, out_dir=out_dir,
                    log=lambda m: q.put(("log", m)),
                    on_round=lambda rr: q.put(("round", rr)),
                    on_judge=lambda jr: q.put(("judge", jr)),
                    gate=gate,
                )
                q.put(("done", result))
            except Exception as e:  # noqa: BLE001
                q.put(("error", f"{type(e).__name__}: {e}"))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    # --------------------------------------------------------- between rounds
    def on_gate(self, next_step: str) -> None:
        """The engine finished a round and is asking what to do before `next_step`."""
        self.gate_next_step = next_step
        if not self.pause_var.get():
            self.resolve_gate("continue")
            return
        label = "Continue to judge" if next_step == "judge" else f"Continue to {next_step}"
        self.gate_continue_btn.configure(text=label, state="normal")
        self.gate_judge_btn.configure(state="normal" if (self.has_judge and next_step != "judge") else "disabled")
        self.gate_stop_btn.configure(state="normal")
        self.note_frame.configure(text=f"Paused. Add a note for {next_step} (optional), then choose")
        self.append_log(f"Waiting for you. Review the results, add a note if you like, then press '{label}'.")
        self.set_status(f"Paused before {next_step}. See the note box on the Run tab.")
        self.bell()

    def resolve_gate(self, action: str) -> None:
        state = self.gate_state
        if not state or state["event"].is_set():
            return
        note = self.note_box.get("1.0", "end-1c").strip()
        self.note_box.delete("1.0", "end")
        if note and self.live_result is not None:
            key = "judge" if self.gate_next_step == "judge" else self.gate_next_step.split()[-1]
            self.live_result.notes[key] = note
        for b in (self.gate_continue_btn, self.gate_judge_btn, self.gate_stop_btn):
            b.configure(state="disabled")
        self.gate_continue_btn.configure(text="Continue")
        self.note_frame.configure(text="Your note for the next step")
        if action != "stop":
            self.set_status("Running...")
        state["decision"] = core.GateDecision(action, note)
        state["event"].set()

    def stop_run(self) -> None:
        if self.procs and messagebox.askyesno("Stop run", "Stop the current run? Finished rounds are kept."):
            self.append_log("Stopping...")
            if self.gate_state and not self.gate_state["event"].is_set():
                self.resolve_gate("stop")      # paused between rounds: nothing to kill, just release the engine
            else:
                self.procs.cancel()

    def poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "start":
                    labels, out_dir = payload
                    self.live_result = core.RunResult(out_dir=out_dir, prompt="", inputs={}, participants=labels)
                    self.run_result = self.live_result
                    self.populate_views(self.live_result)      # clears the previous run's tables
                    self.populate_reports(out_dir)             # only the prompt and plan exist yet
                    self.live_btn.configure(state="normal")
                elif kind == "round":
                    if self.live_result is None:
                        self.live_result = core.RunResult("", "", {}, {k: k for k in payload.answers})
                        self.run_result = self.live_result
                    self.live_result.rounds.append(payload)
                    self.on_live_update(f"Round {payload.index} finished", first=(payload.index == 0))
                elif kind == "judge":
                    if self.live_result is not None:
                        self.live_result.judge = payload
                        self.on_live_update("Judge ruling finished")
                elif kind == "gate":
                    self.on_gate(payload)
                elif kind == "test":
                    key, route, msg = payload
                    self.status_vars[key][route].set(msg)
                    self.test_btns[key][route].configure(state="normal")
                elif kind == "done":
                    self.on_run_done(payload)
                elif kind == "error":
                    self.on_run_error(payload)
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def on_live_update(self, what: str, first: bool = False) -> None:
        """A new round or judge ruling arrived for the run in progress."""
        live = self.live_result
        if self.run_result is live:
            # Only jump to the new view if the user was already on the latest one.
            values = self.round_combo["values"]
            on_latest = not values or self.round_combo.current() >= len(values) - 1
            self.populate_views(live, select_last=on_latest or first)
            if first:
                self.notebook.select(self.tab_results)
            if not on_latest and not first:
                self.set_status(f"{what}. Pick it in the View dropdown when you are ready.")
        else:
            self.set_status(f"{what} in the current run. Click 'Show current run' on the Results tab to see it.")

    def show_live_run(self) -> None:
        if self.live_result is None:
            return
        self.run_result = self.live_result
        self.populate_views(self.live_result, select_last=True)
        self.populate_reports(self.live_result.out_dir)
        self.notebook.select(self.tab_results)

    def on_run_done(self, result: core.RunResult) -> None:
        viewing_live = self.run_result is self.live_result
        self.live_result = result
        self.finish_ui()
        self.live_btn.configure(state="normal")
        if result.cancelled:
            self.set_status("Run cancelled. Partial results saved to " + result.out_dir)
        else:
            self.set_status("Run finished. Results saved to " + result.out_dir)
            self.bell()
        self.refresh_history()
        if viewing_live:
            self.run_result = result
            self.populate_views(result, select_last=True)
            self.populate_reports(result.out_dir)
            if result.rounds:
                self.notebook.select(self.tab_results)
        else:
            self.set_status(self.status.get() + "  Click 'Show current run' on the Results tab to see it.")

    def on_run_error(self, message: str) -> None:
        self.finish_ui()
        self.append_log("ERROR: " + message)
        self.set_status("Run failed")
        messagebox.showerror("Run failed", message)

    def finish_ui(self) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.ask_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        for b in (self.gate_continue_btn, self.gate_judge_btn, self.gate_stop_btn):
            b.configure(state="disabled")
        self.note_frame.configure(text="Your note for the next step")
        self.gate_state = None
        self.procs = None

    # ---------------------------------------------------------------- results
    def populate_views(self, result: core.RunResult, select_last: bool = False) -> None:
        name = os.path.basename(result.out_dir) if result.out_dir else "run"
        live = result is self.live_result and self.procs is not None
        self.results_run.set(f"Run {name}" + (" (in progress)" if live else ""))
        views = []
        for rr in result.rounds:
            rname = "independent answers" if rr.index == 0 else "after revision"
            extra = f", {rr.values_argued} with differing numbers" if rr.values_argued else ""
            views.append(f"Round {rr.index} - {rname}   ({rr.full:.0%} held by all{extra})")
        if result.judge:
            single = bool(result.plan) and not result.plan.is_list
            views.append(("The answer, by " if single else "Judge ruling by ") + result.judge.label)
            views.append("Final answer (all attributes, sortable)")
        self.round_combo["values"] = views
        if views:
            idx = len(views) - 1 if select_last else max(0, self.round_combo.current())
            self.round_combo.current(idx)
            self.show_view(idx)
            return
        # Nothing finished yet: show an empty table rather than a previous run's data.
        self.round_combo.set("")
        self.attr_combo.set("")
        self.attr_combo.configure(state="disabled")
        self.matrix.delete(*self.matrix.get_children())
        self.results_summary.set("Waiting for round 0 to finish. Progress is on the Run tab." if live
                                 else "This run has no finished rounds.")
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.configure(state="disabled")

    def _setup_columns(self, keys, labels, with_judge: bool) -> None:
        cols = ["key", "label"] + keys + ["held", "average"] + (["judge"] if with_judge else [])
        self.matrix["columns"] = cols
        self._sort_state = ("", False)
        heads = {"key": "Key", "label": "Label", "held": "Held", "average": "Average", "judge": "Judge"}
        heads.update({k: labels[k] for k in keys})
        for c in cols:
            self.matrix.heading(c, text=heads[c] + "  ↕", command=lambda c=c: self.sort_matrix(c))
        self.matrix.column("key", width=120, anchor="w", stretch=False)
        self.matrix.column("label", width=240, anchor="w")
        for k in keys:
            self.matrix.column(k, width=140, anchor="e", stretch=False)
        self.matrix.column("held", width=70, anchor="center", stretch=False)
        self.matrix.column("average", width=100, anchor="e", stretch=False)
        if with_judge:
            self.matrix.column("judge", width=150, anchor="e", stretch=False)

    def sort_matrix(self, col: str) -> None:
        """Click a column header to sort by it; click again to reverse. Numbers sort as numbers, blanks last."""
        import re as _re
        col_index = list(self.matrix["columns"]).index(col)
        rows = [iid for iid in self.matrix.get_children("") if iid != "total"]
        prev_col, prev_rev = self._sort_state
        reverse = not prev_rev if prev_col == col else False
        self._sort_state = (col, reverse)

        def sort_key(iid):
            text = str(self.matrix.item(iid, "values")[col_index]).replace("≠", "").strip()
            m = _re.search(r"-?\d[\d,]*\.?\d*", text)
            if text == "" or text in ("held", "excluded"):
                return (2, 0, text)
            if m:
                try:
                    return (0, float(m.group(0).replace(",", "")), text.lower())
                except ValueError:
                    pass
            return (1, 0, text.lower())

        ordered = sorted(rows, key=sort_key, reverse=reverse)
        if reverse:  # keep blanks at the bottom either way
            ordered = [i for i in ordered if sort_key(i)[0] != 2] + [i for i in ordered if sort_key(i)[0] == 2]
        for pos, iid in enumerate(ordered):
            self.matrix.move(iid, "", pos)
        if "total" in self.matrix.get_children(""):
            self.matrix.move("total", "", len(ordered))

    def _sync_attr_combo(self, attributes: list) -> str:
        """Keep the attribute dropdown in step with the round being shown; return the chosen attribute."""
        current = self.attr_combo.get()
        self.attr_combo["values"] = attributes
        if attributes:
            self.attr_combo.configure(state="readonly")
            self.attr_combo.current(attributes.index(current) if current in attributes else 0)
            return self.attr_combo.get()
        self.attr_combo.set("")
        self.attr_combo.configure(state="disabled")
        return ""

    def _cell(self, row: core.MatrixRow, k: str, attr: str) -> str:
        if not row.held[k]:
            return ""
        if not attr:
            return "held"
        text = row.text(k, attr) or "held"
        return ("≠ " + text) if row.attr_agreement(attr) is False else text

    def show_view(self, idx: int) -> None:
        result = self.run_result
        if not result or idx < 0:
            return
        if idx == len(result.rounds) + 1:
            self.show_final_table()
            return
        if idx >= len(result.rounds):
            self.show_judge_view()
            return
        rr = result.rounds[idx]
        keys = list(rr.answers)
        labels = {k: result.participants.get(k, k) for k in keys}
        attr = self._sync_attr_combo(rr.attributes)
        self._setup_columns(keys, labels, with_judge=False)
        self.matrix.delete(*self.matrix.get_children())
        for i, row in enumerate(rr.matrix):
            cells = [row.key, row.label] + [self._cell(row, k, attr) for k in keys]
            cells += [f"{row.count}/{len(keys)}", num(row.average(attr)) if attr else ""]
            self.matrix.insert("", "end", iid=str(i), values=cells, tags=(row.status(),))
        if attr:
            sums = [num(core.column_sum(rr.matrix, k, attr)) for k in keys]
            if any(sums):
                self.matrix.insert("", "end", iid="total", values=["SUM", ""] + sums + ["", ""])
        single = bool(result.plan) and not result.plan.is_list
        prefix = "Single-answer question: rows are the claims each model relied on. " if single else ""
        self.results_summary.set(prefix + core.round_summary(rr) + ".  Click a row for details.")
        self.show_round_notes(rr, labels)

    def show_judge_view(self) -> None:
        result = self.run_result
        if not result or not result.judge or not result.final:
            return
        rr = result.final
        keys = list(rr.answers)
        labels = {k: result.participants.get(k, k) for k in keys}
        jans = result.judge.answer
        attrs = list(rr.attributes)
        for it in jans.items:
            for a in it.attributes:
                if core.norm_attr(a.name) not in {core.norm_attr(x) for x in attrs} and a.name.strip():
                    attrs.append(a.name.strip())
        attr = self._sync_attr_combo(attrs)
        self._setup_columns(keys, labels, with_judge=True)
        self.matrix.delete(*self.matrix.get_children())
        jmap = core.judge_lookup(jans)
        seen = set()
        i = 0
        for row in rr.matrix:
            j = core.judge_item(jmap, row)
            if j:
                seen.add(id(j))
            ja = core.item_attr(j, attr) if (j and attr) else None
            cells = [row.key, row.label] + [self._cell(row, k, attr) for k in keys]
            cells += [f"{row.count}/{len(keys)}", num(row.average(attr)) if attr else "",
                      ("included " + (ja.value if ja else "")).strip() if j else "excluded"]
            self.matrix.insert("", "end", iid=str(i), values=cells, tags=("included" if j else "excluded",))
            i += 1
        self._judge_added = []
        for it in jans.items:  # items the judge added that nobody held
            if id(it) not in seen:
                ja = core.item_attr(it, attr) if attr else None
                cells = [it.key, it.label] + ["" for _ in keys] + [f"0/{len(keys)}", "",
                                                                    ("added " + (ja.value if ja else "")).strip()]
                self.matrix.insert("", "end", iid=f"j{len(self._judge_added)}", values=cells, tags=("included",))
                self._judge_added.append(it)
        if attr:
            nums = [core.item_attr(it, attr) for it in jans.items]
            nums = [a.number for a in nums if a and a.number is not None]
            if nums:
                self.matrix.insert("", "end", iid="total", values=["SUM", ""] + ["" for _ in keys] + ["", "", num(sum(nums))])
        single = bool(result.plan) and not result.plan.is_list
        if single:
            self.results_summary.set("Single-answer question: the judge's answer is below. Rows are the claims "
                                     "the models relied on.")
        else:
            self.results_summary.set(f"Judge kept {len(jans.items)} items.  Click a row for the judge's rationale.")
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        if result.notes.get("judge"):
            self.notes.insert("end", "Your note before the judge pass\n", "h")
            self.notes.insert("end", result.notes["judge"] + "\n\n")
        if single:
            self.notes.insert("end", f"The answer, by {result.judge.label}\n", "h")
            self.notes.insert("end", jans.report_markdown.strip() + "\n\n")
        self.notes.insert("end", f"Rulings by {result.judge.label}\n", "h")
        self.notes.insert("end", jans.changes.strip() + "\n\n")
        self.notes.insert("end", "Could not verify: ", "h")
        self.notes.insert("end", jans.unverified.strip() + "\n")
        self.notes.configure(state="disabled")

    def show_final_table(self) -> None:
        """The judge's final list with every attribute as its own column, sortable by any of them."""
        result = self.run_result
        if not result or not result.judge or not result.final:
            return
        jans = result.judge.answer
        final = result.final
        n = len(final.answers)
        attrs: list = []
        for it in jans.items:
            for a in it.attributes:
                if a.name.strip() and core.norm_attr(a.name) not in {core.norm_attr(x) for x in attrs}:
                    attrs.append(a.name.strip())
        cols = ["key", "label", "held"] + [f"a{i}" for i in range(len(attrs))]
        self.matrix["columns"] = cols
        self._sort_state = ("", False)
        heads = {"key": "Key", "label": "Label", "held": "Held"}
        heads.update({f"a{i}": attrs[i] for i in range(len(attrs))})
        for c in cols:
            self.matrix.heading(c, text=heads[c] + "  ↕", command=lambda c=c: self.sort_matrix(c))
        self.matrix.column("key", width=130, anchor="w", stretch=False)
        self.matrix.column("label", width=220, anchor="w")
        self.matrix.column("held", width=70, anchor="center", stretch=False)
        for i in range(len(attrs)):
            self.matrix.column(f"a{i}", width=150, anchor="e", stretch=False)
        self.attr_combo.set("")
        self.attr_combo.configure(state="disabled")
        self.matrix.delete(*self.matrix.get_children())
        self._final_items = []
        jmap = core.judge_lookup(jans)
        row_for = {}
        for row in final.matrix:
            ji = core.judge_item(jmap, row)
            if ji is not None:
                row_for[id(ji)] = row
        sums = [0.0] * len(attrs)
        counts = [0] * len(attrs)
        for i, it in enumerate(jans.items):
            row = row_for.get(id(it))
            held = f"{row.count}/{n}" if row else f"0/{n}"
            cells = [it.key, it.label, held]
            for j, a in enumerate(attrs):
                at = core.item_attr(it, a)
                cells.append(at.value if at else "")
                if at and at.number is not None:
                    sums[j] += at.number
                    counts[j] += 1
            tag = "included" if not row else row.status()
            self.matrix.insert("", "end", iid=f"f{i}", values=cells, tags=(tag,))
            self._final_items.append((it, row))
        if any(c >= 2 for c in counts):
            self.matrix.insert("", "end", iid="total", values=["SUM", "", ""] + [num(sums[j]) if counts[j] >= 2 else "" for j in range(len(attrs))])
        self.results_summary.set(f"The judge's final {len(jans.items)} items with every attribute. Click a header to sort; "
                                 "row color shows how many models held the item. Click a row for the judge's reasoning.")
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("end", f"Rulings by {result.judge.label}\n", "h")
        self.notes.insert("end", jans.changes.strip() + "\n")
        self.notes.configure(state="disabled")

    def show_round_notes(self, rr: core.RoundResult, labels: Dict[str, str]) -> None:
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        user_note = (self.run_result.notes or {}).get(str(rr.index)) if self.run_result else None
        if user_note:
            self.notes.insert("end", "Your note before this round\n", "h")
            self.notes.insert("end", user_note + "\n\n")
        if rr.merges:
            self.notes.insert("end", "Names the program merged this round\n", "h")
            self.notes.insert("end", "\n".join(rr.merges) + "\n\n")
        for k, ans in rr.answers.items():
            self.notes.insert("end", f"{labels[k]}\n", "h")
            if rr.index > 0:
                self.notes.insert("end", f"Changes: {ans.changes.strip()}\n")
            self.notes.insert("end", f"Could not verify: {ans.unverified.strip()}\n\n")
        self.notes.configure(state="disabled")

    def on_matrix_select(self, _event=None) -> None:
        result = self.run_result
        sel = self.matrix.selection()
        if not result or not sel or sel[0] == "total":
            return
        idx = self.round_combo.current()
        judge_view = idx >= len(result.rounds)
        rr = result.final if judge_view else result.rounds[idx]
        labels = {k: result.participants.get(k, k) for k in rr.answers}
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")

        def attrs_line(attributes) -> str:
            return "; ".join(f"{a.name}: {a.value}" for a in attributes)

        if sel[0].startswith("f"):
            it, row = self._final_items[int(sel[0][1:])]
            self.notes.insert("end", f"{it.key} - {it.label} (judge's final answer)\n", "h")
            self.notes.insert("end", (attrs_line(it.attributes) + "\n" if it.attributes else "") + it.rationale.strip() + "\n")
            if row:
                for k in rr.answers:
                    if row.held[k]:
                        self.notes.insert("end", f"\n{labels[k]}\n", "h")
                        line = attrs_line(row.attrs[k].values())
                        if line:
                            self.notes.insert("end", line + "\n")
                        self.notes.insert("end", row.rationales[k].strip() + "\n")
                    else:
                        self.notes.insert("end", f"\n{labels[k]}: not held\n", "h")
            self.notes.configure(state="disabled")
            return
        if sel[0].startswith("j"):
            added = getattr(self, "_judge_added", [])
            it = added[int(sel[0][1:])] if int(sel[0][1:]) < len(added) else None
            if it:
                self.notes.insert("end", f"{it.key} - {it.label} (added by the judge)\n", "h")
                self.notes.insert("end", (attrs_line(it.attributes) + "\n" if it.attributes else "") + it.rationale + "\n")
        else:
            row = rr.matrix[int(sel[0])]
            self.notes.insert("end", f"{row.key} - {row.label}\n", "h")
            for k in rr.answers:
                if row.held[k]:
                    self.notes.insert("end", f"\n{labels[k]}\n", "h")
                    line = attrs_line(row.attrs[k].values())
                    if line:
                        self.notes.insert("end", line + "\n")
                    self.notes.insert("end", row.rationales[k].strip() + "\n")
                else:
                    self.notes.insert("end", f"\n{labels[k]}: not held\n", "h")
            if judge_view and result.judge:
                j = core.judge_item(core.judge_lookup(result.judge.answer), row)
                self.notes.insert("end", f"\nJudge ({result.judge.label}): {'included' if j else 'excluded'}\n", "h")
                if j:
                    line = attrs_line(j.attributes)
                    if line:
                        self.notes.insert("end", line + "\n")
                    self.notes.insert("end", j.rationale.strip() + "\n")
        self.notes.configure(state="disabled")

    def export_csv(self) -> None:
        result = self.run_result
        if not result or not result.rounds:
            messagebox.showinfo("Nothing to export", "Run or load a consensus first.")
            return
        src = os.path.join(result.out_dir, "final_table.csv")
        if not os.path.exists(src):
            messagebox.showinfo("Not available", "The CSV is written when a run finishes.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="consensus.csv",
                                            filetypes=[("CSV", "*.csv")])
        if dest:
            shutil.copyfile(src, dest)
            self.set_status(f"Exported {dest}")

    # ---------------------------------------------------------------- reports
    def populate_reports(self, out_dir: str) -> None:
        self.reports_dir = out_dir
        files = []
        if out_dir and os.path.isdir(out_dir):
            files = sorted(f for f in os.listdir(out_dir) if f.endswith(".md"))
            order = {"story.md": 0, "consolidated.md": 1, "judge.md": 2, "plan.md": 3, "notes.md": 4}
            files.sort(key=lambda f: (order.get(f, 9), f))
        self.report_combo["values"] = files
        if files:
            self.report_combo.current(0)
            self.show_report()

    def show_report(self) -> None:
        name = self.report_combo.get()
        self.report_view.configure(state="normal")
        self.report_view.delete("1.0", "end")
        if name:
            try:
                with open(os.path.join(self.reports_dir, name), encoding="utf-8") as f:
                    self.report_view.insert("1.0", f.read())
            except OSError as e:
                self.report_view.insert("1.0", str(e))
        self.report_view.configure(state="disabled")

    def open_report_external(self) -> None:
        name = self.report_combo.get()
        if name:
            os.startfile(os.path.join(self.reports_dir, name))

    # ---------------------------------------------------------------- history
    def refresh_history(self) -> None:
        self.history_list.delete(0, "end")
        self._history_runs = core.list_runs(include_incomplete=True)
        live_dir = os.path.normcase(self.live_result.out_dir) if (self.live_result and self.procs is not None) else ""
        for d in self._history_runs:
            label = d
            full = os.path.join(core.RESULTS_DIR, d)
            if live_dir and os.path.normcase(full) == live_dir:
                self.history_list.insert("end", f"{d:<18} RUNNING NOW. Results appear here when it finishes.")
                continue
            if not core.run_is_complete(full):
                self.history_list.insert("end", f"{d:<18} incomplete: no round finished (failed or stopped early). "
                                                "Nothing to load; delete it if you like.")
                continue
            try:
                s = core.run_summary(os.path.join(core.RESULTS_DIR, d))
                parts = " + ".join(s["participants"])
                inputs = ", ".join(f"{k}={v}" for k, v in s["inputs"].items())
                extras = (f"  judge: {s['judge']}" if s["judge"] else "") + ("  (cancelled)" if s["cancelled"] else "")
                label = (f"{d:<18} {s['revisions']} revision(s)  {s['majority']:>4.0%} consensus   {parts}   "
                         f"{inputs}{extras}")
            except Exception:
                pass
            self.history_list.insert("end", label)

    def selected_history_dir(self) -> Optional[str]:
        sel = self.history_list.curselection()
        return os.path.join(core.RESULTS_DIR, self._history_runs[sel[0]]) if sel else None

    def load_history_run(self) -> None:
        d = self.selected_history_dir()
        if not d:
            return
        if not core.run_is_complete(d):
            messagebox.showinfo("Incomplete run", "This run never finished a round, so there are no results to show. "
                                                  "Use 'Open folder' to see what it left behind, or delete it.")
            return
        try:
            self.run_result = core.load_run(d)
        except Exception as e:
            messagebox.showerror("Could not load run", str(e))
            return
        self.populate_views(self.run_result, select_last=True)
        self.populate_reports(d)
        self.set_status(f"Loaded {d}")
        self.notebook.select(self.tab_results)

    def open_history_folder(self) -> None:
        d = self.selected_history_dir()
        if d:
            os.startfile(d)

    def delete_history_run(self) -> None:
        d = self.selected_history_dir()
        if not d:
            return
        if self.procs is not None and self.live_result and os.path.normcase(d) == os.path.normcase(self.live_result.out_dir):
            messagebox.showinfo("Run in progress", "That run is still going. Stop it first if you want to delete it.")
            return
        if messagebox.askyesno("Delete run", f"Delete this run and all its reports?\n{d}"):
            shutil.rmtree(d, ignore_errors=True)
            self.refresh_history()

    def open_results_folder(self) -> None:
        d = self.run_result.out_dir if self.run_result and self.run_result.out_dir else core.RESULTS_DIR
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    # ------------------------------------------------------------------- misc
    def append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status.set(text)

    def rounds_value(self) -> int:
        """The Revision rounds spinbox as an int; a blank or junk entry falls back to 1."""
        try:
            return max(0, min(10, int(self.rounds_var.get())))
        except (tk.TclError, ValueError):
            self.rounds_var.set(1)
            return 1

    def persist_settings(self) -> None:
        s = self.settings
        s["mock"] = bool(self.mock_var.get())
        for k in core.PARTICIPANT_KEYS:
            s[f"via_{k}"] = self.via_vars[k].get()
            s.pop(f"model_{k}", None)
        s["rounds"] = self.rounds_value()
        s["stop_early"] = bool(self.stop_early_var.get())
        s["judge"] = self.judge_var.get()
        s["plan"] = bool(self.plan_var.get())
        s["pause"] = bool(self.pause_var.get())
        s["merge_synonyms"] = bool(self.merge_var.get())
        s["web_search"] = bool(self.search_var.get())
        for k in core.PARTICIPANT_KEYS:
            s[f"use_{k}"] = bool(self.use_vars[k].get())
        s["geometry"] = self.geometry()
        inputs = s.setdefault("inputs", {})
        for name, var in self.input_vars.items():
            if var.get().strip():
                inputs[name] = var.get().strip()
        save_settings(s)

    def on_close(self) -> None:
        if not self.confirm_discard():
            return
        if self.procs is not None:
            if not messagebox.askyesno("Run in progress", "A run is in progress. Stop it and quit?"):
                return
            if self.gate_state and not self.gate_state["event"].is_set():
                self.resolve_gate("stop")
            self.procs.cancel()
            # Give the engine a moment to finish writing the run's files before the process exits.
            t = getattr(self, "worker_thread", None)
            if t is not None and t.is_alive():
                self.set_status("Stopping the run and saving what finished...")
                self.update_idletasks()
                t.join(timeout=8)
        self.persist_settings()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
