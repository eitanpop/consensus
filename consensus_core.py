"""
Consensus engine: several LLMs answer the same prompt, argue over it for a number of
revision rounds, and an optional judge issues a final ruling.

Round 0: every participant answers the prompt independently.
Round k: every participant sees the other participants' latest answers and revises,
         under program instructions that forbid blindly following or stubbornly ignoring.
Judge:   one participant reads every final answer plus the agreement matrix and rules
         on each argued item, producing the final answer.

Consensus is by majority: an item held by more than half the participants is "consensus";
anything else is "argued". Every participant holding the same items is "full agreement".

The engine knows nothing about the subject. Each answer is a written report plus a list of
items. An item has a short key, a label, a rationale, and a list of named attributes that
the prompt itself defines: for a stock prompt that is "amount: $1,200"; for "height and
weight of the top fighters by attractiveness" it is rank, height and weight. The Results
view lets you pick which attribute the columns show.

Participants:
    claude  - the `claude` CLI (Claude subscription, web search)
    codex   - the `codex` CLI (ChatGPT subscription, web search)
    gemini  - the `gemini` CLI (Google account, web search)
    claude_api / openai_api - the SDKs with API keys
    mock_a / mock_b / mock_c - canned answers for trying the UI without any LLM

Prompt wildcards (anywhere in the prompt):
    {{NAME}}            ask for a value
    {{NAME|default}}    ask, pre-filled with a default
    {{NAME|a|b|c}}      a dropdown of choices (first is the default; typing is still allowed)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

APP_NAME = "Consensus"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(APP_DIR, "prompts")
RESULTS_DIR = os.path.join(APP_DIR, "results")
INSTRUCTIONS_PATH = os.path.join(APP_DIR, "program_instructions.json")

CLI_TIMEOUT_SECONDS = int(os.getenv("CLI_TIMEOUT_SECONDS", "2400"))


# ---------------------------------------------------------------------------
# Answer schema (generic)
# ---------------------------------------------------------------------------

class Attribute(BaseModel):
    name: str = Field(description="Attribute name, lowercase and plain, identical for every item: "
                                  "e.g. rank, amount, height, weight, score, year")
    value: str = Field(description="The value as text including its unit if any: '5 ft 11 in', '$1,200', '3'")
    number: Optional[float] = Field(description="The numeric part of value when it is a quantity, in the unit "
                                                "stated in value. null when the value is not numeric.")


class AnswerItem(BaseModel):
    key: str = Field(description="Short canonical identifier for the item that another model would also use: "
                                 "a ticker, a person's full name, a product name, an option label. Reuse the "
                                 "exact key another answer used when you mean the same item.")
    label: str = Field(description="Human-readable name of the item")
    attributes: List[Attribute] = Field(description="Exactly the per-item facts or quantities the prompt asks "
                                                    "for, the same set and order for every item. Empty if the "
                                                    "prompt asks for none.")
    rationale: str = Field(description="One sentence: why this item is in your answer")


class Answer(BaseModel):
    report_markdown: str = Field(description="The complete written answer the prompt asks for, in markdown, "
                                             "with every section, analysis, caveat and source. Everything "
                                             "except the final list of items.")
    items: List[AnswerItem] = Field(description="The final list the prompt asks for, one entry per item, in the "
                                                "order the prompt wants")
    unverified: str = Field(description="Anything you could not verify, or 'none'")
    changes: str = Field(description="If revising or judging: what changed, what you kept despite "
                                     "disagreement, and why. Otherwise 'n/a'")


class PlannedAttribute(BaseModel):
    name: str = Field(description="Plain lowercase attribute name, e.g. rank, amount, height, year")
    kind: str = Field(description="'number' or 'text'")
    unit: str = Field(description="For numbers: the single unit every model must use, e.g. 'USD', 'cm', 'kg', "
                                  "'years'. Empty for text or unitless numbers.")
    format: str = Field(description="The exact way to write the value, with an example: e.g. 'whole dollars with "
                                    "thousands separators, like $1,200', 'integer', 'two decimals, like 3.25', "
                                    "'four-digit year, like 1954', 'integer, 1 = best'")
    allowed_values: List[str] = Field(description="For text attributes: the closed list of values every model must "
                                                  "choose from, e.g. ['warm', 'mild', 'cold']. Empty for numbers "
                                                  "or when a closed list is impossible.")
    description: str = Field(description="What the attribute means, in a few words")


class Plan(BaseModel):
    is_list: bool = Field(description="true if the question asks for a set, list or ranking of things; false if "
                                      "it asks for a single answer, explanation, decision or opinion")
    item_noun: str = Field(description="What one item is, e.g. 'fighter', 'stock', 'book', 'option'. For a "
                                       "non-list question: 'claim'")
    key_format: str = Field(description="Exactly how every model must write an item's key so the same thing gets "
                                        "the same key, with an example: e.g. 'US ticker symbol in capitals, like "
                                        "CAMT', 'city name in English without the country, like Valencia', "
                                        "'full name as commonly written, like Conor McGregor', "
                                        "'the common English name, singular, like rain jacket'")
    attributes: List[PlannedAttribute] = Field(description="The per-item facts or quantities the question asks "
                                                           "for or clearly implies. Empty if none.")
    notes: str = Field(description="Ambiguities you resolved, in one line, e.g. 'assumed books, not films'. "
                                   "Or 'none'.")


class SynonymGroup(BaseModel):
    canonical_key: str = Field(description="The one key every model should use for this item from now on")
    members: List[str] = Field(description="Every key or label from the answers that refers to this same item, "
                                           "written exactly as given")
    reason: str = Field(description="Why these are the same thing, in a few words")


class Synonyms(BaseModel):
    groups: List[SynonymGroup] = Field(description="Only groups of two or more names that denote the same "
                                                   "real-world item. Empty if there are none.")


def harden_schema(model: type) -> Dict[str, Any]:
    """Pydantic schema with additionalProperties=false on every object (required by strict modes)."""
    schema = model.model_json_schema()

    def harden(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                harden(v)
        elif isinstance(node, list):
            for v in node:
                harden(v)

    harden(schema)
    return schema


def strict_json_schema() -> Dict[str, Any]:
    return harden_schema(Answer)


# ---------------------------------------------------------------------------
# Program instructions (editable in the app, saved to program_instructions.json)
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTIONS: Dict[str, str] = {
    "plan": (
        "You are writing the answer template for a question that several models will answer independently. "
        "The template must make two models that mean the same thing write it identically, so answers can be "
        "compared item by item. Read the question and decide:\n"
        "- whether it asks for a list of things or for a single answer, and what one item is;\n"
        "- the key format: the one unambiguous way to name an item, with an example (a ticker symbol, a city "
        "name in English without the country, a person's full name, a product's common singular name);\n"
        "- only the attributes the question explicitly asks for, plus a number the question's own criteria "
        "cannot be checked without (a rent cap needs rent). Never add nice-to-have attributes the question "
        "did not mention. For each: number or text, "
        "the single unit (the one the question uses, else the most common one), and the exact written form "
        "with an example. Money: whole units with thousands separators unless the question needs cents. "
        "Measurements: integers unless precision matters, and say how many decimals. Years: four digits. "
        "Scores: state the scale.\n"
        "- for a text attribute, a closed list of allowed values whenever one is reasonable (e.g. climate: "
        "warm | mild | cold), so models cannot paraphrase. Only leave it open when no sensible list exists.\n"
        "- a 'rank' attribute (integer, 1 = best) only when the question explicitly asks for an order, for "
        "example with the words top, best, rank, most, or 'in order'. Never add rank to a list where order "
        "does not matter, such as what to pack or what to buy.\n"
        "Do not answer the question. Do not invent attributes the question does not call for."
    ),
    "reconcile": (
        "Several models listed items under their own names. Find names that refer to the same real-world "
        "item and group them: the same company under ticker and full name, the same city spelled "
        "differently, the same product under a synonym. Choose one canonical key per group, preferring the "
        "form the question's key format asks for. Group only when you are confident they are the same "
        "thing; leave genuinely different items alone. Do not evaluate or rank the items."
    ),
    "base": (
        "Follow the user's prompt exactly. Put the entire written answer in report_markdown and the "
        "final list in items, following whatever count, total, limits, ordering and sizing rules the "
        "prompt specifies.\n"
        "- Keys: use a short canonical key for each item (a ticker, a person's full name, a product name) "
        "so another model would produce the same key for the same thing.\n"
        "- Attributes: give every item the same attributes, which are exactly the per-item facts or "
        "quantities the prompt asks for (for example amount, height, weight, year, score). Name them in "
        "plain lowercase words. If the prompt asks for an ordered list such as 'top 5', include a 'rank' "
        "attribute starting at 1. If the prompt asks for nothing per item, leave attributes empty. Use "
        "one unit per attribute across all items and state it in the value text; put the numeric part "
        "in number.\n"
        "- Flag anything you could not verify in the unverified field. Do not invent figures. "
        "Distinguish facts from opinion."
    ),
    "revision": (
        "Other models answered the same prompt. You are revising, not deferring.\n"
        "- Do not adopt an item, figure or claim just because another model stated it. Verify it first; "
        "if you cannot verify it, say so and do not rely on it.\n"
        "- Do not drop or keep an item just to match the others. Keep an item you still believe in "
        "and say why; drop one when their evidence genuinely beats yours.\n"
        "- Treat the other answers as a source of leads and of errors to catch. Look for mistakes in "
        "their numbers, dates, eligibility and logic, and point them out explicitly.\n"
        "- When you reuse an item another model listed, use their exact key for it, and use the same "
        "attribute names and units they used so the answers can be compared.\n"
        "- Where you still disagree after checking, state the disagreement and the evidence on each side "
        "rather than splitting the difference.\n"
        "- In the changes field list what you added, dropped, resized or kept despite disagreement, with "
        "the reason for each."
    ),
    "judge": (
        "You are the judge. Several models answered the prompt and revised their answers over one or "
        "more rounds. You have every final answer and a summary of which items they agree on.\n"
        "- Items held by a majority are the presumptive consensus. Keep each one unless you find a "
        "concrete, verified reason to exclude it, and say what that reason is.\n"
        "- Rule on every argued item (held by a minority): include or exclude, based on the evidence "
        "in the answers and your own verification, not on how many models held it.\n"
        "- Where the models give different values for the same attribute, decide which is right and "
        "say why.\n"
        "- Do not favor any model. Weigh the evidence and the fit with the prompt.\n"
        "- Produce one complete final answer in the format the prompt requires, including any totals, "
        "limits and sizing rules, using the same keys and attribute names as the answers.\n"
        "- In the changes field give your ruling on each argued item and on any consensus item you "
        "removed, one line each."
    ),
}


def load_instructions() -> Dict[str, str]:
    data = dict(DEFAULT_INSTRUCTIONS)
    try:
        with open(INSTRUCTIONS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for k in DEFAULT_INSTRUCTIONS:
            if isinstance(saved.get(k), str) and saved[k].strip():
                data[k] = saved[k]
    except (OSError, ValueError):
        pass
    return data


def save_instructions(data: Dict[str, str]) -> None:
    with open(INSTRUCTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump({k: data.get(k, DEFAULT_INSTRUCTIONS[k]) for k in DEFAULT_INSTRUCTIONS}, f, indent=2)


# ---------------------------------------------------------------------------
# Cancellable subprocess registry
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    pass


class ProcessRegistry:
    """Tracks running CLI processes so a run can be cancelled from another thread."""

    def __init__(self) -> None:
        self._procs: set = set()
        self._lock = threading.Lock()
        self.cancelled = threading.Event()

    def run(self, cmd: List[str], stdin_text: str, cwd: str, timeout: Optional[int] = None
            ) -> Tuple[int, str, str]:
        if self.cancelled.is_set():
            raise Cancelled()
        timeout = max(30, timeout or CLI_TIMEOUT_SECONDS)
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", cwd=cwd, creationflags=creationflags,
        )
        with self._lock:
            self._procs.add(proc)
            if self.cancelled.is_set():        # cancel() may have run between the check above and here
                self._kill(proc)
                self._procs.discard(proc)
                raise Cancelled()
        try:
            out, err = proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            raise RuntimeError(f"{os.path.basename(cmd[0])} timed out after {timeout}s "
                               "(raise CLI_TIMEOUT_SECONDS if the work needs longer)")
        finally:
            with self._lock:
                self._procs.discard(proc)
        if self.cancelled.is_set():
            raise Cancelled()
        return proc.returncode, out, err

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            self._kill(p)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

# ask_json(user_text, system_text, json_schema, procs) -> dict matching the schema
AskJsonFn = Callable[[str, str, Dict[str, Any], ProcessRegistry], Dict[str, Any]]


@dataclass
class Participant:
    key: str
    label: str
    ask_json: AskJsonFn

    def ask(self, user_text: str, system_text: str, procs: ProcessRegistry) -> Answer:
        return Answer.model_validate(self.ask_json(user_text, system_text, strict_json_schema(), procs))


def _require_cli(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' CLI not found on PATH. Install it and sign in.")
    return path


def extract_json(text: str) -> str:
    """Return the first balanced {...} object in text (handles ```json fences and chatter)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unterminated JSON object in output")


def make_claude_cli(model: str = "", quick: bool = False) -> Participant:
    """quick=True disables web tools, for short calls such as planning."""
    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        exe = _require_cli("claude")
        tools = "" if quick else "WebSearch,WebFetch"
        cmd = [
            exe, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--system-prompt", system_text,
            "--tools", tools,
            "--allowedTools", tools,
            "--strict-mcp-config",       # no MCP servers / claude.ai connectors
            "--disable-slash-commands",  # no skills
            "--no-session-persistence",
        ]
        if model:
            cmd += ["--model", model]
        code, out, err = procs.run(cmd, user_text, cwd=tempfile.gettempdir())
        if code != 0:
            raise RuntimeError(f"claude exited {code}:\n{err.strip() or out.strip()}")
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude returned non-JSON output:\n{out[:2000]}")
        if payload.get("is_error"):
            raise RuntimeError(f"claude reported an error: {payload.get('result')}")
        structured = payload.get("structured_output")
        if structured is None:
            raise RuntimeError(f"claude returned no structured_output. Raw result: {payload.get('result')}")
        return structured

    return Participant("claude", f"Claude ({model or 'default'})", ask_json)


def make_codex_cli(model: str = "", quick: bool = False) -> Participant:
    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        exe = _require_cli("codex")
        with tempfile.TemporaryDirectory(prefix="codex_consensus_") as workdir:
            schema_path = os.path.join(workdir, "schema.json")
            out_path = os.path.join(workdir, "last_message.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f)
            cmd = [exe] + ([] if quick else ["--search"]) + [
                "exec", "-",
                "--output-schema", schema_path,
                "--output-last-message", out_path,
                "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--color", "never",
                "-C", workdir,
            ]
            if model:
                cmd += ["--model", model]
            # codex exec has no system-prompt flag, so the instructions lead the prompt.
            code, out, err = procs.run(cmd, f"{system_text}\n\n{user_text}", cwd=workdir)
            if code != 0:
                raise RuntimeError(f"codex exited {code}:\n{err.strip() or out.strip()}")
            if not os.path.exists(out_path):
                raise RuntimeError(f"codex produced no final message. Output:\n{out[-2000:]}")
            with open(out_path, encoding="utf-8") as f:
                raw = f.read().strip()
        try:
            return json.loads(extract_json(raw))
        except ValueError as e:
            raise RuntimeError(f"codex final message was not JSON: {e}\n{raw[:2000]}")

    return Participant("codex", f"Codex ({model or 'default'})", ask_json)


GEMINI_FORMAT_NOTE = (
    "\n\nRESPONSE FORMAT: reply with a single JSON object and nothing else (no markdown fences, no "
    "commentary) matching this JSON schema exactly:\n{schema}"
)


def make_gemini_cli(model: str = "", quick: bool = False) -> Participant:
    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        exe = _require_cli("gemini")
        cmd = [exe, "--output-format", "json", "--approval-mode", "yolo"]
        if model:
            cmd += ["--model", model]
        # gemini has no system-prompt flag and no schema flag: instructions lead the prompt,
        # the schema is appended, and the JSON is extracted from the response text.
        full = f"{system_text}\n\n{user_text}" + GEMINI_FORMAT_NOTE.format(schema=json.dumps(schema))
        # Run from the app folder: gemini downgrades approval mode in untrusted folders, and a
        # stable folder can be trusted once in gemini's own settings.
        code, out, err = procs.run(cmd, full, cwd=APP_DIR)
        payload = None
        for stream in (out, err):
            try:
                payload = json.loads(extract_json(stream))
                break
            except ValueError:
                continue
        if payload is None:
            msg = (err or out).strip()
            if "Auth" in msg or "GEMINI_API_KEY" in msg:
                msg += ("\n\nGemini is not signed in. Open a terminal in the app folder, run `gemini`, choose "
                        "'Login with Google', then /quit. Or put GEMINI_API_KEY=... in a .env file next to the app.")
            raise RuntimeError(f"gemini exited {code} and returned no JSON:\n{msg[:2000]}")
        if isinstance(payload, dict) and payload.get("error"):
            msg = payload["error"].get("message", payload["error"]) if isinstance(payload["error"], dict) else payload["error"]
            if "Auth" in str(msg) or "GEMINI_API_KEY" in str(msg):
                msg = f"{msg}\n\nRun `gemini` once in a terminal and choose 'Login with Google' to sign in."
            raise RuntimeError(f"gemini reported an error: {msg}")
        if code != 0:
            raise RuntimeError(f"gemini exited {code}:\n{err.strip() or out.strip()}")
        text = payload.get("response", out) if isinstance(payload, dict) and "response" in payload else out
        if isinstance(text, dict):
            return text
        try:
            return json.loads(extract_json(str(text)))
        except ValueError as e:
            raise RuntimeError(f"gemini response did not contain the answer JSON: {e}\n{str(text)[:2000]}")

    return Participant("gemini", f"Gemini ({model or 'default'})", ask_json)


GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-flash-lite-latest"]
# One Gemini attempt (no web search, one model) should answer within minutes; a stalled connection
# must not hold a round for the full per-call timeout. Overridable with GEMINI_TIMEOUT_SECONDS.
GEMINI_ATTEMPT_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "420"))


def gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Pydantic JSON schema to the subset the Gemini API accepts (no $ref, uppercase types)."""
    defs = schema.get("$defs", {})

    def conv(node: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in node:
            node = defs[node["$ref"].split("/")[-1]]
        if "anyOf" in node:
            non_null = [n for n in node["anyOf"] if n.get("type") != "null"]
            out = conv(non_null[0]) if non_null else {"type": "STRING"}
            out["nullable"] = True
            return out
        t = node.get("type")
        out: Dict[str, Any] = {}
        if t == "object":
            props = node.get("properties", {})
            out["type"] = "OBJECT"
            out["properties"] = {k: conv(v) for k, v in props.items()}
            if node.get("required"):
                out["required"] = list(node["required"])
            out["propertyOrdering"] = list(props.keys())
        elif t == "array":
            out["type"] = "ARRAY"
            out["items"] = conv(node.get("items", {"type": "string"}))
        elif t in ("string", "number", "integer", "boolean"):
            out["type"] = t.upper()
        else:
            out["type"] = "STRING"
        if node.get("description"):
            out["description"] = node["description"]
        if node.get("enum"):
            out["enum"] = [str(e) for e in node["enum"]]
        return out

    return conv(schema)


class _GeminiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Gemini API {status}: {message}")
        self.status = status
        self.message = message


def _gemini_call(model: str, body: Dict[str, Any], api_key: str, timeout: int,
                 cancelled: Optional[threading.Event] = None) -> Dict[str, Any]:
    """One HTTP call to the Gemini API. Waits in a helper thread so Stop can interrupt a stalled request."""
    import urllib.error
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    box: Dict[str, Any] = {}

    def fetch() -> None:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                box["data"] = json.loads(resp.read().decode("utf-8"))
        except BaseException as e:  # noqa: BLE001 - handed back to the caller's thread
            box["error"] = e

    t = threading.Thread(target=fetch, daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.5)
        if cancelled is not None and cancelled.is_set():
            raise Cancelled()
    try:
        if "error" in box:
            raise box["error"]
        data = box["data"]
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8")).get("error", {})
            msg = err.get("message", str(e))
        except Exception:
            msg = str(e)
        raise _GeminiError(e.code, msg)
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), TimeoutError) or "timed out" in str(e.reason).lower():
            raise _GeminiError(599, f"no response from {model} within {timeout}s")
        raise RuntimeError(f"Gemini API connection error: {e.reason}")
    except TimeoutError:
        raise _GeminiError(599, f"no response from {model} within {timeout}s")
    if "error" in data:
        raise _GeminiError(int(data["error"].get("code", 500)), data["error"].get("message", ""))
    return data


def make_gemini_api(model: str = "", quick: bool = False) -> Participant:
    """
    Gemini through the Gemini API with GEMINI_API_KEY (the CLI no longer accepts personal accounts).
    Tries web search (Google Search grounding) first and drops it if the key's tier refuses it;
    steps down to older Flash models if the requested one is overloaded or unavailable.
    """
    requested = model or os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    # Learned during a run and reused by later calls of the same participant: whether the key
    # allows search grounding, and which model last answered (so we don't re-try overloaded ones).
    state: Dict[str, Any] = {"search_refused": False, "working_model": None}

    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Put GEMINI_API_KEY=... in a .env file next to the app "
                               "(get a key at aistudio.google.com).")
        if procs.cancelled.is_set():
            raise Cancelled()
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": gemini_schema(schema)},
        }
        chain = [requested] + [m for m in GEMINI_FALLBACK_MODELS if m != requested]
        if state["working_model"] in chain:
            chain = chain[chain.index(state["working_model"]):]   # start where the last call succeeded
        use_search = not quick and not state["search_refused"]
        search_dropped_this_call = False
        notes: List[str] = []
        if not quick and state["search_refused"]:
            notes.append("no web search: search grounding is not available on this key")
        last_error: Optional[Exception] = None
        for m in chain:
            for attempt in range(2):
                if procs.cancelled.is_set():
                    raise Cancelled()
                b = dict(body)
                if use_search:
                    b["tools"] = [{"google_search": {}}]
                try:
                    data = _gemini_call(m, b, api_key, min(GEMINI_ATTEMPT_TIMEOUT, CLI_TIMEOUT_SECONDS),
                                        procs.cancelled)
                    cand = (data.get("candidates") or [{}])[0]
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    if not text:
                        raise RuntimeError(f"Gemini ({m}) returned no text (finishReason={cand.get('finishReason')})")
                    out = json.loads(extract_json(text))
                    state["working_model"] = m
                    if search_dropped_this_call and not use_search:
                        # The same model answered as soon as search was removed: the key's tier refuses
                        # grounding, so skip search for the rest of the run. A plain rate limit would
                        # have failed the retry too, and is not made sticky.
                        state["search_refused"] = True
                    answered_by = data.get("modelVersion") or m
                    if m != requested:
                        notes.append(f"answered by {answered_by} because {requested} was unavailable")
                    if notes and isinstance(out, dict) and "unverified" in out:
                        out["unverified"] = "(" + "; ".join(notes) + ") " + str(out.get("unverified", ""))
                    return out
                except _GeminiError as e:
                    last_error = e
                    if e.status == 429 and use_search:
                        use_search = False                    # try the same model once without search
                        search_dropped_this_call = True
                        notes.append("no web search: search grounding is not available on this key")
                        continue
                    if e.status == 429:
                        break                                 # this model's quota is used up: try the next one
                    if e.status in (404, 503, 599) or "high demand" in e.message.lower():
                        break                                 # overloaded, retired or stalled: try the next model
                    raise
                except ValueError as e:
                    raise RuntimeError(f"Gemini ({m}) returned invalid JSON: {e}")
            else:
                continue
        raise RuntimeError(f"Gemini: every model in the fallback chain failed ({', '.join(chain)}). "
                           f"Last error: {last_error}")

    label = f"Gemini API ({model})" if model else "Gemini API"
    return Participant("gemini", label, ask_json)


def make_claude_api(model: str = "claude-opus-5", quick: bool = False) -> Participant:
    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic()

        def call():
            with client.messages.stream(
                model=model, max_tokens=64000, system=system_text,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user_text}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            ) as stream:
                return stream.get_final_message()

        response = run_in_thread(call, procs.cancelled)
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined the request (stop_reason=refusal).")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise RuntimeError(f"Claude returned no text output (stop_reason={response.stop_reason}).")
        return json.loads(text)

    return Participant("claude_api", f"Claude API ({model})", ask_json)


def make_openai_api(model: str = "gpt-5", quick: bool = False) -> Participant:
    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        import openai

        client = openai.OpenAI()
        completion = run_in_thread(lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_text},
                      {"role": "user", "content": user_text}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema.get("title", "output"), "schema": schema, "strict": True}},
        ), procs.cancelled)
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"OpenAI declined the request: {message.refusal}")
        if not message.content:
            raise RuntimeError("OpenAI returned no output.")
        return json.loads(message.content)

    return Participant("openai_api", f"OpenAI API ({model})", ask_json)


MOCK_PLAN = {"is_list": True, "item_noun": "item", "key_format": "ticker symbol in capitals, like CAMT",
             "attributes": [{"name": "amount", "kind": "number", "unit": "USD",
                             "format": "whole dollars with thousands separators, like $1,200",
                             "allowed_values": [], "description": "allocation"},
                            {"name": "rank", "kind": "number", "unit": "", "format": "integer, 1 = best",
                             "allowed_values": [], "description": "order"}],
             "notes": "mock plan"}


def make_mock(key: str, label: str, universe: List[Tuple[str, str]], delay: float = 1.0,
              quick: bool = False) -> Participant:
    """Canned participant: starts with its own list, then adopts some of what it is shown."""
    state = {"round": 0}

    def ask_json(user_text: str, system_text: str, schema: Dict[str, Any], procs: ProcessRegistry) -> Dict[str, Any]:
        for _ in range(int(delay * 10)):
            if procs.cancelled.is_set():
                raise Cancelled()
            time.sleep(0.1)
        if schema.get("title") == "Plan":
            return dict(MOCK_PLAN)
        if schema.get("title") == "Synonyms":
            # pretend 'TSEM Corp' and 'Tower Semi' were spotted as the same company
            return {"groups": [{"canonical_key": "TSEM", "members": ["TSEM", "Tower Semi", "TSEM Corp"],
                                "reason": "same company (mock)"}]}
        state["round"] += 1
        shown = re.findall(r"^- ([A-Z0-9.\-]{1,8}) \(", user_text, flags=re.M)
        keys = [k for k, _ in universe]
        extra: List[str] = []
        if "JUDGE" in user_text[:400].upper():
            counts: Dict[str, int] = {}
            for k in shown:
                counts[k] = counts.get(k, 0) + 1
            keys = [k for k, c in counts.items() if c >= 2] or list(counts)[:5]
        elif shown:
            extra = [k for k in dict.fromkeys(shown) if k not in keys][:2]
            keys = keys[:-1] + extra
        names = dict(universe)
        n = len(keys)
        per = round(10000.0 / n, 2)
        items = []
        for i, k in enumerate(keys):
            amt = per if i < n - 1 else round(10000.0 - per * (n - 1), 2)
            items.append(AnswerItem(
                key=k, label=names.get(k, f"{k} Corp"),
                attributes=[Attribute(name="amount", value=f"${amt:,.0f}", number=amt),
                            Attribute(name="rank", value=str(i + 1), number=i + 1)],
                rationale=f"{label} rationale for {k}, round {state['round']}"))
        report = f"# {label} report, round {state['round']}\n\n" + "\n".join(
            f"## {it.key}\nMock analysis of {it.label}. Not real data.\n" for it in items)
        return Answer(report_markdown=report, items=items, unverified="everything (mock)",
                      changes="n/a" if state["round"] == 1 else f"Adopted {extra} after cross-reference.").model_dump()

    return Participant(key, label, ask_json)


MOCK_A = [("CYBR", "CyberArk"), ("MNDY", "monday.com"), ("NICE", "NICE Ltd"), ("GLBE", "Global-e"), ("CAMT", "Camtek")]
MOCK_B = [("MNDY", "monday.com"), ("GLBE", "Global-e"), ("FROG", "JFrog"), ("TSEM", "Tower Semi"), ("NVMI", "Nova")]
MOCK_C = [("GLBE", "Global-e"), ("CYBR", "CyberArk"), ("WIX", "Wix"), ("TSEM", "Tower Semi"), ("INMD", "InMode")]

PARTICIPANT_KEYS = ["claude", "codex", "gemini"]


# How each participant can be reached: "cli" = the vendor's command-line tool on a subscription,
# "api" = the vendor's API with a key from .env. Gemini is API-only (its tool no longer accepts personal logins).
VIA_OPTIONS: Dict[str, List[str]] = {"claude": ["cli", "api"], "codex": ["cli", "api"], "gemini": ["api"]}
CLI_TOOL_KEYS = ("claude", "codex")
VIA_LABELS = {"cli": "CLI (subscription)", "api": "API (key)"}
API_DEFAULT_MODELS = {"claude": "claude-opus-5", "codex": "gpt-5", "gemini": GEMINI_DEFAULT_MODEL}


@dataclass
class ParticipantConfig:
    enabled: Dict[str, bool] = field(default_factory=lambda: {"claude": True, "codex": True, "gemini": False})
    via: Dict[str, str] = field(default_factory=lambda: {"claude": "cli", "codex": "cli", "gemini": "api"})
    models: Dict[str, str] = field(default_factory=lambda: {"claude": "", "codex": "", "gemini": ""})
    mock: bool = False                      # canned answers instead of any real model
    web_search: bool = True                 # off = models answer from memory only (much faster)


def make_participant(key: str, via: str, model: str = "", quick: bool = False) -> Participant:
    """One participant by key and route. Raises with a clear message when the route is not set up."""
    if key == "gemini" or via == "api":
        if key == "claude":
            if not os.getenv("ANTHROPIC_API_KEY", "").strip():
                raise RuntimeError("Claude via API needs ANTHROPIC_API_KEY (Settings tab), or switch Claude to CLI.")
            return make_claude_api(model or API_DEFAULT_MODELS["claude"], quick=quick)
        if key == "codex":
            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise RuntimeError("Codex via API needs OPENAI_API_KEY (Settings tab), or switch Codex to CLI.")
            return make_openai_api(model or API_DEFAULT_MODELS["codex"], quick=quick)
        if not os.getenv("GEMINI_API_KEY", "").strip():
            raise RuntimeError("Gemini needs GEMINI_API_KEY (Settings tab; free key at aistudio.google.com).")
        return make_gemini_api(model, quick=quick)
    if key == "claude":
        return make_claude_cli(model, quick=quick)
    if key == "codex":
        return make_codex_cli(model, quick=quick)
    raise ValueError(f"unknown participant {key}")


def build_participants(cfg: ParticipantConfig) -> List[Participant]:
    parts: List[Participant] = []
    if cfg.mock:
        mocks = {"claude": ("mock_a", "Mock A", MOCK_A), "codex": ("mock_b", "Mock B", MOCK_B),
                 "gemini": ("mock_c", "Mock C", MOCK_C)}
        for k in PARTICIPANT_KEYS:
            if cfg.enabled.get(k):
                parts.append(make_mock(*mocks[k]))
    else:
        for k in PARTICIPANT_KEYS:
            if cfg.enabled.get(k):
                parts.append(make_participant(k, cfg.via.get(k, "cli"), cfg.models.get(k, ""),
                                              quick=not cfg.web_search))
    if len(parts) < 2:
        raise RuntimeError("Enable at least two participants.")
    return parts


def make_planner(cfg: ParticipantConfig) -> Participant:
    """A quick, tool-less participant for planning the format and merging names: the first enabled one."""
    if cfg.mock:
        return make_mock("mock_planner", "Mock planner", MOCK_A, delay=0.3, quick=True)
    for k in PARTICIPANT_KEYS:
        if cfg.enabled.get(k):
            via = cfg.via.get(k, "cli")
            model = cfg.models.get(k, "")
            if k == "claude" and via == "cli" and not model:
                model = "sonnet"            # fast enough for a planning call
            return make_participant(k, via, model, quick=True)
    raise RuntimeError("Enable at least one participant.")


def plan_format(question: str, planner: Participant, instructions: Dict[str, str],
                procs: ProcessRegistry) -> Tuple[Plan, str]:
    """Ask the planner for the answer template; return the plan and the text to append to the prompt."""
    data = planner.ask_json(f"QUESTION:\n{question}", instructions["plan"], harden_schema(Plan), procs)
    plan = Plan.model_validate(data)
    return plan, render_plan(plan)


def render_plan(plan: Plan) -> str:
    lines = ["ANSWER FORMAT (planned by the program so every model's answer can be compared item by item):"]
    if plan.is_list:
        lines.append(f"- Each item is one {plan.item_noun or 'item'}.")
    else:
        lines.append("- This question asks for a single answer. Give the full answer in the report, and list "
                     "each distinct conclusion or recommendation it rests on as an item.")
    if plan.key_format.strip():
        lines.append(f"- Key: {plan.key_format.strip()}. Every model must write the key the same way.")
    if plan.attributes:
        lines.append("- Give every item exactly these attributes, in this order, written exactly as specified:")
        for a in plan.attributes:
            bits = [a.kind.strip() or "number"]
            if a.unit.strip():
                bits.append(f"unit {a.unit.strip()}")
            if a.format.strip():
                bits.append(a.format.strip())
            if a.allowed_values:
                bits.append("one of: " + " | ".join(v.strip() for v in a.allowed_values if v.strip()))
            desc = f" ({a.description.strip()})" if a.description.strip() else ""
            lines.append(f"    - {a.name.strip()}: " + "; ".join(bits) + desc)
        lines.append("  For number attributes put the numeric value in number and the formatted text in value. "
                     "For text attributes with a list, use one of the listed values verbatim.")
    else:
        lines.append("- No per-item attributes are needed.")
    if plan.notes.strip() and plan.notes.strip().lower() != "none":
        lines.append(f"- Assumption: {plan.notes.strip()}")
    return "\n".join(lines)


def plan_summary(plan: Plan) -> str:
    attrs = ", ".join(f"{a.name}" + (f" ({a.unit})" if a.unit.strip() else "") for a in plan.attributes) or "no attributes"
    kind = f"list of {plan.item_noun or 'items'}" if plan.is_list else "single answer"
    key = f"; key = {plan.key_format.strip()}" if plan.key_format.strip() else ""
    return f"{kind}{key}; {attrs}" + (f"; assumed: {plan.notes.strip()}" if plan.notes.strip().lower() not in ("", "none") else "")


def scored_text_attrs(plan: Optional[Plan]) -> set:
    """Text attributes with a closed list of values are comparable, so they get scored like numbers."""
    if not plan:
        return set()
    return {norm_attr(a.name) for a in plan.attributes if a.allowed_values}


def reconcile_keys(answers: Dict[str, Answer], labels: Dict[str, str], planner: Participant,
                   instructions: Dict[str, str], procs: ProcessRegistry) -> List[str]:
    """
    Ask the planner which differently-named items are the same thing, then rewrite those items'
    keys to one canonical key so every model carries the same key from now on.
    Returns human-readable lines describing the merges (empty if none).
    """
    rows = build_matrix(answers)
    if all(r.status() == "all" for r in rows) or len(answers) < 2:
        return []
    listing = []
    for pkey, ans in answers.items():
        listing.append(f"{labels.get(pkey, pkey)}:")
        listing += [f"  - key: {it.key} | label: {it.label}" for it in ans.items]
    text = ("Each model below listed these items (key | label). Group the names that refer to the same "
            "real-world item under one canonical key.\n\n" + "\n".join(listing))
    data = planner.ask_json(text, instructions["reconcile"], harden_schema(Synonyms), procs)
    groups = Synonyms.model_validate(data).groups
    mapping: Dict[str, str] = {}
    for g in groups:
        canon = g.canonical_key.strip()
        if not canon:
            continue
        for m in g.members:
            n = normalize_key(m)
            if n and n != normalize_key(canon):
                mapping[n] = canon
    if not mapping:
        return []
    merged: List[str] = []
    for pkey, ans in answers.items():
        for it in ans.items:
            canon = mapping.get(normalize_key(it.key)) or mapping.get(normalize_key(it.label))
            if canon and normalize_key(it.key) != normalize_key(canon):
                merged.append(f"{labels.get(pkey, pkey)}: '{it.key}' -> '{canon}'")
                it.key = canon
    return merged


# ---------------------------------------------------------------------------
# Prompt wildcards: {{NAME}}, {{NAME|default}}, {{NAME|a|b|c}}
# ---------------------------------------------------------------------------

WILDCARD = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*((?:\|[^|{}]*)*)\}\}")


@dataclass
class Wildcard:
    name: str
    default: str = ""
    choices: List[str] = field(default_factory=list)


def _parse_options(raw: str) -> Tuple[str, List[str]]:
    parts = [p.strip() for p in raw.split("|")[1:]] if raw else []
    parts = [p for p in parts if p != ""]
    if len(parts) >= 2:
        return parts[0], parts
    return (parts[0] if parts else ""), []


def find_wildcards(prompt_text: str) -> List[Wildcard]:
    """Unique wildcards in order of first appearance. A later occurrence may add a default or choices."""
    seen: Dict[str, Wildcard] = {}
    for m in WILDCARD.finditer(prompt_text):
        name = m.group(1).strip()
        default, choices = _parse_options(m.group(2))
        w = seen.setdefault(name, Wildcard(name))
        if not w.default and default:
            w.default = default
        if not w.choices and choices:
            w.choices = choices
    return list(seen.values())


def context_line(prompt_text: str, name: str) -> str:
    """The first line of the prompt that mentions the wildcard, for showing while asking."""
    for line in prompt_text.splitlines():
        if re.search(r"\{\{\s*" + re.escape(name) + r"\s*[|}]", line):
            return " ".join(line.split())
    return ""


def fill_wildcards(prompt_text: str, values: Dict[str, str]) -> str:
    wildcards = find_wildcards(prompt_text)
    missing = [w.name for w in wildcards if not values.get(w.name) and not w.default]
    if missing:
        raise ValueError(f"No value for: {', '.join(missing)}")
    defaults = {w.name: w.default for w in wildcards}

    def sub(m: re.Match) -> str:
        name = m.group(1).strip()
        return values.get(name) or defaults.get(name, "")

    return WILDCARD.sub(sub, prompt_text)


def with_date_header(prompt_text: str) -> str:
    return f"TODAY'S DATE: {date.today():%A, %B %d, %Y} (supplied by the program)\n\n{prompt_text}"


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

def list_prompts() -> List[str]:
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    return sorted(f[:-4] for f in os.listdir(PROMPTS_DIR) if f.lower().endswith(".txt"))


def prompt_path(name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    if not safe:
        raise ValueError("Prompt name is empty")
    return os.path.join(PROMPTS_DIR, safe + ".txt")


def load_prompt(name: str) -> str:
    with open(prompt_path(name), encoding="utf-8") as f:
        return f.read()


def save_prompt(name: str, text: str) -> str:
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = prompt_path(name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def delete_prompt(name: str) -> None:
    os.remove(prompt_path(name))


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def normalize_key(text: str) -> str:
    """
    Case- and accent-insensitive form that ignores separators but keeps distinguishing symbols:
    'BRK.B', 'brk-b' and '$BRK B' match; 'Málaga' matches 'Malaga'; but 'C', 'C++' and 'C#' stay distinct.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.strip().upper()
    folded = re.sub(r"^\$", "", folded)                      # '$CAMT' -> 'CAMT'
    return re.sub(r"[\s.\-_/\\,'\"()\[\]:;!?]", "", folded)  # separators and quotes only


def numbers_equal(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-9


def norm_attr(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


@dataclass
class MatrixRow:
    key: str
    label: str
    held: Dict[str, bool]                      # participant -> holds this item
    rationales: Dict[str, str]
    attrs: Dict[str, Dict[str, Attribute]]     # participant -> normalized attribute name -> Attribute
    count: int = 0

    def status(self) -> str:
        n = len(self.held)
        if self.count == n:
            return "all"
        if self.count * 2 > n:
            return "majority"
        return "minority"

    @property
    def is_consensus(self) -> bool:
        return self.status() in ("all", "majority")

    def holders(self) -> List[str]:
        return [k for k, h in self.held.items() if h]

    text_scored: set = field(default_factory=set)   # normalized names of text attributes with a closed value list

    def attr_agreement(self, attr: str) -> Optional[bool]:
        """True/False when every holder gave comparable values for attr and they are identical / not;
        None when the attribute is free text or missing for a holder (shown, not scored).
        Numbers compare exactly. Text compares after normalization, only for attributes the plan gave
        a closed list of values."""
        holders = self.holders()
        if len(holders) < 2:
            return None
        nums = [self.number(k, attr) for k in holders]
        if all(x is not None for x in nums):
            return all(numbers_equal(nums[0], x) for x in nums[1:])
        if norm_attr(attr) in self.text_scored:
            texts = [normalize_key(self.text(k, attr)) for k in holders]
            if all(texts):
                return all(t == texts[0] for t in texts[1:])
        return None

    def argued_attrs(self, attrs: List[str], ignore: Tuple[str, ...] = ("rank",)) -> List[str]:
        """Attributes whose numbers differ between holders (rank is reported per round, not per row)."""
        return [a for a in attrs if norm_attr(a) not in ignore and self.attr_agreement(a) is False]

    def attr(self, pkey: str, attr: str) -> Optional[Attribute]:
        return self.attrs.get(pkey, {}).get(norm_attr(attr))

    def text(self, pkey: str, attr: str) -> str:
        a = self.attr(pkey, attr)
        return a.value if a else ""

    def number(self, pkey: str, attr: str) -> Optional[float]:
        a = self.attr(pkey, attr)
        return a.number if a else None

    def average(self, attr: str) -> Optional[float]:
        nums = [self.number(k, attr) for k in self.attrs]
        nums = [x for x in nums if x is not None]
        return sum(nums) / len(nums) if nums else None


def attribute_names(answers: Dict[str, Answer]) -> List[str]:
    """Attribute display names across all answers, in order of first appearance (merged by normalized name)."""
    seen: Dict[str, str] = {}
    for a in answers.values():
        for it in a.items:
            for at in it.attributes:
                seen.setdefault(norm_attr(at.name), at.name.strip())
    return [v for v in seen.values() if v]


def build_matrix(answers: Dict[str, Answer]) -> List[MatrixRow]:
    """One row per distinct item. Items match on key, or on label when keys differ ('LOTR' vs 'The Lord of the Rings')."""
    keys = list(answers)
    rows: List[MatrixRow] = []
    alias: Dict[str, MatrixRow] = {}
    for pkey in keys:
        for it in answers[pkey].items:
            nk, nl = normalize_key(it.key), normalize_key(it.label)
            row = alias.get(nk) or alias.get(nl)
            if row is None:
                row = MatrixRow(it.key.strip(), it.label.strip(), {k: False for k in keys},
                                {k: "" for k in keys}, {k: {} for k in keys})
                rows.append(row)
            for a in (nk, nl):
                if a:
                    alias.setdefault(a, row)
            row.held[pkey] = True
            row.rationales[pkey] = it.rationale
            row.attrs[pkey] = {norm_attr(at.name): at for at in it.attributes if at.name.strip()}
    for r in rows:
        r.count = sum(r.held.values())
    first_attr = next(iter(attribute_names(answers)), None)
    rows.sort(key=lambda r: (-r.count, -(r.average(first_attr) or 0) if first_attr else 0, r.key))
    return rows


def full_agreement(rows: List[MatrixRow]) -> float:
    """Fraction of all items that every participant holds."""
    return (sum(1 for r in rows if r.status() == "all") / len(rows)) if rows else 0.0


def majority_agreement(rows: List[MatrixRow]) -> float:
    return (sum(1 for r in rows if r.is_consensus) / len(rows)) if rows else 0.0


def same_order(rows: List[MatrixRow], keys: List[str]) -> Optional[bool]:
    """Do all participants rank the items they share in the same order? None if there is no rank attribute."""
    ranked = [r for r in rows if r.status() == "all" and all(r.number(k, "rank") is not None for k in keys)]
    if not ranked or len(keys) < 2:
        return None
    orders = [tuple(r.key for r in sorted(ranked, key=lambda r: r.number(k, "rank"))) for k in keys]
    return all(o == orders[0] for o in orders[1:])


def values_fully_agreed(rows: List[MatrixRow], attrs: List[str]) -> bool:
    """Every item held by everyone has identical numbers for every scored attribute (rank excluded)."""
    return all(not r.argued_attrs(attrs) for r in rows if r.status() == "all")


def column_sum(rows: List[MatrixRow], pkey: str, attr: str) -> Optional[float]:
    nums = [r.number(pkey, attr) for r in rows if r.held.get(pkey)]
    nums = [x for x in nums if x is not None]
    return sum(nums) if nums else None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _num(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def item_as_text(it: AnswerItem) -> str:
    attrs = "; ".join(f"{a.name}: {a.value}" for a in it.attributes)
    return f"- {it.key} ({it.label})" + (f" [{attrs}]" if attrs else "") + f" - {it.rationale}"


def answer_as_text(a: Answer) -> str:
    lines = [a.report_markdown.strip(), "", "FINAL LIST:"] + [item_as_text(it) for it in a.items]
    if a.unverified and a.unverified.strip().lower() != "none":
        lines += ["", f"Could not verify: {a.unverified}"]
    if a.changes and a.changes.strip().lower() != "n/a":
        lines += ["", f"Changes from earlier answer: {a.changes}"]
    return "\n".join(lines)


def _block(title: str, body: str) -> str:
    bar = "-" * 80
    return f"\n{title}:\n{bar}\n{body}\n{bar}\n"


def notes_block(notes: Dict[str, str]) -> str:
    """The user's between-round notes, rendered for a prompt. Empty string if there are none."""
    if not notes:
        return ""
    lines = []
    for key, text in notes.items():
        when = "before the judge pass" if key == "judge" else f"before round {key}"
        lines.append(f"- Added {when}: {text.strip()}")
    return _block("NOTES FROM THE PERSON WHO ASKED (added between rounds; these amend the original prompt "
                  "and outrank any model's opinion)", "\n".join(lines))


def revision_prompt(prompt: str, me: Participant, others: List[Tuple[Participant, Answer]],
                    own: Answer, round_no: int, total_rounds: int, instructions: Dict[str, str],
                    notes: Optional[Dict[str, str]] = None) -> str:
    text = (f"{len(others)} other model(s) answered the same prompt. This is revision round {round_no} of "
            f"{total_rounds}. Your latest answer and theirs are below.\n")
    text += _block("ORIGINAL PROMPT", prompt)
    text += notes_block(notes or {})
    text += _block("YOUR LATEST ANSWER", answer_as_text(own))
    for p, a in others:
        text += _block(f"ANOTHER MODEL'S LATEST ANSWER ({p.label})", answer_as_text(a))
    text += "\nINSTRUCTIONS FOR THIS REVISION:\n" + instructions["revision"]
    if notes:
        text += ("\n- Apply the notes from the person who asked. If a note changes the criteria, re-evaluate "
                 "every item against the new criteria and say in the changes field what the note made you change.")
    text += "\n\nProduce your revised answer in the format the original prompt requires."
    return text


def judge_prompt(prompt: str, finals: List[Tuple[Participant, Answer]], rows: List[MatrixRow],
                 rounds_done: int, instructions: Dict[str, str], notes: Optional[Dict[str, str]] = None) -> str:
    n = len(finals)
    text = (f"JUDGE PASS. {n} models answered the prompt and revised over {rounds_done} round(s). "
            f"Their final answers and an agreement summary are below.\n")
    text += _block("ORIGINAL PROMPT", prompt)
    text += notes_block(notes or {})
    for p, a in finals:
        text += _block(f"FINAL ANSWER FROM {p.label}", answer_as_text(a))
    labels = {p.key: p.label for p, _ in finals}
    attrs = attribute_names({p.key: a for p, a in finals})
    cons = [r for r in rows if r.is_consensus]
    argued = [r for r in rows if not r.is_consensus]
    summary = [f"Consensus items (held by a majority of the {n}):"]
    for r in cons:
        line = f"- {r.key} ({r.label}): held by {r.count}/{n}"
        diffs = r.argued_attrs(attrs)
        if diffs:
            parts = []
            for a in diffs:
                parts.append(a + ": " + ", ".join(f"{labels[k]} {r.text(k, a)}" for k in r.holders()))
            line += ". VALUES STILL DISPUTED, rule on them: " + "; ".join(parts)
        summary.append(line)
    if not cons:
        summary.append("- none")
    summary += ["", "Argued items (held by a minority):"]
    summary += [f"- {r.key} ({r.label}): held only by " + ", ".join(labels[k] for k in r.holders())
                for r in argued] or ["- none"]
    order = same_order(rows, [p.key for p, _ in finals])
    if order is not None:
        summary += ["", "Ranking of shared items: " + ("identical across models" if order else
                    "DIFFERS between models, decide the order")]
    text += _block("AGREEMENT SUMMARY", "\n".join(summary))
    text += "\nINSTRUCTIONS FOR THE JUDGE:\n" + instructions["judge"]
    if notes:
        text += "\n- Apply the notes from the person who asked; they amend the original prompt."
    text += "\n\nProduce the final answer in the format the original prompt requires."
    return text


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    index: int                              # 0 = independent answers
    answers: Dict[str, Answer]
    matrix: List[MatrixRow]
    attributes: List[str]                   # attribute display names seen this round
    full: float                             # fraction of items held by everyone
    majority: float                         # fraction of items held by a majority
    seconds: float
    values_argued: int = 0                  # items held by everyone whose numbers still differ
    order_agreed: Optional[bool] = None     # same ranking of shared items (None: no rank attribute)
    merges: List[str] = field(default_factory=list)   # synonym merges the program applied this round

    @property
    def settled(self) -> bool:
        """Full consensus: everyone holds the same items, with identical numbers and the same order."""
        return self.full >= 1.0 and self.values_argued == 0 and self.order_agreed is not False


@dataclass
class JudgeResult:
    key: str
    label: str
    answer: Answer
    seconds: float


@dataclass
class RunResult:
    out_dir: str
    prompt: str
    inputs: Dict[str, str]
    participants: Dict[str, str]            # key -> label
    rounds: List[RoundResult] = field(default_factory=list)
    judge: Optional[JudgeResult] = None
    plan: Optional[Plan] = None
    notes: Dict[str, str] = field(default_factory=dict)   # "1", "2", ... or "judge" -> the user's note
    stopped_early: bool = False
    stopped_by_user: bool = False
    cancelled: bool = False

    @property
    def final(self) -> Optional[RoundResult]:
        return self.rounds[-1] if self.rounds else None


@dataclass
class GateDecision:
    """What to do after a round: continue to the next step, skip straight to the judge, or stop."""
    action: str = "continue"          # "continue" | "judge" | "stop"
    note: str = ""                    # the user's note for the next step, if any


# gate(round_just_finished, next_step) -> GateDecision; next_step is "round N" or "judge"
GateFn = Callable[[RoundResult, str], GateDecision]


def make_round(index: int, answers: Dict[str, Answer], seconds: float, text_scored: Optional[set] = None,
               merges: Optional[List[str]] = None) -> RoundResult:
    rows = build_matrix(answers)
    for r in rows:
        r.text_scored = text_scored or set()
    attrs = attribute_names(answers)
    rr = RoundResult(index, answers, rows, attrs, full_agreement(rows), majority_agreement(rows), seconds)
    rr.merges = list(merges or [])
    rr.values_argued = sum(1 for r in rows if r.status() == "all" and r.argued_attrs(attrs))
    rr.order_agreed = same_order(rows, list(answers))
    return rr


def round_summary(rr: RoundResult) -> str:
    n_all = sum(1 for r in rr.matrix if r.status() == "all")
    n_maj = sum(1 for r in rr.matrix if r.is_consensus)
    s = f"{n_all} of {len(rr.matrix)} items held by everyone, {n_maj} by a majority"
    if n_all:
        s += f"; values still differ on {rr.values_argued} of the {n_all}"
    if rr.order_agreed is not None:
        s += "; same order" if rr.order_agreed else "; order differs"
    return s


def _write(path: str, text: str) -> None:
    """Atomic write: the file is either the old version or the complete new one, never truncated."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # survive the folder being deleted mid-run
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def run_in_thread(fn: Callable[[], Any], cancelled: Optional[threading.Event], poll: float = 0.5) -> Any:
    """Run a blocking call in a helper thread so a cancel event can interrupt the wait."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        t.join(poll)
        if cancelled is not None and cancelled.is_set():
            raise Cancelled()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_consensus(
    prompt: str,
    inputs: Dict[str, str],
    participants: List[Participant],
    rounds: int = 1,
    stop_early: bool = True,
    judge_key: Optional[str] = None,
    planner: Optional[Participant] = None,
    merge_synonyms: bool = False,
    merger: Optional[Participant] = None,
    instructions: Optional[Dict[str, str]] = None,
    out_dir: Optional[str] = None,
    procs: Optional[ProcessRegistry] = None,
    log: Callable[[str], None] = print,
    on_round: Optional[Callable[[RoundResult], None]] = None,
    on_judge: Optional[Callable[[JudgeResult], None]] = None,
    gate: Optional[GateFn] = None,
) -> RunResult:
    """
    rounds     = number of revision rounds after the independent first answers.
    stop_early = stop revising as soon as every participant holds exactly the same items.
    judge_key  = key of the participant that rules at the end, or None for no judge.
    planner    = a participant that first decides the answer format (items, attributes, units)
                 and has it appended to the prompt; None to send the prompt as written.
    merge_synonyms = after each round, use the planner to merge differently-named items that mean
                 the same thing (one short extra call per round; off by default).
    gate       = called after each round, before the next step, so the user can add a note
                 (injected into every later prompt), skip to the judge, or stop.
    """
    procs = procs or ProcessRegistry()
    instructions = instructions or load_instructions()
    system_text = instructions["base"]
    stamp = datetime.now()
    out_dir = out_dir or os.path.join(RESULTS_DIR, stamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    labels = {p.key: p.label for p in participants}
    result = RunResult(out_dir=out_dir, prompt=prompt, inputs=inputs, participants=labels)
    by_key = {p.key: p for p in participants}
    if judge_key and judge_key not in by_key:
        raise ValueError(f"Judge '{judge_key}' is not one of the participants")

    if planner is not None:
        try:
            log(f"Planning the answer format with {planner.label}...")
            t0 = time.time()
            result.plan, block = plan_format(prompt, planner, instructions, procs)
            prompt = f"{prompt}\n\n{block}"
            result.prompt = prompt
            _write(os.path.join(out_dir, "plan.md"), f"# Answer format plan ({planner.label})\n\n{block}\n")
            log(f"Planned in {time.time() - t0:.0f}s: {plan_summary(result.plan)}")
        except Cancelled:
            result.cancelled = True
            log("Run cancelled.")
            return result
        except Exception as e:  # noqa: BLE001 - no plan is better than no run
            log(f"Planning failed ({str(e).strip().splitlines()[0][:200]}); continuing with the prompt as written.")
            result.plan = None
    _write(os.path.join(out_dir, "prompt.txt"), prompt)

    EMPTY_RETRY_NOTE = (
        "\n\nNOTE FROM THE PROGRAM: your previous attempt returned no items. An empty answer cannot be compared "
        "with the other models' answers. If a tool such as web search is unavailable, answer anyway from your "
        "own knowledge, list the items the prompt asks for, and state clearly in the unverified field which "
        "figures you could not check."
    )

    def failed_answer(label: str, error: str) -> Answer:
        return Answer(report_markdown=f"{label} did not answer this round. Error: {error}", items=[],
                      unverified=f"no answer: {error}", changes="n/a")

    def ask_all(texts: Dict[str, str]) -> Dict[str, Answer]:
        """
        Ask every participant in parallel. A participant that errors or returns no items is asked once
        more; if it still fails, it sits out the round with an empty answer so the others can continue.
        Only when every participant fails does the round fail.
        """
        from concurrent.futures import as_completed

        started = time.time()
        answers: Dict[str, Answer] = {}
        errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(participants)) as pool:
            futures = {pool.submit(p.ask, texts[p.key], system_text, procs): p for p in participants}
            for f in as_completed(futures):
                p = futures[f]
                try:
                    answers[p.key] = f.result()
                    status = "answered"
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - one model's failure must not sink the round
                    errors[p.key] = str(e).strip().splitlines()[0][:300]
                    answers[p.key] = failed_answer(p.label, errors[p.key])
                    status = f"FAILED ({errors[p.key][:120]})"
                waiting = [q.label for q in participants if q.key not in answers]
                log(f"{p.label} {status} after {(time.time() - started) / 60:.1f} min"
                    + (f"; still waiting for {', '.join(waiting)}" if waiting else ""))
        answers = {p.key: answers[p.key] for p in participants}   # keep participant order
        # One more chance for anyone who failed or returned nothing, then the round proceeds without them.
        retry_list = [p for p in participants if not answers[p.key].items]
        if retry_list:
            log("Asking again once: " + ", ".join(p.label for p in retry_list))
            with ThreadPoolExecutor(max_workers=len(retry_list)) as pool:
                futures = {p.key: pool.submit(p.ask, texts[p.key] + EMPTY_RETRY_NOTE, system_text, procs) for p in retry_list}
                for k, f in futures.items():
                    try:
                        retry = f.result()
                    except Cancelled:
                        raise
                    except Exception as e:  # noqa: BLE001
                        errors[k] = str(e).strip().splitlines()[0][:300]
                        retry = failed_answer(labels[k], errors[k])
                    if retry.items:
                        answers[k] = retry
                        errors.pop(k, None)
                    else:
                        answers[k] = retry
                        log(f"{labels[k]} sits out this round: " + (errors.get(k) or "no items returned"))
        if all(not a.items for a in answers.values()):
            raise RuntimeError("Every participant failed this round: " + "; ".join(f"{labels[k]}: {v}" for k, v in errors.items()))
        return answers

    text_scored = scored_text_attrs(result.plan)

    def record(index: int, answers: Dict[str, Answer], started: float) -> RoundResult:
        merges: List[str] = []
        quick = merger or planner
        if merge_synonyms and quick is not None:
            try:
                merges = reconcile_keys(answers, labels, quick, instructions, procs)
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001 - a failed merge must not lose the round
                log(f"Synonym merge skipped: {e}")
            if merges:
                log("Merged names that mean the same thing: " + "; ".join(merges))
        rr = make_round(index, answers, time.time() - started, text_scored, merges)
        result.rounds.append(rr)
        for key, ans in answers.items():
            _write(os.path.join(out_dir, f"round{index}_{key}.md"), answer_as_text(ans))
        log(f"Finished round {index} in {rr.seconds/60:.1f} min: {round_summary(rr)}.")
        if on_round:
            on_round(rr)
        return rr

    def ask_gate(rr: RoundResult, next_step: str) -> GateDecision:
        """Give the user a chance to comment before the next step; record the note if any."""
        if gate is None:
            return GateDecision()
        decision = gate(rr, next_step)
        if procs.cancelled.is_set():
            raise Cancelled()
        if decision.note.strip():
            key = "judge" if next_step == "judge" else next_step.split()[-1]
            result.notes[key] = decision.note.strip()
            _write(os.path.join(out_dir, "notes.md"), "# Your notes between rounds\n\n" + "\n".join(
                f"- Before {'the judge pass' if k == 'judge' else 'round ' + k}: {v}" for k, v in result.notes.items()) + "\n")
            log(f"Your note will be sent with the next step: {decision.note.strip()[:200]}")
        return decision

    try:
        log(f"Starting round 0: {', '.join(labels.values())} answering independently...")
        t0 = time.time()
        current = ask_all({p.key: prompt for p in participants})
        rr = record(0, current, t0)

        go_to_judge = False
        for i in range(1, rounds + 1):
            decision = ask_gate(rr, f"round {i}")
            if decision.action == "stop":
                log("Stopped by you. Finished rounds are kept.")
                result.stopped_by_user = True
                break
            if decision.action == "judge":
                log("Skipping the remaining revision rounds at your request.")
                go_to_judge = True
                break
            if stop_early and rr.settled and not decision.note.strip():
                log("Everyone holds the same items with the same numbers and order, stopping early.")
                result.stopped_early = True
                break
            log(f"Starting round {i} of {rounds}: each model reviews and argues the others' answers...")
            t0 = time.time()
            texts = {
                p.key: revision_prompt(prompt, p, [(o, current[o.key]) for o in participants if o.key != p.key],
                                       current[p.key], i, rounds, instructions, result.notes)
                for p in participants
            }
            current = ask_all(texts)
            rr = record(i, current, t0)

        if judge_key and not result.stopped_by_user:
            if not go_to_judge:
                decision = ask_gate(rr, "judge")
                if decision.action == "stop":
                    log("Stopped by you before the judge pass.")
                    result.stopped_by_user = True
            if not result.stopped_by_user:
                judge = by_key[judge_key]
                log(f"Starting judge pass: {judge.label} is ruling on the argued items...")
                t0 = time.time()
                text = judge_prompt(prompt, [(p, current[p.key]) for p in participants], rr.matrix,
                                    len(result.rounds) - 1, instructions, result.notes)
                try:
                    ans = judge.ask(text, system_text, procs)
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - keep the rounds even if the ruling fails
                    log(f"Judge pass failed ({str(e).strip().splitlines()[0][:200]}); the final round stands as the result.")
                    ans = None
                if ans is not None:
                    result.judge = JudgeResult(judge.key, judge.label, ans, time.time() - t0)
                    _write(os.path.join(out_dir, "judge.md"), answer_as_text(ans))
                    log(f"Finished judge pass in {result.judge.seconds/60:.1f} min: {len(ans.items)} items in the final answer.")
                    if on_judge:
                        on_judge(result.judge)
    except Cancelled:
        result.cancelled = True
        log("Run cancelled.")
    finally:
        if result.rounds:
            save_summary(result)
    return result


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def judge_lookup(judge: Optional[Answer]) -> Dict[str, AnswerItem]:
    """Judge items by normalized key and by normalized label."""
    out: Dict[str, AnswerItem] = {}
    if judge:
        for it in judge.items:
            for a in (normalize_key(it.key), normalize_key(it.label)):
                if a:
                    out.setdefault(a, it)
    return out


def judge_item(jmap: Dict[str, AnswerItem], row: MatrixRow) -> Optional[AnswerItem]:
    return jmap.get(normalize_key(row.key)) or jmap.get(normalize_key(row.label))


def item_attr(it: AnswerItem, attr: str) -> Optional[Attribute]:
    n = norm_attr(attr)
    return next((a for a in it.attributes if norm_attr(a.name) == n), None)


def matrix_markdown(rows: List[MatrixRow], labels: Dict[str, str], attr: Optional[str],
                    judge: Optional[Answer] = None) -> str:
    keys = list(labels)
    headers = ["Key", "Label"] + [labels[k] for k in keys] + ["Held"] + (["Average"] if attr else [])
    jmap = judge_lookup(judge)
    if judge:
        headers.append("Judge")
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = [r.key, r.label.replace("|", "/")]
        argued_here = attr is not None and r.attr_agreement(attr) is False
        for k in keys:
            cell = ("" if not r.held[k] else (r.text(k, attr) or "held")) if attr else ("held" if r.held[k] else "")
            cells.append(cell + (" (differs)" if argued_here and cell else ""))
        cells.append(f"{r.count}/{len(keys)}")
        if attr:
            cells.append(_num(r.average(attr)))
        if judge:
            j = judge_item(jmap, r)
            if not j:
                cells.append("excluded")
            else:
                a = item_attr(j, attr) if attr else None
                cells.append("included" + (f" {a.value}" if a else ""))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _rank_of(a: Answer, key: str) -> Optional[float]:
    for it in a.items:
        if normalize_key(it.key) == normalize_key(key):
            at = item_attr(it, "rank")
            return at.number if at else None
    return None


def _keys_of(a: Answer) -> Dict[str, str]:
    """normalized key -> display key, for one answer."""
    return {normalize_key(it.key): it.key for it in a.items if normalize_key(it.key)}


def _ordered_keys(a: Answer, limit: int = 0) -> List[str]:
    items = list(a.items)
    if any(item_attr(it, "rank") and item_attr(it, "rank").number is not None for it in items):
        items.sort(key=lambda it: (item_attr(it, "rank").number if item_attr(it, "rank") and item_attr(it, "rank").number is not None else 1e9))
    keys = [it.key for it in items]
    return keys[:limit] if limit else keys


def _clip(text: str, n: int = 700) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3] + "..."


def narrative(result: RunResult) -> str:
    """A plain-language story of the run, built from the data: who said what, who moved, who held out, the ruling."""
    labels = result.participants
    keys = list(labels)
    n = len(keys)
    out: List[str] = [f"# The story of this run", ""]
    if result.inputs:
        out.append("Inputs: " + ", ".join(f"{k} = {v}" for k, v in result.inputs.items()) + "  ")
    out.append(f"Participants: {', '.join(labels.values())}.  ")
    if result.plan:
        out.append(f"Format planned: {plan_summary(result.plan)}.")
    out.append("")

    prev: Optional[RoundResult] = None
    for rr in result.rounds:
        out.append(f"## Round {rr.index}: " + ("independent answers" if rr.index == 0 else "after arguing"))
        out.append("")
        note = result.notes.get(str(rr.index))
        if note:
            out.append(f"Your note before this round: \"{note}\"")
            out.append("")
        for k in keys:
            a = rr.answers[k]
            if not a.items:
                out.append(f"- **{labels[k]}** gave no items. {_clip(a.report_markdown, 200)}")
                continue
            ordered = _ordered_keys(a)
            shown = ", ".join(ordered[:8]) + (f", and {len(ordered) - 8} more" if len(ordered) > 8 else "")
            if prev is None:
                out.append(f"- **{labels[k]}** listed {len(a.items)}: {shown}.")
            else:
                before = _keys_of(prev.answers[k])
                now = _keys_of(a)
                added = [now[x] for x in now if x not in before]
                dropped = [before[x] for x in before if x not in now]
                moved = []
                for x in now:
                    if x in before:
                        r0, r1 = _rank_of(prev.answers[k], now[x]), _rank_of(a, now[x])
                        if r0 is not None and r1 is not None and r0 != r1:
                            moved.append(f"{now[x]} {int(r0)}→{int(r1)}")
                bits = []
                if added:
                    bits.append("added " + ", ".join(added))
                if dropped:
                    bits.append("dropped " + ", ".join(dropped))
                if moved:
                    bits.append("re-ranked " + ", ".join(moved[:6]) + (" ..." if len(moved) > 6 else ""))
                if not bits:
                    bits.append("kept its list unchanged")
                out.append(f"- **{labels[k]}** {'; '.join(bits)}. Now {len(a.items)}: {shown}.")
                if a.changes and a.changes.strip().lower() != "n/a":
                    out.append(f"  - In its words: {_clip(a.changes)}")
        held_all = [r for r in rr.matrix if r.status() == "all"]
        argued = [r for r in rr.matrix if not r.is_consensus]
        majority = [r for r in rr.matrix if r.status() == "majority"]
        out.append("")
        out.append(f"After round {rr.index}: {len(held_all)} held by everyone"
                   + (f" ({', '.join(r.key for r in held_all[:10])})" if held_all else "")
                   + (f"; {len(majority)} held by a majority ({', '.join(r.key for r in majority[:10])})" if majority else "")
                   + (f"; {len(argued)} argued: " + ", ".join(f"{r.key} ({'/'.join(labels[k] for k in r.holders())})" for r in argued[:10]) if argued else "; nothing argued")
                   + ".")
        if rr.order_agreed is not None:
            top = []
            for k in keys:
                o = _ordered_keys(rr.answers[k], 3)
                if o:
                    top.append(f"{labels[k]}: {' > '.join(o)}")
            out.append(("Same order for shared items. " if rr.order_agreed else "Orders differ. ") + "Top 3 each: " + "; ".join(top) + ".")
        out.append("")
        prev = rr

    if result.stopped_early:
        out.append("Everyone held the same items, so the remaining rounds were skipped.")
        out.append("")
    if result.judge:
        j = result.judge
        final = result.final
        out.append(f"## The judge ({j.label})")
        out.append("")
        if result.notes.get("judge"):
            out.append(f"Your note before the ruling: \"{result.notes['judge']}\"")
            out.append("")
        jmap = judge_lookup(j.answer)
        kept, cut = [], []
        for r in final.matrix:
            (kept if judge_item(jmap, r) else cut).append(r)
        extra = [it for it in j.answer.items if not any(judge_item(jmap, r) is it for r in final.matrix)]
        out.append(f"The judge read all {n} final answers and the agreement summary, kept {len(kept)} of the "
                   f"{len(final.matrix)} items on the table, dropped {len(cut)}"
                   + (f", and added {len(extra)} of its own" if extra else "") + ".")
        argued_rows = [r for r in final.matrix if not r.is_consensus]
        if argued_rows:
            out.append("")
            out.append("Rulings on the argued items:")
            for r in argued_rows:
                ji = judge_item(jmap, r)
                holders = "/".join(labels[k] for k in r.holders())
                if ji:
                    out.append(f"- {r.key} (held by {holders}): included. {_clip(ji.rationale, 300)}")
                else:
                    out.append(f"- {r.key} (held by {holders}): excluded.")
        dropped_cons = [r for r in cut if r.is_consensus]
        if dropped_cons:
            out.append("")
            out.append("Consensus items the judge removed: " + ", ".join(r.key for r in dropped_cons) + ".")
        if j.answer.changes and j.answer.changes.strip().lower() != "n/a":
            out.append("")
            out.append("In the judge's words: " + _clip(j.answer.changes, 1200))
        out.append("")
        out.append("## Final ranking")
        out.append("")
        for i, it in enumerate(_ordered_keys(j.answer), start=1):
            item = next(x for x in j.answer.items if x.key == it)
            attrs = "; ".join(f"{a.name}: {a.value}" for a in item.attributes if norm_attr(a.name) != "rank")
            out.append(f"{i}. **{item.key}**" + (f" ({attrs})" if attrs else "") + f" - {_clip(item.rationale, 200)}")
    else:
        out.append("No judge pass was run; the final round above is the end state.")
    return "\n".join(out)


def save_summary(result: RunResult) -> None:
    _write(os.path.join(result.out_dir, "story.md"), narrative(result))
    final = result.final
    assert final is not None
    labels = result.participants
    keys = list(labels)
    n = len(keys)
    cons = [r for r in final.matrix if r.is_consensus]
    argued = [r for r in final.matrix if not r.is_consensus]
    judge_ans = result.judge.answer if result.judge else None
    attrs = final.attributes

    md = ["# Consensus run", "",
          f"Folder: {result.out_dir}  ",
          f"Inputs: {json.dumps(result.inputs)}  ",
          f"Participants: {', '.join(labels.values())}  ",
          f"Revision rounds completed: {len(result.rounds) - 1}"
          + (" (stopped early: everyone agreed)" if result.stopped_early else "")
          + (" (stopped by you)" if result.stopped_by_user else "")
          + (" (CANCELLED)" if result.cancelled else ""),
          f"Judge: {result.judge.label if result.judge else 'none'}",
          ""]
    if result.notes:
        md += ["## Your notes between rounds", ""]
        md += [f"- Before {'the judge pass' if k == 'judge' else 'round ' + k}: {v}" for k, v in result.notes.items()]
        md.append("")
    md += ["## Agreement by round", ""]
    for rr in result.rounds:
        md.append(f"- Round {rr.index}: {round_summary(rr)}")
    md += ["", f"## Final matrix (round {final.index})", ""]
    if attrs:
        for a in attrs:
            md += [f"### {a}", "", matrix_markdown(final.matrix, labels, a, judge_ans), ""]
    else:
        md += [matrix_markdown(final.matrix, labels, None, judge_ans), ""]

    md += [f"## Consensus ({len(cons)}): held by a majority of the {n}", ""]
    for r in cons:
        line = f"- **{r.key}** ({r.label}), held by {r.count}/{n}"
        diffs = r.argued_attrs(attrs)
        if diffs:
            line += ". Values still differ on " + "; ".join(
                a + " (" + ", ".join(f"{labels[k]} {r.text(k, a)}" for k in r.holders()) + ")" for a in diffs)
        md.append(line)
    if not cons:
        md.append("_none_")
    if final.order_agreed is not None:
        md += ["", "Ranking of shared items: " + ("same across models." if final.order_agreed else "differs between models.")]
    md += ["", f"## Argued ({len(argued)}): held by a minority", ""]
    for r in argued:
        holders = [labels[k] for k in keys if r.held[k]]
        md.append(f"- **{r.key}** ({r.label}): held by {', '.join(holders)}")
        for k in keys:
            if r.held[k]:
                vals = "; ".join(f"{a}: {r.text(k, a)}" for a in attrs if r.text(k, a))
                md.append(f"  - {labels[k]}: {vals + '. ' if vals else ''}{r.rationales[k]}")
    if not argued:
        md.append("_none_")

    if result.judge:
        md += ["", f"## Judge ruling ({result.judge.label})", "", "Final list:", ""]
        for it in judge_ans.items:
            md.append(item_as_text(it))
        md += ["", "Rulings:", "", judge_ans.changes, ""]

    md += ["", "## Change notes by round", ""]
    for rr in result.rounds[1:]:
        for k in keys:
            md += [f"### Round {rr.index}, {labels[k]}", "", rr.answers[k].changes, ""]
    md += ["Full reports: round<N>_<participant>.md and judge.md in this folder."]
    _write(os.path.join(result.out_dir, "consolidated.md"), "\n".join(md))

    jmap = judge_lookup(judge_ans)
    with open(os.path.join(result.out_dir, "final_table.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        head = ["key", "label", "status", "held_count", "values_differ_on"]
        for k in keys:
            head += [f"{labels[k]}: held"] + [f"{labels[k]}: {a}" for a in attrs]
        if judge_ans:
            head += ["judge: included"] + [f"judge: {a}" for a in attrs]
        w.writerow(head)
        for r in final.matrix:
            row = [r.key, r.label, r.status(), r.count, "; ".join(r.argued_attrs(attrs))]
            for k in keys:
                row += ["yes" if r.held[k] else "no"] + [r.text(k, a) for a in attrs]
            if judge_ans:
                j = judge_item(jmap, r)
                row += ["yes" if j else "no"] + [(item_attr(j, a).value if j and item_attr(j, a) else "") for a in attrs]
            w.writerow(row)

    data = {
        "app": APP_NAME, "format": 3,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_dir": result.out_dir,
        "inputs": result.inputs,
        "participants": labels,
        "summary": {   # read by the History tab so it need not parse every answer
            "revisions": max(0, len(result.rounds) - 1),
            "full": final.full, "majority": final.majority,
            "participants": list(labels.values()),
            "judge": result.judge.label if result.judge else "",
            "cancelled": result.cancelled,
        },
        "stopped_early": result.stopped_early,
        "stopped_by_user": result.stopped_by_user,
        "cancelled": result.cancelled,
        "notes": result.notes,
        "plan": result.plan.model_dump() if result.plan else None,
        "rounds": [
            {"index": rr.index, "seconds": rr.seconds, "merges": rr.merges,
             "answers": {k: a.model_dump() for k, a in rr.answers.items()}}
            for rr in result.rounds
        ],
        "judge": ({"key": result.judge.key, "label": result.judge.label, "seconds": result.judge.seconds,
                   "answer": result.judge.answer.model_dump()} if result.judge else None),
    }
    _write(os.path.join(result.out_dir, "run.json"), json.dumps(data, indent=2))


def _answer_from_any(d: dict) -> Answer:
    """Accept the current Answer shape and the two earlier shapes (single value / stock picks)."""
    items: List[AnswerItem] = []
    if "items" in d:
        for it in d["items"]:
            if "attributes" in it:
                items.append(AnswerItem.model_validate(it))
            else:
                v = it.get("value")
                attrs = [Attribute(name="value", value=_num(v), number=v)] if v is not None else []
                items.append(AnswerItem(key=it.get("key", ""), label=it.get("label", ""), attributes=attrs,
                                        rationale=it.get("rationale", "")))
    else:
        for p in d.get("picks", []):
            v = p.get("allocation_usd")
            attrs = [Attribute(name="amount", value=f"${v:,.0f}", number=v)] if v is not None else []
            items.append(AnswerItem(key=p.get("ticker", ""), label=p.get("company", ""), attributes=attrs,
                                    rationale=p.get("thesis", "")))
    return Answer(report_markdown=d.get("report_markdown", ""), items=items,
                  unverified=d.get("unverified", "n/a"), changes=d.get("changes", "n/a"))


def _upgrade_legacy(data: dict) -> dict:
    """Convert run.json files written by earlier versions into the current shape."""
    if "stage1" in data and "rounds" not in data:
        labels = data.get("models", {})
        rounds = []
        for i, stage in enumerate(("stage1", "stage2")):
            if stage in data:
                rounds.append({"index": i, "seconds": 0.0, "answers": data[stage]})
        data = {"inputs": data.get("inputs", {}), "stopped_early": False, "cancelled": False,
                "participants": {k: labels.get(k, k) for k in rounds[0]["answers"]} if rounds else {},
                "rounds": rounds, "judge": None}
    for rd in data.get("rounds", []):
        if "reports" in rd and "answers" not in rd:
            rd["answers"] = rd.pop("reports")
    data.setdefault("judge", None)
    return data


def load_run(out_dir: str) -> RunResult:
    with open(os.path.join(out_dir, "run.json"), encoding="utf-8") as f:
        data = _upgrade_legacy(json.load(f))
    prompt = ""
    p = os.path.join(out_dir, "prompt.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            prompt = f.read()
    result = RunResult(out_dir=out_dir, prompt=prompt, inputs=data.get("inputs", {}),
                       participants=data["participants"],
                       stopped_early=data.get("stopped_early", False), cancelled=data.get("cancelled", False),
                       stopped_by_user=data.get("stopped_by_user", False), notes=dict(data.get("notes") or {}))
    if data.get("plan"):
        try:
            result.plan = Plan.model_validate(data["plan"])
        except Exception:
            result.plan = None
    text_scored = scored_text_attrs(result.plan)
    for rd in data["rounds"]:
        answers = {k: _answer_from_any(v) for k, v in rd["answers"].items()}
        result.rounds.append(make_round(rd["index"], answers, rd.get("seconds", 0.0), text_scored, rd.get("merges")))
    j = data.get("judge")
    if j:
        result.judge = JudgeResult(j["key"], j["label"], _answer_from_any(j["answer"]), j.get("seconds", 0.0))
    return result


def run_summary(out_dir: str) -> dict:
    """Cheap summary of a saved run for the history list: the stored summary when present, else a full load."""
    try:
        with open(os.path.join(out_dir, "run.json"), encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary")
        if isinstance(s, dict) and "revisions" in s:
            return {**s, "inputs": data.get("inputs", {})}
    except (OSError, ValueError):
        pass
    r = load_run(out_dir)
    f = r.final
    return {
        "revisions": max(0, len(r.rounds) - 1),
        "full": f.full if f else 0.0,
        "majority": f.majority if f else 0.0,
        "participants": list(r.participants.values()),
        "judge": r.judge.label if r.judge else "",
        "inputs": r.inputs,
        "cancelled": r.cancelled,
    }


# ---------------------------------------------------------------------------
# Global configuration (.env next to the app)
# ---------------------------------------------------------------------------

ENV_PATH = os.path.join(APP_DIR, ".env")

def read_env_file() -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, v = s.split("=", 1)
                    values[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def write_env_file(updates: Dict[str, str]) -> None:
    """Set or remove keys in .env, keeping every other line as it is, and apply them to this process."""
    global CLI_TIMEOUT_SECONDS
    lines: List[str] = []
    try:
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        pass
    done = set()
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                done.add(k)
                if updates[k].strip():
                    out.append(f"{k}={updates[k].strip()}")
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in done and v.strip():
            out.append(f"{k}={v.strip()}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip("\n") + "\n")
    for k, v in updates.items():
        if v.strip():
            os.environ[k] = v.strip()
        else:
            os.environ.pop(k, None)
    try:
        CLI_TIMEOUT_SECONDS = int(os.getenv("CLI_TIMEOUT_SECONDS", "2400"))
    except ValueError:
        CLI_TIMEOUT_SECONDS = 2400


def participant_status(key: str, via: str = "cli") -> str:
    """One line on whether a participant looks usable for the chosen route, without calling it."""
    if key == "gemini" or via == "api":
        env = {"claude": "ANTHROPIC_API_KEY", "codex": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[key]
        return f"API: {env} set" if os.getenv(env, "").strip() else f"API: {env} not set (enter it below)"
    tool = {"claude": "claude", "codex": "codex"}[key]
    return f"CLI: {tool} tool found" if shutil.which(tool) else f"CLI: {tool} tool not found on PATH"


def test_participant(key: str, model: str = "", via: str = "cli") -> str:
    """Make one tiny call (no web search) and return 'OK ...' or the error text."""
    schema = {"title": "Ping", "type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    t0 = time.time()
    if key == "claude" and via == "cli" and not model:
        model = "sonnet"
    p = make_participant(key, via, model, quick=True)
    data = p.ask_json('Reply with {"ok": true}.', "You reply with JSON matching the schema.", schema, ProcessRegistry())
    if not isinstance(data, dict) or data.get("ok") is not True:
        raise RuntimeError(f"unexpected reply: {data}")
    return f"OK ({p.label}, {time.time() - t0:.0f}s)"


def list_runs(include_incomplete: bool = False) -> List[str]:
    """Run folders, newest first. Incomplete ones (no round finished, so no run.json) only if asked."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs = []
    for d in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "run.json")) or include_incomplete:
            runs.append(d)
    return sorted(runs, reverse=True)


def run_is_complete(out_dir: str) -> bool:
    return os.path.isfile(os.path.join(out_dir, "run.json"))
