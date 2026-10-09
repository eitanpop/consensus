# Getting started with Consensus

Consensus asks two or three AI models the same question, makes them argue about their
answers, and has one of them act as judge. You get one ranked answer plus a record of what
each model said and where they disagreed.

Before it can run, each model needs a way to be reached. This page walks through that.
Plan on 20 to 30 minutes the first time. After that, you just double-click the app.

## What you need

| Model | What it runs on | What you need |
|---|---|---|
| Claude | the `claude` command-line tool | a Claude subscription (Pro or Max) and Node.js |
| OpenAI | the `codex` command-line tool | a ChatGPT subscription (Plus or Pro) and Node.js |
| Gemini | the Gemini API | a free Gemini API key |

You need at least two of the three. All three gives majority votes instead of ties.

Also needed once: Python (the program itself is written in it).

## Step 1: Install Python

1. Go to https://www.python.org/downloads/ and download the latest Python 3.
2. Run the installer. On the first screen, tick **"Add python.exe to PATH"** before clicking
   Install. This matters: without it the app cannot be launched by double-clicking.
3. Open a terminal (press the Windows key, type `cmd`, press Enter) and type `python --version`.
   You should see a version number.

Then, in the terminal, go to the folder that holds the app (the one with `Consensus.pyw` in
it) and install the two small libraries it uses. For example, if the folder is on your Desktop
and called `consensus`:

```
cd %USERPROFILE%\Desktop\consensus
pip install -r requirements.txt
```

## Step 2: Install Node.js

The Claude and OpenAI tools are installed with Node.js, so install that first.

1. Go to https://nodejs.org and download the LTS version.
2. Run the installer with the default options.
3. In a new terminal, type `node --version`. You should see a version number.

## Step 3: Set up Claude

1. In a terminal: `npm install -g @anthropic-ai/claude-code`
2. Type `claude` and press Enter. The first time, it asks you to log in: pick the
   subscription option and finish in the browser with the account that has your Claude plan.
3. When it shows a prompt, type `/exit` to leave.

## Step 4: Set up OpenAI (Codex)

1. In a terminal: `npm install -g @openai/codex`
2. Type `codex` and press Enter. Choose **Sign in with ChatGPT** and finish in the browser
   with the account that has your ChatGPT plan.
3. Press Ctrl+C to leave.

## Step 5: Set up Gemini (optional)

Google no longer lets its command-line tool sign in with personal accounts, so Gemini runs
through an API key instead.

1. Go to https://aistudio.google.com, sign in with a Google account, and create an API key.
   The free tier works and needs no credit card.
2. Open the app, go to the **Settings** tab, paste the key into Gemini's "API key" box, and
   click **Save settings**.

The free tier has limits: Gemini answers without web search, and the newest Gemini models
are often busy, in which case the app quietly uses an older one and says so in the results.

## Step 6: Check everything

1. Double-click `Consensus.pyw` in the app's folder.
2. Go to the **Settings** tab. Each model has a **Test** button. Press each one. After a few
   seconds (Gemini can take longer) it should say "OK". If it says FAILED, the message tells
   you which step above to revisit.

## Step 7: Your first run

1. On the **Run** tab, click **Quick test (no research)** in the Saved prompts list.
2. Leave the inputs as they are and click **Run**. This prompt tells the models not to
   search the web, so the whole thing takes a few minutes.
3. Watch the Progress box. When it finishes, the **Results** tab opens: one row per item,
   one column per model, green rows are agreed, orange are argued.
4. Open the **Reports** tab and read `story.md`: a plain-language account of what each model
   said, what changed each round, and how the judge ruled.

## Everyday use

- **Ask a question**: type in the box at the top of the Run tab and press Enter.
- **Saved prompts**: for longer, reusable prompts. Write `{{NAME}}` anywhere to be asked for
  a value each time (the Wildcard guide button explains the forms).
- **Web search**: on by default. Research runs take 5 to 15 minutes per round because the
  models search and verify. For opinion or advice questions, untick Web search in Run
  settings and runs finish in about a minute.
- **Pause after each round**: lets you read the results and type a note to the models
  before they argue again.

## If something goes wrong

- **"claude tool not found"** or **"codex tool not found"** in Settings: redo step 3 or 4,
  then close and reopen the app.
- **A model fails during a run**: the run continues without it for that round and says so
  in the Progress box. Press Test in Settings afterwards to see what's wrong.
- **Gemini says quota or high demand**: the free tier is rate-limited. Put
  `gemini-flash-lite-latest` in Gemini's model box in Settings, or wait and retry.
- **A run seems stuck**: the Progress box says which model it is waiting on. Research calls
  can legitimately take 10 minutes. Stop ends the run and keeps finished rounds.
- **The window looks frozen after the PC slept**: click anywhere; it is usually just the
  screen not having refreshed.
