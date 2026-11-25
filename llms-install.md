# Installing oce-math-mcp: guide for AI agents

This file is for an AI agent (Claude Code, Cline, Cursor and similar) that installs this MCP server for a
user. Follow the steps in order. Every step has a check: do not go to the next step until the check passes.
The human guide with the same steps is the Installation section of `README.md`.

## Before you start: ask the user

Ask these questions once, before any change:

1. Where to put the repo. Use a local folder, not a network drive (decision D26 in `docs/DECISIONS.md`).
2. Whether you may install `uv` if it is missing (step 1). It installs into the user's profile.
3. Whether they want Lean 4 for `lean_check` (step 6). It downloads about 10 GB and is optional.
4. Which MCP client to register the server in (step 5).

## Step 1: tools

Check:

```
git --version
uv --version
```

Both must print a version. If `uv` is missing and the user agreed, install it with the official installer.

Windows (PowerShell):

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS and Linux:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new shell (the installer changes `PATH`) and run `uv --version` again. Python is not a
separate requirement: if Python 3.12 or newer is missing, `uv` downloads it in step 3.

## Step 2: clone

```
git clone https://github.com/oce-projects/oce-math-mcp.git
```

Check: the folder `oce-math-mcp` contains `pyproject.toml` and `uv.lock`. Remember its absolute path: the
next steps call it `<REPO>`.

## Step 3: install the Python environment

```
uv sync --compile-bytecode --directory <REPO>
```

Check: the command ends without an error. On a network drive, set `UV_PROJECT_ENVIRONMENT` to a local
folder first (README, step 2).

## Step 4: self-check

```
uv run --directory <REPO> oce-math-mcp --self-check
```

Check: the exit code is 0 and the JSON report has `"ok": true` and `"tools": 10`. The command starts the
server over stdio exactly as an MCP client does, so a pass here means the server works.

If `"ok"` is false, read `checks[].detail` and `error` in the report, and do not register the server.
The field `lean` is information only and never makes the check fail.

## Step 5: register the server

Use the absolute `<REPO>` path.

Claude Code:

```
claude mcp add --scope user oce-math -- uv run --directory <REPO> oce-math-mcp
```

Check: `claude mcp list` shows `oce-math`.

Clients that read an `mcpServers` JSON file (Claude Desktop, Cline, Cursor and others): add this entry
to the existing file and keep the other servers. On Windows write `<REPO>` with forward slashes
(`C:/...`) or with every backslash written twice: JSON reads a single backslash as an escape.

```json
{
  "mcpServers": {
    "oce-math": {
      "command": "uv",
      "args": ["run", "--directory", "<REPO>", "oce-math-mcp"]
    }
  }
}
```

Claude Desktop reads `%APPDATA%\Claude\claude_desktop_config.json` on Windows and
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS. Tell the user to restart the
client.

If the client cannot start the server while step 4 passes, the client does not see `uv` on its `PATH`:
put the full path of `uv` in `command` (`where.exe uv` on Windows, `which uv` on macOS and Linux).

## Step 6 (optional): Lean 4 and Mathlib

Only if the user agreed. It downloads elan, the Lean toolchain and Mathlib (about 10 GB):

```
uv run --directory <REPO> oce-math-mcp --install-lean
```

Check: step 4 again reports `"lean": "installed"`.

The first `lean_check` after a restart can answer `TIMEOUT`: Lean loads Mathlib from disk the first time.
Run the same check again; the next ones take about 20 s.

## Do not

- Do not start the HTTP transport with `--host 0.0.0.0`: it has no authentication.
- Do not use `--no-pool` for normal use: without the computing processes the 5 s timeout cannot stop a
  computation.
- Do not edit files inside the repo to make a check pass. Report the failing check to the user instead.
