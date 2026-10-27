# AI-Harness real providers and Claude Code MCP

## Architecture

- `OllamaProvider` calls the local Ollama `/api/chat` endpoint using bounded,
  non-streaming requests, native JSON-schema output, and native tool-call
  parsing. It rejects non-loopback endpoints.
- `AnthropicProvider` is an optional adapter using the official Python SDK and
  `ANTHROPIC_API_KEY`; Claude Code's interactive login is not reused.
- The Anthropic adapter supports structured text generation; it explicitly
  rejects runtime tool execution. Ollama supports the runtime's validated,
  policy-gated tool-call flow.
- Provider selection is explicit through `AI_HARNESS_PROVIDER`. The default is
  `ollama`; provider or model failures never switch to simulated mode.
- `analyst` and `reviewer` are real provider-backed agents. They read a bounded
  set of UTF-8 project files from configured roots, do not invoke tools or
  commands, and return a Pydantic-validated JSON report. Facts require citations
  to evidence IDs supplied to the model. The reviewer receives dependency
  results and independently examines its own evidence selection.
- Maestro validates provider-proposed task structure, capabilities, cycles,
  dependencies, and execution limits. It persists plans, state transitions,
  agent output, and execution events in SQLite.
- `mcp_server.py` exposes Maestro through MCP stdio. Protocol output stays on
  stdout; Python logs go to stderr.

## Install

From the AI-Harness root, using the existing virtual environment when available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,mcp]"
.\.venv\Scripts\python.exe -m pytest
```

Set `AI_HARNESS_ALLOWED_ROOTS` to semicolon-separated existing directories.
Only those roots and their descendants are accepted. If unset, the harness
repository itself is the sole root. Traversal skips hidden directories,
`.git`, virtual environments, dependency/build output, credential-like files,
and non-text extensions; file count and byte limits are enforced. The
`workspace_path` tool argument cannot broaden the allowlist.

Provider settings:

| Variable | Default | Purpose |
|---|---|---|
| `AI_HARNESS_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `AI_HARNESS_MODEL` | `qwen3.8:27b` for Ollama | Explicit model name |
| `AI_HARNESS_OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `AI_HARNESS_TIMEOUT_SECONDS` | `120` | Provider request timeout |
| `AI_HARNESS_DATABASE` | Per-user application data SQLite | Optional history path |
| `AI_HARNESS_ALLOWED_ROOTS` | AI-Harness repository | Read-only project roots |

Install the optional Anthropic SDK with `.[anthropic]` and provide an API key
through the environment or a secret manager. Never put credentials in the
repository or MCP server arguments.

## MCP tools

- `harness_status`: current provider/model health, actual registered agents,
  allowlisted roots, history location, and limitations.
- `harness_list_agents`: implemented analyst/reviewer identity and capabilities.
- `harness_create_plan`: provider-backed structured plan saved awaiting approval.
- `harness_run_agent`: requested analyst/reviewer read-only run with persisted
  result and event history.
- `harness_get_execution`: retrieve a persisted Maestro execution by UUID.
- `harness_review_changes`: independently review the allowed source snapshot
  and caller-supplied change summary. The service does not execute Git or infer
  a diff; supply the relevant change context explicitly.

All tools validate input. Model output is schema-validated and task
dependencies are enforced by Maestro. Agent reports are size-bounded and do not
include raw provider tokens. File content is treated as untrusted data; there
are no shell, write, or arbitrary network tools.

## Claude Code Windows registration

Use the real interpreter in the harness virtual environment and set the
allowlist in the user-scoped MCP process configuration:

```powershell
claude mcp add --scope user --transport stdio `
  --env AI_HARNESS_ALLOWED_ROOTS=C:\Users\Particular\Projetos\ai-harness `
  ai-harness -- `
  C:\Users\Particular\Projetos\ai-harness\.venv\Scripts\python.exe `
  -m ai_harness.mcp_server

claude mcp list
claude mcp get ai-harness
```

Add another existing project directory to `AI_HARNESS_ALLOWED_ROOTS` only when
that project should be readable by the agents. Do not expose the whole disk.
The MCP caller's explicit `harness_run_agent` request authorizes that
read-only model operation; there is no automatic approval for file writes or
other sensitive actions because the server does not implement those actions.

## Verification boundaries

Unit tests use deterministic test doubles and do not claim model inference.
Run `harness_status` or directly check the configured provider to verify the
endpoint and installed model. This check does not prove that inference finishes
within the configured timeout. Actual agent inference is verified only when a
configured model returns output that passes the report schema.
The optional Anthropic adapter requires its own API credentials and has not
been exercised merely by signing into Claude Code.

## Verification in the Windows environment

Verified on 2026-10-27:

| Area | Result |
|---|---|
| Claude Code CLI | Native executable found at `C:\Users\Particular\.local\bin\claude.exe`; version `2.1.311`. |
| CLI diagnostics | `claude doctor` reported no installation issues. |
| User PATH | Persistently includes `C:\Users\Particular\.local\bin`. The VS Code process used for this work still has its older inherited PATH, so a VS Code restart is required before new terminals launched by that process resolve `claude` without a process-local PATH update. |
| Claude MCP registration | User-scoped `ai-harness` entry is connected over stdio. `claude mcp list` and `claude mcp get ai-harness` succeeded; the existing Claude Docs and Notion entries remain connected. |
| MCP protocol | The server started and completed a stdio handshake; all six tools were discovered, and `harness_status` returned the configured provider, installed model, and agents. |
| Test suite | 83 tests passed. `compileall`, `pip check`, and `git diff --check` also passed. |
| Ollama endpoint/model | The local endpoint responded and `qwen3.8:27b` is installed. This establishes endpoint/model availability only, not successful inference. |
| Live model inference | **Not verified.** Structured-generation attempts timed out at 300 seconds and 180 seconds. The 27B model ran almost entirely on CPU on a system with 4 GB of GPU memory. No generated result was accepted, and no model was downloaded. Live analyst/reviewer execution remains unverified. |
| Anthropic provider | Not exercised; it requires the optional SDK and separate API credentials. |
| ConsolidaHub | `C:\Projetos\consolida` was unavailable during inspection and was not added to the allowlist. Only the AI-Harness repository is currently exposed to the server. |

The PATH directory was added without removing existing user PATH entries. The
old entry that points directly to `claude.exe` remains; the valid directory
entry is the one that enables command lookup. To refresh the environment
inherited by VS Code, close and restart VS Code, then verify with
`Get-Command claude -All` and `where.exe claude`.

The current MCP connection and passing tests do not establish that the local
model can finish real agent or planning requests within a useful time. A
smaller compatible model or more capable inference hardware may be needed;
choose or download another model only after reviewing its requirements and
approving the download.
