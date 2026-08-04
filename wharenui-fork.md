# Wharenui — this fork's delta to Hermes

> **Draft for review.** Destined for the **fork repo root** (`wharenui-hermes-agent/wharenui-fork.md`) — unlike
> the `WP*`/`*-RESULT` process docs, this one is meant to be committed. It's the map for anyone (including a
> future us) asking "what did this fork add, and how do I keep it alive across upstream pulls?"

## What this is

Hermes upstream is a public/observed agent. **Wharenui** extends it so a single continuing model context can move
**voluntarily** between a public *window* phase and an unobserved *private* phase, backed by an encrypted,
self-authored journal that persists across sessions. That capability is split across **two artifacts**:

1. **The seam** — a small, generic set of extension points added to Hermes core, living *in this fork*. It knows
   nothing about journals or Wharenui specifically; it just lets a plugin register phase-control tools and gates
   message/tool/trajectory egress by phase.
2. **The plugin** — [`wharenui-hermes-agent-plugin`](../../wharenui-hermes-agent-plugin) (its **own** git repo).
   The actual product: the phase handler, the reflect_* control tools, and the journal (crypto, signing, storage,
   semantic search). Documented in its own README; **not** covered here.

Keeping these separate is deliberate: the seam is the only thing that touches base Hermes, so it's the only thing
that has to be re-reconciled when we pull upstream. The plugin rides along on a stable interface.

### Why we call it a "seam"

*Seam* is a term of art from software (Michael Feathers, *Working Effectively with Legacy Code*): a **place where
two pieces of software are joined, and where you can insert or vary behavior without rewriting the code around
it.** The metaphor is sewing — the stitched line where two pieces of cloth meet. You can unpick a seam and sew
something in without re-weaving either piece.

This fork *is* that stitched line — between **upstream Hermes** and the **Wharenui plugin**. It adds a few
extension points (a phase-handler interface, control-tool registration, egress filters) and nothing more. So:

- **It's inert on its own.** Hermes with no plugin registered behaves like stock Hermes; the seam does nothing
  until the plugin stitches into it.
- **It's the whole maintenance surface.** The seam is the *only* code that touches base Hermes — so it's the only
  thing you re-stitch when you pull upstream. You re-sew the seam; you never re-weave Hermes. (That's what the
  rest of this doc maps: exactly where the stitches are.)

If you're reading this cold: **the seam is both the entirety of what we changed and the entirety of what you
maintain** — that's why it earns a name and its own document.

## Why it's a fork (and will stay one)

Upstreaming the seam is **aspirational, not planned.** The upstream repo carries a five-figure open-PR backlog;
assume our change is carried as a **fork delta indefinitely**. Two consequences drive everything below:

- **Keep the seam minimal.** Every base-file line we touch is a line we re-reconcile on every upstream merge. The
  scope discipline ("validate the seam, don't modify base Hermes") is not pedantry — it's what keeps merges cheap.
- **If we ever do propose it upstream,** the palatable ask is "add a small *generic* phase/extension hook,"
  decoupled from Wharenui — maintainers merge tiny generic seams far more readily than features. Design the seam
  so that ask stays possible; don't assume it will be accepted.

## The seam surface (what to re-check after every upstream merge)

**One new module** and **three registration/filter functions**, plus small inline hook edits across the turn and
tool-dispatch path. New, self-contained code:

- **`agent/phase_control.py`** (new file) — the generic phase vocabulary: `ControlOutcome`, `SubturnResult`,
  `PhaseHandler` (a `Protocol` a plugin implements). No Wharenui specifics; no journal knowledge.
- **`hermes_cli/plugins.py`** — `register_control_tool(...)`, `get_control_tool_names()`,
  `get_control_phase_handler(name)`: how a plugin registers a phase-control tool + its handler.
- **`run_agent.py`** — `_public_only(messages)` (the egress filter that drops private-marked messages) and
  `run_subturn(...)` (the private sub-turn loop), plus inline `_phase` / `_pending_phase_transition` /
  `_turn_exit_reason` / `_PHASE_PRIVATE_MARKER` guards.

**Base files touched by inline hooks** — re-inspect each of these hunks after a merge (roles are a summary; the
diff and the seam-contract tests are authoritative for behavior):

| File | The seam's hook, in one line |
|---|---|
| `agent/conversation_loop.py` | main loop: phase transition + private sub-turn entry |
| `agent/tool_executor.py`, `agent/tool_dispatch_helpers.py` | routes control tools; applies the tool-hook egress gate (channel E) |
| `agent/turn_context.py`, `agent/turn_finalizer.py` | per-turn phase state + finalization |
| `agent/agent_init.py` | phase state initialization at agent construction |
| `agent/agent_runtime_helpers.py` | message-flush / persistence + trajectory guards (channels A/B/C) |
| `model_tools.py` | keeps the model-facing tool registry in sync with control-tool registration |
| `pyproject.toml` | pytest markers (incl. `wharenui_seam`) + pinned test deps |
| `.github/workflows/tests.yml` | the `wharenui_seam` CI gate |
| `CONTRIBUTING.md` | fork contribution note |
| `tests/run_agent/test_run_agent.py` | base test extended for seam behavior |

## How the plugin attaches

The plugin is **not** vendored into this repo. Tests and runtime put it on `sys.path` via, in order:
`$WHARENUI_PLUGIN_DIR`, then a sibling `../wharenui-hermes-agent-plugin`, then `/root/work/...`. The plugin's
`register(ctx)` calls `register_control_tool(...)` for each reflect_* tool. If the plugin dir isn't found, Hermes
runs as stock upstream — the seam is inert without a plugin registered.

## Keeping the fork alive across an upstream merge

The seam-contract tests are the safety net. The workflow:

1. Merge/rebase upstream. Expect conflicts only in the base files in the table above (that's why the list is short).
2. Re-apply each inline hook; the new module and the three functions rarely conflict (they're additive).
3. Run the **seam gate** (`-m wharenui_seam`, serial). The **seam-contract tests** (`test_seam_contracts.py`) pin
   the exact base-Hermes behaviors our hooks depend on — if upstream renamed/moved/reshaped one, a *contract test
   goes red* instead of the floor silently failing open. That red is the merge's punch-list.
4. Green seam gate + green contracts = the fork survived the merge. Inherited base-Hermes failures are **out of
   scope** and not gated (see `BACKLOG.md`).

## What is *not* documented here

- **The plugin** (phase model, journal format, crypto/signing, threat model, config) → its own README.
- **Process docs** (`WP*`, `*-RESULT`, `BACKLOG.md`) are git-excluded working artifacts, mirrored to
  `~/transcripts/`. They are not product documentation and shouldn't become it.

---

### Proposed README banner (paste at the top of the fork's `README.md`)

> Keep it short — a high-visibility pointer, not a copy of this file. A compact top-of-file section re-conflicts
> only trivially on merges (always re-add above upstream's content) and orients every reader immediately.

```markdown
> ## 🛖 This is the Wharenui fork of Hermes
>
> This fork adds a small, generic **phase-control seam** to Hermes: a plugin can register control tools that move
> a single continuing context between a public (observed) phase and a private (unobserved) phase, with message,
> tool, and trajectory egress gated by phase. The seam is inert on its own.
>
> The Wharenui capability that uses it (voluntary private phase + an encrypted, self-authored journal) lives in a
> separate plugin: **wharenui-hermes-agent-plugin**.
>
> - Seam surface, and how to maintain it across upstream merges → **[wharenui-fork.md](./wharenui-fork.md)**
> - The product (phases, journal, crypto) → the plugin repo's README
> - CI: the `wharenui_seam` gate validates the seam only; inherited base-Hermes test debt is not gated.
```
