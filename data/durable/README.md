# Durable architectural decisions

Operator-curated rules that govern the writer ↔ reviewer review loop.
Loaded at runtime by the constrained-kickoff prompt builder when the
review hits the repeated-blocker escalation threshold (see PR #38).

## How it's used

When the reviewer flags the same critical blocker in two or more
rounds of a single review session, the orchestrator stops generic
rebuilds and switches to a constrained-architecture rebuild. That
prompt embeds **every** ADR with `status: durable` from this
directory as hard structural constraints. Edit the markdown files
here to change the rules; no code change required.

## Format

Each ADR is one markdown file with YAML frontmatter:

```markdown
---
id: adr-XXXX                         # required, unique
title: One-line summary              # required
status: durable                      # required: provisional | durable | superseded | rejected | flagged | archived
valid_from: YYYY-MM-DD               # when the rule became true in the world
valid_to:                            # null while current; set when superseded
transaction_time: YYYY-MM-DD         # when we recorded it (may differ from valid_from)
supersedes: []                       # list of earlier ADR ids this replaces
superseded_by:                       # null while current
applies_to:                          # tags — currently advisory, no filtering yet
  - some-domain
provenance:                          # where the rule came from
  - PR-XX, JSONL transcript line ZZ
---

The actual rule body in plain markdown. Brief, imperative.
The model sees `### {id}: {title}\n\n{body}` in its prompt.
```

## Status semantics

- `durable` — currently in force; loaded into prompts.
- `provisional` — in operator review queue; **not** loaded into
  prompts yet. Lives in `../proposed/` by convention but the
  status field is the source of truth.
- `superseded` — replaced by a newer ADR. Kept for audit history;
  not loaded into prompts.
- `rejected` — operator declined to promote. Kept for audit only.
- `flagged` — needs operator attention (auto-flagged by entropy
  heuristics; not yet implemented).
- `archived` — no longer relevant (e.g. the subsystem was removed).

Only `durable` ADRs influence the next review.

## Append, don't rewrite

To change a rule, **add a new ADR** with `supersedes: [adr-NNNN]`
and flip the old one's status to `superseded` with
`superseded_by: adr-MMMM`. Don't edit a confirmed `durable` ADR in
place — the audit trail loses the previous decision and the
provenance chain breaks.
