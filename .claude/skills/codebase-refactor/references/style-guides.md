# Style principles (condensed from Google/Microsoft-style engineering conventions)

Apply these during the judgment-call editing step (Step 4.3 in SKILL.md). This is a condensed
cheat sheet, not a replacement for a full style guide — when in doubt about a language-specific
convention the user's team already follows (existing linter config, `.editorconfig`, CONTRIBUTING.md),
that project's own conventions win over these defaults.

## Universal principles
- **Single Responsibility**: a file/class/function should have one reason to change. If a file mixes
  data access, business logic, and formatting/output, that's a split candidate.
- **Extract, don't duplicate**: if the same logic appears 3+ times, extract it — but don't over-abstract
  something used twice; premature abstraction is its own smell.
- **Name for intent, not implementation**: `calculate_shipping_cost()` not `calc2()` or `doStuff()`.
  Booleans read as questions (`is_valid`, `has_permission`). Avoid abbreviations unless they're
  domain-standard (`id`, `url`, `http` are fine; `cst_mgr` is not).
- **Early returns over deep nesting**: guard clauses at the top of a function beat wrapping the whole
  body in nested `if`s.
- **Comments explain why, not what**: the code should already say what it does. Comment the
  non-obvious reasoning, trade-off, or gotcha.
- **Public functions/classes get docstrings** (Python docstrings, JSDoc, Go doc comments, Javadoc) —
  what it does, params, return, and any thrown/raised errors that matter to the caller.
- **Dependencies point one direction**: lower-level modules shouldn't import from higher-level ones.
  If splitting a file reveals a circular dependency, that's a structural signal, not just a naming one.

## Per-language notes

**Python** — PEP 8 naming (`snake_case` functions/vars, `PascalCase` classes), type hints on public
function signatures, f-strings over `.format()`/`%`, `pathlib` over manual string path joins.

**JavaScript/TypeScript** — `camelCase` vars/functions, `PascalCase` classes/components/types,
prefer `const`, avoid default exports in favor of named exports for anything non-trivial (easier to
grep/refactor), colocate types with usage unless shared widely.

**Go** — short receiver names, exported identifiers `PascalCase` with doc comments starting with the
identifier name, errors returned not thrown, avoid unnecessary interfaces (accept concrete types,
return concrete types unless there's a real need for abstraction).

**Java** — `PascalCase` classes, `camelCase` methods/fields, prefer composition over inheritance,
package-private by default, only widen visibility when there's a real caller outside the package.

## Splitting a file — how to decide the boundary
1. List the file's top-level functions/classes and what each one actually depends on.
2. Group by what changes together (same data, same feature) — not by superficial similarity (e.g.
   "all the getters" is usually the wrong grouping; "everything related to user auth" is usually right).
3. Each resulting file should be independently understandable without needing to also read its sibling.
4. Keep the public interface stable where possible — internal reorganization shouldn't force every
   caller elsewhere in the codebase to change, unless that's explicitly part of the refactor.
