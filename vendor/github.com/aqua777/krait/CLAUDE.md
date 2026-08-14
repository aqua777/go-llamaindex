# Krait

**Combines Viper and Cobra to provide unified interface for defining configuration capabilities through defaults, config files, environment variables and command line parameters**

Krait provides a unified interface for building command-line applications by combining
the functionality of Cobra (command-line interface) and Viper (configuration management) into
a single, cohesive package.

Krait simplifies the process of creating CLI applications with rich configuration support by
offering a fluent API that handles both command definitions and configuration management.

```
Config Sources (priority: CLI flags > env vars > config file > defaults)
                              |
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
 CLI Flags               Env Vars             Config File
 (Cobra/pflag)           (Viper AutoEnv)      (Viper ReadInConfig)
      │                       │                       │
      └───────────────────────┼───────────────────────┘
                              │
                         krait.Command
                    ┌─────────┴──────────┐
                    │                    │
               viper (named)       argsViper (var-ptr)
               GetString(key)      *string, *bool, etc.
                    │                    │
                    └─────────┬──────────┘
                              │
                   internalProcessParams()
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        SanityCheck()   BeforeRun()        Run()
                                               │
                                          AfterRun()
```

## LLM-Driven Project

This project uses LLM-assisted development. All documents are optimized for LLM context efficiency while maintaining comprehensive coverage.

---

## Core Principles

These principles are **non-negotiable** and guide all decisions.

<!--
CUSTOMIZATION GUIDE:
- Keep 5-9 principles (fewer = forgotten, more = ignored)
- Each principle needs: name, one-line summary, concrete examples
- Include both "what to do" and "what this prohibits"
- Principles should conflict sometimes (forces explicit tradeoffs)
-->

### 1. {User Focus Principle}

Developers define CLI commands, parameters, and configuration in one fluent chain—no
wiring Cobra and Viper together by hand, no boilerplate for binding flags to env vars
or config files.

**Examples:**
- `krait.App(...).WithStringP(...).WithRun(...)` is a complete, runnable CLI command
- Shared params defined once in a `ConfigParams` object and reused across subcommands
- Values read at runtime via `krait.GetString("key")`—no manual `viper.BindPFlag` calls

**What this prohibits:**
- Exposing raw `*cobra.Command` or `*viper.Viper` objects to the caller
- Requiring separate setup steps for flags, env vars, and config file binding

### 2. {Configuration Philosophy}

Opinionated about mechanics, flexible about content. The package hard-wires how config
sources are resolved (CLI flags > env vars > config file > defaults) and how the
execution lifecycle runs (processParams → SanityCheck → BeforeRun → Run → AfterRun).
Callers configure what: flag names, env var names, types, defaults, config file paths,
and the functions that run at each lifecycle hook. Unsupported flag types panic at
startup—fail loudly rather than silently mishandle data.

### 3. {Quality Philosophy}

Correctness is guaranteed by a comprehensive unit test suite, not by defensive runtime
checks. Each sub-package maintains ≥90% test coverage. Code that cannot be easily tested
must be documented as such in the relevant test file.

### 4. {Error Handling Philosophy}

Fail fast, fail loudly. Configuration is resolved at startup—any invalid input, missing
required value, unsupported type, or binding failure must abort immediately with a clear
error message identifying what failed and why. Never silently fall back to a default or
guess intent. Programmer errors (e.g. unsupported flag type) panic; user errors (e.g.
bad config file path) return an error that Cobra surfaces before the run function is
ever called.

### 5. No Future Scaffolding

Don't write code for functionality that doesn't exist yet. Add it when the feature is implemented, not before.

**What this prohibits:**
- Loops that iterate but do nothing (`for x in items: pass`)
- Method stubs for unimplemented features
- Parameters accepted but ignored
- `# TODO:` comments with placeholder code

**What this permits:**
- Extension points that ARE used (abstract classes with implementations)
- Comments explaining design decisions (not placeholders)

### 6. {Breaking Change Policy}

Breaking changes must be documented in `CHANGES.md` and signaled by a minor version bump.
Non-breaking changes (new features, bug fixes) require only a patch version bump. No
deprecation shims—when something breaks, it breaks explicitly and is documented.

### 7. One Declaration, All Sources

Every parameter is declared exactly once. A single `With*` call registers the flag,
binds the environment variable, sets the default, and wires the config file key
simultaneously. If adding support for a new configuration source requires touching more
than one place in user code, that is a design defect in krait—not an expected workflow.

---

## Documentation Structure

| Document | Purpose | When to Load |
|----------|---------|--------------|
| `CLAUDE.md` | Principles, invariants, navigation | Always |
| `docs/CAPABILITIES.md` | What the system does | Feature overview |
| `docs/architecture/*.md` | Design + rationale | Implementation, design questions |
| `docs/sprints/` | Sprint specs and status | Sprint work |

**Loading strategy:** See `docs/architecture/README.md` for document index and reading order.

---

## Key Invariants

Always true in this codebase:

| Invariant | Meaning |
|-----------|---------|
| Deterministic | Same inputs = identical outputs |
| Fail fast | Invalid input = error, not default |
| {Custom} | {Description} |
| {Custom} | {Description} |

---

## DO NOT

- Write code for features that don't exist yet (Principle #5)
- {Principle violation from #1}
- {Principle violation from #2}
- {Domain-specific prohibition}
- {Domain-specific prohibition}

---

## Anti-Patterns

Concrete patterns to avoid and detect during review.

### Over-Engineering

| Pattern | Example | Fix |
|---------|---------|-----|
| One-use abstractions | `class BaseX` with single subclass | Inline it |
| Impossible error handling | Validating internal inputs | Trust internal code |
| Defensive copies | `list(x)` for already-owned data | Remove copy |
| Premature helpers | `_format_x()` called once | Inline it |
| Hypothetical features | Designing for "what if" | Build for current needs |

### Dead Code

| Pattern | Detection |
|---------|-----------|
| Unused imports | Linter / grep for usage |
| Uncalled functions | Grep for function name |
| Unreachable branches | `if False:` or impossible conditions |
| Commented-out code | Delete it (git has history) |

### Test Anti-Patterns

| Pattern | Why It's Bad | Fix |
|---------|--------------|-----|
| Testing framework internals | Tests the library, not our code | Test behavior instead |
| No assertions | False confidence | Add meaningful assertions |
| Implementation details | Brittle to refactoring | Test public interfaces |
| Duplicate coverage | Maintenance burden | Consolidate or delete |

### {Domain-Specific Anti-Patterns}

| Pattern | Why It's Bad | Fix |
|---------|--------------|-----|
| Calling `krait.Get*` before `Execute()` | Returns zero values from a bare Viper; the command's configured defaults and env bindings are not yet active | Only read values inside `Run`, `BeforeRun`, `AfterRun`, or `SanityCheck` |
| Registering the same parameter on each subcommand individually | Defeats the purpose of `ConfigParams`; changes must be made in multiple places | Define shared params once with `NewConfigParams()` and apply via `WithParams()` |
| Reading raw `*cobra.Command` or `*viper.Viper` from `Command` fields | Bypasses krait's source-priority logic and lifecycle hooks | Use krait's typed getters (`GetString`, `GetInt`, etc.) |
| Using `viper.GetString` / `cobra.Command` directly alongside krait | Creates a parallel config path that ignores env var and config file bindings krait manages | Migrate all param declarations to krait's fluent API |
| Checking `IsDebug()` outside of a command's run lifecycle | `currentCommand` is nil before `Execute()`; always returns false | Call `IsDebug()` only from within `Run`, `BeforeRun`, or `AfterRun` |
| Defining config file path as a plain string default without `WithConfig` | Env var and CLI flag override for the config path won't work | Always use `WithConfig` so the path itself participates in source priority |

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md) | What the system does |
| [docs/architecture/](docs/architecture/) | Design rationale and constraints |
| [docs/sprints/](docs/sprints/) | Sprint specifications |


## Code Navigation
  - Prefer `mcp__cclsp__*` tools over reading entire files for understanding code structure.
  - **Working with pylsp:** `find_definition`, `find_references`, `get_hover`, `get_diagnostics`, `rename_symbol`. Use these freely.
  - For "who calls this function?" questions, use `find_references` or Grep.
  - LSP tools work in both the main conversation and subagents.
