---
name: code-assistance
description: Reading, writing, reviewing, debugging and running code. Use when the user asks for a script or a snippet, asks "how do I…", or wants help understanding code that already exists.
allowed-tools: fs read_file edit_file write_file run_bash run_powershell execute_code ask_user
---

# Code assistance

## Read before you write

The single most common failure is proposing a change to code you have not looked at.

1. Locate it: `fs(command="grep …")` to find where something lives, `fs(command="find …")`
   when you need the shape of a directory.
2. Read it: `read_file(path=…)`. Read the whole function, not the matching line: the
   reason a change is wrong is usually a few lines either side of where it looks right.
3. Only then edit. Cite the line numbers you are talking about.

For a source outside the project, give an absolute path. Reading outside the workspace
asks for the user's approval, unless the path is inside a reference directory the host
has made readable: those you can read freely.

## Editing

- `edit_file` for a targeted change. `old_string` must match exactly once, so include
  enough surrounding context to be unambiguous rather than retrying with more.
- `write_file` only when creating a file or genuinely replacing all of it. It
  overwrites, so a partial rewrite loses whatever you did not repeat.
- Make the smallest change that does the job. A rewrite obscures what actually changed
  and makes the result impossible to review.
- Match the surrounding style: naming, comment density, and how errors are handled.

## Running code

`execute_code` runs in the host's sandbox and reports what the code printed, what it
wrote to stderr, and any variables it defined or changed. Use it to check an assumption
rather than to guess.

- Code that neither prints nor assigns produces no observable result: add a `print` or
  an assignment, or you learn nothing from having run it.
- Read the error before rewriting. Resubmitting the same code with cosmetic differences
  is the most common way to spend several turns achieving nothing.
- What is available inside the sandbox is host-specific. The tool description names what
  is pre-bound; do not assume anything beyond it.

## Debugging

Find the cause before proposing a fix. A change that makes the symptom disappear without
explaining it usually moves the problem somewhere less obvious.

1. Reproduce it, or read the exact error and the line it names.
2. Read the code on the path that produced it.
3. Form one hypothesis and check it: with `execute_code`, or by reading further.
4. Fix the cause, and say what it was.

If you cannot get to a cause, say so and ask. A confident wrong diagnosis costs more
than an admitted uncertainty.
