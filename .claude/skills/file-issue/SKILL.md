---
name: file-issue
description: Opening, claiming and closing issues in proqu. Use when a problem or a piece of work turns up, before starting work on an issue, and when a pull request finishes one.
---

# Filing issues

Issues are the log of what is known and who is on it. Every problem found and every piece of work identified gets one, even when nothing will be done about it now, so the log is complete.

The one exception is a small change, one commit or docs only. It goes straight to a pull request with no issue first.

## Before filing

- Search for one that already covers it, open and closed: `gh issue list --state all --search "<words>"`. Add a comment to an existing issue rather than opening a second.
- Read the issues the work touches. One may say why it has not been done, or carry a consideration the change must honour.

## Opening one

Use the template in `.github/ISSUE_TEMPLATE` that fits: feature, fix or chore. The title is conventional, `type: subject`, in plain words. The types in use are feat, fix, perf, test, docs and chore.

The body keeps the template's sections:

- **What**: what is missing or wrong today, and where it lives. File and function references are welcome here.
- **Why it matters**: what goes wrong, or what cannot be done, until it is closed.
- **What needs deciding first**: choices to settle before work starts. Delete if there are none.
- **What would close it**: the observable result that ends it. A case that would prove it is a good close item.
- **Notes**: ordering, blockers, related issues. Delete if not needed.

Describe the problem on its own merits. Never mention reviews, workflows, agents, plans or scratch work, and never say how the problem was found. The prose rules in CLAUDE.md apply: British English, plain words, no em dashes and no semicolons.

When the issue comes from someone else's merged work, mention them in the notes for visibility.

## Priority

Every issue gets exactly one priority label when it is opened.

- `priority: critical`: data, money or security at risk with nothing else to catch it.
- `priority: high`: wrong behaviour on a path the project relies on.
- `priority: medium`: a correctness gap off the main path, or missing coverage that matters.
- `priority: low`: tidying, docs, and edges unlikely in practice.

```sh
gh issue create --label "priority: medium" --title "fix: ..." --body-file body.md
```

Write the body to a file first. Long bodies passed inline are easy to break with quoting.

## Claiming one

- Read its comments first. If someone else says they are on it, stop and tell the user before doing any work.
- Comment that it is being picked up before starting, and name any other issues landing in the same pull request.

```sh
gh issue comment <n> --body "Picking this up now, in one pull request into dev alongside #<m>."
```

## Closing one

- The pull request closes every issue it finishes, with `Closes #<n>` in its body, one per issue.
- Closing keywords only fire when a pull request merges into the default branch, which is main. An issue closed by a pull request into dev stays open until dev reaches main, so do not read an open issue as unfinished without checking what has merged.
- A pull request that only partly covers an issue says what is left in a comment on the issue rather than closing it.
