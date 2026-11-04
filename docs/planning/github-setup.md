# GitHub configuration and remaining owner actions

Existing repository is reused; no duplicate repository is created. Authenticated owner: `Saad-Salman204`; repository is private and default branch is `main`. See [machine-readable verified state](github-state.json) and [backlog](backlog.md).

## Labels and milestones

20 labels: type feature/research/documentation/testing/maintenance/bug; area data/diffusion/generation/classification/evaluation/infrastructure; priority P0/P1/P2; status blocked/needs-review; effort small/medium/large. Prefixes are `type:`, `area:`, `priority:`, `status:`, `effort:`. Existing GitHub issue open/closed state remains authoritative; no redundant open/closed labels.

Three milestones use relative weeks: Foundation & Baseline; Diffusion & Synthetic Generation; Evaluation & Finalization. No fabricated calendar due dates: set dates after kickoff is agreed. Every planned issue has one milestone, priority, area, type and effort; dependencies use actual issue numbers. Individual assignees remain unset until names/handles are confirmed.

## Projects board — not created

Verified CLI access failed because the token lacks `read:project` (and no `project` write scope was granted). No board URL exists. Repository permissions alone do not grant Projects access.

Exact owner setup path:

1. In GitHub, open your profile → Projects → New project → Board. Name it `Industrial defect diffusion — three weeks` and keep it private.
2. Set Status options: Backlog, Ready, In Progress, In Review, Blocked, Done.
3. Add the existing repository issues using Add item → Add items from repository; include all 20, without recreating them.
4. Save a Board view grouped by Status. Save a Table view grouped by Milestone. Optionally add a Workstream single-select field with Data, Generative ML, Classification & Evaluation, Shared.
5. Link project to the repository from its Projects tab. Set initial items to Backlog; set Ready only when dependency artifacts are available. I01/I02 are initial implementation candidates.
6. Add teammates after confirming handles. Enable closed-issue → Done automation; never auto-complete planning-only items.

For CLI-based creation after interactive authorization: run `gh auth refresh -h github.com -s project`, then `gh project list --owner Saad-Salman204 --format json` to avoid duplicates. Create only if absent with `gh project create --owner Saad-Salman204 --title "Industrial defect diffusion — three weeks" --format json`. Add each existing issue URL with `gh project item-add <project-number> --owner Saad-Salman204 --url <issue-url>`. Configure status/options/views in the web UI and record the verified URL here. These are pending instructions, not executed successful commands.

## Branch protection — blocked by repository plan

The branch-protection API returned HTTP 403: upgrade to GitHub Pro or make the repository public to enable this feature. Privacy is preserved; no upgrade or visibility change was performed. Required review/check enforcement is therefore not configured. Follow the manual one-peer-review/CI policy until the account supports private-repository protection.

Once supported: Settings → Branches → Add branch protection rule for `main`; require PR, one approval, dismiss stale reviews, require conversation resolution and the `foundation` check after it has run. Disable force pushes/deletions and apply to administrators where practical. Do not require CODEOWNER approval until real owners are configured. Verify with `gh api repos/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/branches/main/protection`.

## Teammates and review

Settings → Collaborators → Add people using confirmed handles. Do not infer usernames from proposal names. Assign agreed issues, activate CODEOWNERS only for consenting users with access, and request a peer review on the foundation PR. Publishing a branch is not approval to merge it. No implementation task should be marked complete merely because its issue was created.
