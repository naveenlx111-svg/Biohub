# Research member workspaces

Each team member owns one folder under this directory:

```text
research_members/
  member_name/
    notebooks/
    experiments/
    notes/
```

Use member folders for notebook sources, kernel configurations, exploratory scripts, and working notes. Use the shared locations for team-level information:

- `experiments/ledger.csv` for every planned, running, completed, or rejected experiment;
- `docs/results/` for reviewed outcomes;
- `docs/roadmap.md` for team priorities;
- `README.md` for repository-wide workflow.

Experiment IDs are global across the team. Reserve an ID in the shared ledger before launching a run. Work on a short-lived task branch and merge through a pull request so another member can review configuration and result attribution.

Do not commit competition data, generated submissions, downloaded checkpoints, access tokens, or Kaggle output directories.
