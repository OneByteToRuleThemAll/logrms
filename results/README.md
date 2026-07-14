# Experimental evidence

These CSV files are the current evidence used by the accompanying paper.

- `run_summary.csv`: final 49.5M-parameter comparison, including wall time,
  optimizer-step time, full-step time, persistent state, peak allocation, and
  final losses.
- `all_evaluations.csv`: validation evaluations from the 49.5M study.
- `all_checkpoints.csv`: recorded checkpoints from the 49.5M study.
- `tinystories_5m_trajectories_current.csv`: optimizer trajectories from the
  5.06M study.
- `logrms_5m_execution_parity_current.csv`: eager-reference and compiled LogRMS
  trajectories used to check execution parity.

The files record single-run evidence. See `docs/reproducibility.md` and the
paper for protocol details and limits on interpretation.
