# Contributing

Bug reports, reproducibility reports, and focused pull requests are welcome.

For code changes:

1. Install the test dependencies with `python -m pip install -e '.[test]'`.
2. Add or update tests for changed behavior.
3. Run `pytest` before opening a pull request.
4. Explain any change to optimizer mathematics, persistent state, checkpoint
   format, or compilation behavior explicitly.

Changes to the compiled implementation should retain numerical parity tests
against the canonical reference implementation. New mathematical variants
should be proposed separately instead of silently changing the registered
optimizer.
