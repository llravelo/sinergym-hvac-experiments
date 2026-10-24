# Working state (temporary — durable facts belong in CLAUDE.md, not here)

**Current task:** regenerate results by running `make run` on the training
machine, then copy `results/cadence_v2/gui_bundle/cadence_v2.zip` to the GUI repo.

**Unresolved questions:**
- `../43008-at03-gui/notebook/cadence_hvac_dqn_ppo_v2.ipynb` was changed on
  2026-10-24 to inline the bundle contract. Decide which notebook copy is
  canonical; if the inlined one, drop `tools/cadence_bundle.py`.
- No official rubric/AI-use policy found in repo.

**Next step:** first `make run` on the training machine. Watch the
"Environment probe" cell for env-id or variable-name errors.
