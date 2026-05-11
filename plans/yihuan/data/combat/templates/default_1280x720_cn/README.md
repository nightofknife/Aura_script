# Combat Template Directory

This directory now contains the baseline `1280x720` combat templates used by
`yihuan_combat`:

- `target_lock_diamond.png`: target-lock diamond shown after pressing the middle mouse button
- `challenge_success.png`: challenge result banner used as a strong post-combat signal

The service still uses HSV and region rules for enemy health bars, team HUD, and
ability readiness. If we need extra stability later, we can add more templates here
and reference them from `data/combat/default_1280x720_cn.yaml`.
