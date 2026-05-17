# Combat Template Directory

This directory now contains the baseline `1280x720` combat templates used by
`yihuan_combat`:

- `target_lock_diamond.png`: target-lock diamond shown after pressing the middle mouse button
- `remaining_enemy_marker.png`: the cyan mark to the left of the "remaining enemies" HUD text
- `challenge_success.png`: challenge result banner used as a strong post-combat signal
- `reward_marker.png`: tight post-combat reward location marker, with surrounding scene pixels removed
- `claim_memento_prompt.png`: tight dialog-bubble icon from the "claim memento" interaction prompt; the profile searches it only inside a fixed small ROI
- `reward_claim_single_button.png`: post-interaction reward confirmation button for normal claim
- `reward_claim_double_button.png`: post-interaction reward confirmation button for double claim
- `reward_result_exit_button.png`: reward result screen button for leaving after the final run
- `reward_result_retry_button.png`: reward result screen button for starting the next challenge

The service still uses HSV and region rules for enemy health bars, team HUD,
remaining-enemy marker fallback detection, and ability readiness. If we need extra
stability later, we can add more templates here and reference them from
`data/combat/default_1280x720_cn.yaml`.
