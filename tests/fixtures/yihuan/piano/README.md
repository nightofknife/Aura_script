# Yihuan Piano MIDI Fixtures

- `no_roll_needed.mid`
  Uses a same-onset chord that spans natural and sharp notes on different physical keys.
  Expected behavior:
  `strict` succeeds
  `roll` succeeds
  No conflict-driven staggering is required

- `roll_required.mid`
  Uses a same-onset chord that contains `1` and `#1` in the same register, which share one physical key.
  Expected behavior:
  `strict` fails with `physical_key_conflict`
  `roll` succeeds by staggering the conflicting notes

- `happy_birthday_tri_batch.mid`
  Uses the public-domain melody of Happy Birthday with a test arrangement that deliberately
  places natural, sharp, and flat batches on the same beat.
  Expected behavior:
  `strict` succeeds because there is no physical-key conflict
  `roll` groups the same-onset notes into three batches in this order:
  natural notes, then `Shift` notes, then `Ctrl` notes
