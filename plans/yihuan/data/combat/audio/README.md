# Combat Audio Samples

This directory stores waveform samples used by Yihuan combat audio triggers.

- `dodge_sample.wav`: baseline dodge cue sample used by the audio-dodge runtime

Current implementation borrows the sample-matching idea from `NTESoundTrigger`
and keeps the sample local to the Yihuan combat package so the feature can be
calibrated independently from visual combat logic.
