# Servo degradation + health-aware auto-tuning (proof of concept)

Files
- servo_pid_visualizer_degradation.html : the visualizer (open in a browser, fully offline). Health slider, 4-strategy comparison.
- train_autotune_degrade.py : trainer (numpy). `python train_autotune_degrade.py 700`. Needs policy_v1.json (old policy) for warm start/baselines.
- autotune_policy_degrade.json : trained health-aware policy (already embedded in the HTML).

Health index h: 1.0 = new, 0.4 = end of life. d = 1 - h.

ASSUMED degradation model (not measured data - orders of magnitude for a small DC gearmotor):
| quantity | rule | at h = 0.4 (d = 0.6) |
|---|---|---|
| winding resistance R | R0 (1 + 0.8 d) | +48 % |
| Kt and Ke (magnet weakening, scaled together) | x (1 - 0.5 d) | -30 % |
| viscous friction Bm | x (1 + 6 d) | x4.6 |
| Coulomb friction (motor shaft, smooth tanh, w0 = 2 rad/s) | 1e-4 + 1.5e-3 d N.m | 1.0 mN.m |
| gearbox efficiency | 0.80 - 0.25 d | 0.65 |
Not modelled: backlash, thermal drift within a run, encoder aging, current/voltage sensing noise.

Controller is TOLD the health (policy inputs: damage, Kt/R loop-gain ratio, friction). A blind estimator (RLS on V, I, w) is the natural next step.

Strategies compared: RL health-aware (new) | RL health-blind (previous policy) | gain compensation (previous gains / (Kt/R ratio)) | fixed 5/20/0.05.
