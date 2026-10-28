# 3-axis arm digital twin: pick & place with motor degradation (proof of concept)

Open `arm_digital_twin.html` in a browser (works offline, ~1 MB).

What it is
- Procedural 3-axis arm (J1 base yaw, J2 shoulder, J3 elbow) carrying real-ish automotive parts: spark plug 0.06 kg, fuel injector 0.10, oil filter 0.45, piston+rod 0.60, brake disc 1.20, alternator 1.50.
- Each joint = DC gearmotor + 1000:1 gearbox + encoder + PID (same model as the servo twin), coupled through arm gravity/inertia.
- Per-joint health slider (100% new -> 40% worn). Cycle: lift -> swing 75 deg -> lower, with the part held.
- RL tuner outputs all 9 PID gains from payload + per-joint health. Compared against: RL tuned when new, gain compensation (new gains / (Kt/R)), fixed gains.

ASSUMPTIONS (not measured data)
- Arm: L2 0.20 m, L3 0.18 m, link masses 0.25/0.18 kg, gripper 0.10 kg, shoulder height 0.30 m, 1.5 kg rated payload.
- Wear at d = 1 - h: R x(1+0.8d), Kt and Ke x(1-0.5d), viscous friction x(1+6d), Coulomb friction 0.1+1.5d mN.m, gearbox efficiency 0.80-0.25d.
- Not modelled: backlash, thermal drift, joint coupling beyond gravity/inertia, encoder aging. Health is told to the controller, not estimated.
- The gripper is kept vertical by a visual-only passive wrist.

Held-out results (150 cycles per row, brake disc / alternator payloads, % placed within 2 mm):
health 100%: aware 100 | new-tuned 100 | compensation 100 | fixed 35
health  80%: aware 100 | new-tuned 100 | compensation 100 | fixed 0
health  60%: aware 100 | new-tuned   0 | compensation   0 | fixed 0
health  40%: nobody places it (motors near their torque limit) but aware has the smallest error

Files: train_arm.py (REINFORCE trainer, `python train_arm.py aware 350`; `nominal` for the baseline), arm_sim.py (physics, shared with the page), eval_arm.py, arm_policy_*.json, arm_config.json.
Note: policies were trained for ~250-300 iterations; several gains sit at the upper search bound (Kp 300, Ki 600).
