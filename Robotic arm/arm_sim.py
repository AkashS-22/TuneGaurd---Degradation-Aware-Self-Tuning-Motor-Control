"""3-axis arm (RRR) physics for the pick-and-place degradation demo.  numpy, vectorised over B contexts.

Joints: J1 base yaw (vertical axis, no gravity), J2 shoulder (pitch), J3 elbow (pitch, RELATIVE angle).
Angles for the pitch joints are measured from straight UP:  phi2 = q2 (upper arm),  phi3 = q2 + q3 (forearm).
Each joint = the same DC gearmotor + 1000:1 gearbox + encoder + PID model as the single-servo twin,
with per-joint health (degradation) and a coupled gravity load computed from the arm pose.

ALL ARM DIMENSIONS / MASSES / DEGRADATION COEFFICIENTS ARE ASSUMPTIONS (desk-scale demo arm).
"""
import json
import numpy as np

MOTOR = dict(V=12.0, ratio=1000.0, eff=0.80, cpr=9360, R=1.6, Kt=1.974e-3, Ke=6.366e-3, Jm=5.0e-7, Bm=1.0e-6, tau_f=0.01)
ARM = dict(L2=0.20, L3=0.18, lt=0.09, m2=0.25, lc2=0.10, m3=0.18, lc3=0.09, mg=0.10, Jbase=0.002, z0=0.30, g=9.81)
DT = 0.002
H_MIN = 0.4
FRIC_W0 = 2.0
NOISE_COUNTS = 1.0
MP_MAX = 1.5            # rated payload [kg]
V_LIMIT = 12.0

# pick-and-place cycle (carrying the part): task-space waypoints -> joint space by IK
R_REACH, Z_LOW, Z_HIGH, SWING_DEG = 0.26, 0.14, 0.26, 75.0
HOLD0, T_LIFT, T_SWING, T_LOWER, T_SETTLE = 0.3, 0.9, 2.0, 0.9, 0.6
T_TOTAL = HOLD0 + T_LIFT + T_SWING + T_LOWER + T_SETTLE
N_STEPS = int(round(T_TOTAL / DT))

PARTS = [  # name, mass [kg]  (real-ish masses, picked to fit the arm's 1.5 kg rating)
    ("No part", 0.0), ("Spark plug", 0.06), ("Fuel injector", 0.10), ("Oil filter", 0.45),
    ("Piston + con-rod", 0.60), ("Brake disc", 1.20), ("Alternator", 1.50)]


def ik(r, z_tool, A=ARM):
    """Planar 2-link IK, elbow-up. Returns (q2, q3) [rad] with phi from up, q3 relative."""
    zw = z_tool + A['lt'] - A['z0']
    d2 = r * r + zw * zw
    c = (d2 - A['L2'] ** 2 - A['L3'] ** 2) / (2 * A['L2'] * A['L3'])
    q3 = np.arccos(np.clip(c, -1, 1))
    alpha = np.arctan2(r, zw)
    q2 = alpha - np.arctan2(A['L3'] * np.sin(q3), A['L2'] + A['L3'] * np.cos(q3))
    return q2, q3


def waypoints():
    sw = np.radians(SWING_DEG)
    q2l, q3l = ik(R_REACH, Z_LOW); q2h, q3h = ik(R_REACH, Z_HIGH)
    return np.array([[0, q2l, q3l], [0, q2h, q3h], [sw, q2h, q3h], [sw, q2l, q3l]])   # P0 pick, P1 lift, P2 swung, P3 place


def reference():
    """(N,3) min-jerk joint references [rad] for the cycle."""
    W = waypoints(); t = np.arange(N_STEPS) * DT + DT
    ref = np.tile(W[0], (N_STEPS, 1))
    segs = [(HOLD0, T_LIFT, 0, 1), (HOLD0 + T_LIFT, T_SWING, 1, 2), (HOLD0 + T_LIFT + T_SWING, T_LOWER, 2, 3)]
    for t0, T, a, b in segs:
        tau = np.clip((t - t0) / T, 0, 1); s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        mask = t >= t0
        ref[mask] = W[a] + (W[b] - W[a]) * s[mask, None]
    return ref


def degrade(h):
    """health (...,) -> aged joint-motor parameters.  h=1 new, H_MIN worn.  (ASSUMED coefficients)"""
    d = 1.0 - np.asarray(h, float); M = MOTOR
    return dict(R=M['R'] * (1 + 0.8 * d), Kt=M['Kt'] * (1 - 0.5 * d), Ke=M['Ke'] * (1 - 0.5 * d),
                Bm=M['Bm'] * (1 + 6 * d), tauC=1e-4 + 1.5e-3 * d, eff=M['eff'] - 0.25 * d)


def gravity_dir(q, mt, A=ARM):
    """Gravity torque acting in the +angle direction on each joint [N.m].  q: (B,3), mt: (B,) tool+payload mass."""
    p2, p3 = q[:, 1], q[:, 1] + q[:, 2]
    a = A['m3'] * A['lc3'] + mt * A['L3']
    g2 = A['g'] * ((A['m2'] * A['lc2'] + (A['m3'] + mt) * A['L2']) * np.sin(p2) + a * np.sin(p3))
    g3 = A['g'] * a * np.sin(p3)
    return np.stack([np.zeros_like(g2), g2, g3], axis=1)


def inertia_out(q, mt, A=ARM):
    p2, p3 = q[:, 1], q[:, 1] + q[:, 2]
    r2 = A['lc2'] * np.sin(p2); r3 = A['L2'] * np.sin(p2) + A['lc3'] * np.sin(p3); rt = A['L2'] * np.sin(p2) + A['L3'] * np.sin(p3)
    J1 = A['Jbase'] + A['m2'] * r2 ** 2 + A['m3'] * r3 ** 2 + mt * rt ** 2
    c3 = np.cos(q[:, 2])
    J2 = A['m2'] * A['lc2'] ** 2 + A['m3'] * (A['L2'] ** 2 + A['lc3'] ** 2 + 2 * A['L2'] * A['lc3'] * c3) + mt * (A['L2'] ** 2 + A['L3'] ** 2 + 2 * A['L2'] * A['L3'] * c3)
    J3 = A['m3'] * A['lc3'] ** 2 + mt * A['L3'] ** 2
    return np.stack([J1, J2, J3], axis=1)


def simulate(gains, mp, health, noise=NOISE_COUNTS, seed=0):
    """gains (B,3,3) [joint][Kp,Ki,Kd]; mp (B,) payload kg; health (B,3).
    Returns q (N,B,3) rad, V (N,B,3), ref (N,3)."""
    gains = np.asarray(gains, float); mp = np.asarray(mp, float); health = np.asarray(health, float)
    B = gains.shape[0]; M = MOTOR; A = ARM; dt = DT
    dg = degrade(health)                                   # each (B,3)
    ratio = M['ratio']; rpc = 2 * np.pi / M['cpr']
    ref = reference(); q0 = ref[0]
    mt = A['mg'] + mp
    w = np.zeros((B, 3)); th = np.zeros((B, 3))
    integ = np.zeros((B, 3)); prev = np.zeros((B, 3)); fd = np.zeros((B, 3))
    alpha = dt / (M['tau_f'] + dt)
    Kp, Ki, Kd = gains[:, :, 0], gains[:, :, 1], gains[:, :, 2]
    rng = np.random.default_rng(seed)
    qlog = np.empty((N_STEPS, B, 3)); Vlog = np.empty((N_STEPS, B, 3))
    R, Kt, Ke, Bm, tauC, eff = dg['R'], dg['Kt'], dg['Ke'], dg['Bm'], dg['tauC'], dg['eff']
    for n in range(N_STEPS):
        q = q0 + th / ratio
        qm = q + noise * rpc * rng.standard_normal((B, 3)) if noise else q
        err = ref[n] - np.round(qm / rpc) * rpc
        fd += alpha * ((err - prev) / dt - fd)
        tent = integ + err * dt
        u = Kp * err + Ki * tent + Kd * fd
        hi, lo = u > V_LIMIT, u < -V_LIMIT
        V = np.clip(u, -V_LIMIT, V_LIMIT)
        integ = np.where((hi & (err > 0)) | (lo & (err < 0)), integ, tent)
        prev = err
        tau_load = -gravity_dir(q, mt) / (ratio * eff)                  # reflected to the motor shaft
        Jm = M['Jm'] + inertia_out(q, mt) / ratio ** 2
        f = lambda ww: (Kt * (V - Ke * ww) / R - Bm * ww - tauC * np.tanh(ww / FRIC_W0) - tau_load) / Jm
        k1w = f(w); k2w = f(w + .5 * dt * k1w); k3w = f(w + .5 * dt * k2w); k4w = f(w + dt * k3w)
        k1t = w; k2t = w + .5 * dt * k1w; k3t = w + .5 * dt * k2w; k4t = w + dt * k3w
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        th = th + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        qlog[n] = q0 + th / ratio; Vlog[n] = V
    return qlog, Vlog, ref


def cycle_cost(q, V, ref):
    """(B,) cost, lower is better. Tracking + final accuracy + voltage chatter, averaged over joints."""
    e = np.degrees(ref[:, None, :] - q)                              # (N,B,3) deg
    N = e.shape[0]; tail = max(1, N // 10)
    track = np.mean(np.abs(e), axis=0) / 3.0
    fin = np.mean(np.abs(e[-tail:]), axis=0) / 1.0
    chat = np.mean(np.abs(np.diff(V, axis=0)), axis=0) / V_LIMIT
    return np.minimum(track + 2 * fin + 3 * chat, 6.0).mean(axis=1)


def metrics(q, V, ref):
    """dict of per-context metrics for reporting."""
    e = np.degrees(ref[:, None, :] - q)
    final = np.abs(e[-max(1, e.shape[0] // 20):]).mean(axis=0)       # (B,3) deg
    maxe = np.abs(e).max(axis=0)
    inband = np.abs(e) < 1.0                                          # within 1 deg
    # settle time = last time any joint is outside 1 deg, after the last reference move
    out_any = (~inband).any(axis=2)                                   # (N,B)
    idx = np.where(out_any.any(axis=0), N_STEPS - 1 - np.argmax(out_any[::-1], axis=0), 0)
    settle = (idx + 1) * DT
    settle = np.where(out_any[-1], np.nan, settle)
    return dict(final=final, maxe=maxe, settle=settle, ok=(final < 1.0).all(axis=1))


def capacity_features(health, mp):
    """holding-torque load fraction (needed / available) for J2 and J3 at the worst waypoint -> (B,2)."""
    health = np.asarray(health, float); mp = np.asarray(mp, float); B = len(mp); M = MOTOR
    dg = degrade(health); W = waypoints(); mt = ARM['mg'] + mp
    need = np.zeros((B, 3))
    for w in W:
        need = np.maximum(need, np.abs(gravity_dir(np.tile(w, (B, 1)), mt)))
    cap = M['V'] * dg['Kt'] / dg['R'] * M['ratio'] * dg['eff']
    return (need / cap)[:, 1:]


def features(mp, health):
    """Policy input (B,12): payload, per-joint (damage, loop-gain ratio, friction), holding load fractions J2/J3."""
    mp = np.asarray(mp, float); health = np.asarray(health, float); dg = degrade(health); M = MOTOR
    cols = [mp / MP_MAX]
    for j in range(3):
        cols += [(1 - health[:, j]) / (1 - H_MIN), (dg['Kt'][:, j] / dg['R'][:, j]) / (M['Kt'] / M['R']), dg['tauC'][:, j] / 1e-3]
    lf = capacity_features(health, mp)
    cols += [lf[:, 0], lf[:, 1]]
    return np.stack(cols, axis=1)


def export_config(path):
    W = waypoints()
    d = dict(MOTOR=MOTOR, ARM=ARM, DT=DT, H_MIN=H_MIN, FRIC_W0=FRIC_W0, MP_MAX=MP_MAX, V_LIMIT=V_LIMIT,
             HOLD0=HOLD0, T_LIFT=T_LIFT, T_SWING=T_SWING, T_LOWER=T_LOWER, T_SETTLE=T_SETTLE,
             WAYPOINTS=W.tolist(), R_REACH=R_REACH, Z_LOW=Z_LOW, Z_HIGH=Z_HIGH, SWING_DEG=SWING_DEG, PARTS=PARTS)
    json.dump(d, open(path, 'w'))


if __name__ == '__main__':
    W = waypoints(); print('waypoints [deg]\n', np.degrees(W).round(1)); print('N', N_STEPS, 'T', T_TOTAL)
    mp = np.array([0.0, 0.6, 1.5]); h = np.ones((3, 3))
    print('capacity fractions h=1', capacity_features(h, mp).round(2).tolist())
    print('capacity fractions h=.4', capacity_features(np.full((3, 3), .4), mp).round(2).tolist())
    G = np.tile([[30, 60, .1]] * 3, (3, 1, 1)); q, V, ref = simulate(G, mp, h, noise=0)
    m = metrics(q, V, ref); print('final err deg\n', m['final'].round(2), 'settle', m['settle'], 'cost', cycle_cost(q, V, ref).round(2))
