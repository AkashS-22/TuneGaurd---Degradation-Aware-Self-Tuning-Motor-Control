"""RL auto-tuner for the servo PID twin (numpy only -- no PyTorch / Stable-Baselines needed).

What it learns
--------------
A policy  pi(payload) -> (Kp, Ki, Kd).  Each "episode" is one step response:
  1. sample a payload (arm length, mass, gravity on/off, motor speed limit)
  2. the policy proposes gains (Gaussian exploration around its mean)
  3. simulate the 90 deg step with those gains (same physics as the browser twin)
  4. reward = -cost(step response)      (one-step MDP == contextual bandit)
Trained with REINFORCE (policy gradient) + a per-payload baseline.
Output: autotune_policy.json  (tiny MLP weights, evaluated in plain JS by the browser page)

Run:  python train_autotune.py            (about 2-4 minutes on a laptop)
"""
import json, sys, time
import numpy as np

# ------------------------------------------------------------------ plant (mirrors the JS / Python twin, method_c)
CFG = dict(V_rated=12.0, ratio=180.0, eff=0.80, cpr=9360,
           R=1.6, Kt=1.974e-3, Ke=6.366e-3, Jm=5.0e-7, Bm=1.0e-6,
           dt=0.001, duration=2.0, ref_deg=90.0, tau_f=0.01)

NOISE_COUNTS = 1.0   # RMS encoder jitter in counts (0 = ideal encoder)

def simulate_batch(Kp, Ki, Kd, mass, length, gravity, vpct, ref_deg=None, cfg=CFG, noise=None, nrng=None):
    """Vectorised closed loop (encoder quantisation, filtered D, anti-windup, +-V saturation, RK4).
    Arguments are arrays of shape (B,). Returns true output angle [rad] (N,B) and voltage (N,B)."""
    Kp, Ki, Kd, mass, length, gravity, vpct = (np.asarray(x, float) for x in (Kp, Ki, Kd, mass, length, gravity, vpct))
    B = Kp.shape[0]
    dt = cfg['dt']; N = int(round(cfg['duration'] / dt))
    ref = np.radians(np.full(B, cfg['ref_deg']) if ref_deg is None else np.asarray(ref_deg, float))   # (B,) target angle
    ratio, eff, rpc = cfg['ratio'], cfg['eff'], 2 * np.pi / cfg['cpr']
    R, Kt, Ke, Bm = cfg['R'], cfg['Kt'], cfg['Ke'], cfg['Bm']
    Jm = cfg['Jm'] + mass * length ** 2 / ratio ** 2          # payload inertia reflected to the motor shaft
    vmax = cfg['V_rated'] * vpct / 100.0                        # "motor speed limit" = voltage cap
    alpha = dt / (cfg['tau_f'] + dt)
    w = np.zeros(B); th = np.zeros(B)
    integ = np.zeros(B); prev = np.zeros(B); fd = np.zeros(B)
    th_log = np.empty((N, B)); v_log = np.empty((N, B))
    noise = NOISE_COUNTS if noise is None else noise
    nrng = nrng if nrng is not None else np.random.default_rng(0)
    for n in range(N):
        th_true = th / ratio
        th_meas = th_true + noise * rpc * nrng.standard_normal(B)
        err = ref - np.round(th_meas / rpc) * rpc
        fd += alpha * ((err - prev) / dt - fd)
        tent = integ + err * dt
        u = Kp * err + Ki * tent + Kd * fd
        hi, lo = u > vmax, u < -vmax
        V = np.clip(u, -vmax, vmax)
        integ = np.where((hi & (err > 0)) | (lo & (err < 0)), integ, tent)   # conditional anti-windup
        prev = err
        tau = gravity * mass * 9.81 * length * np.sin(th_true) / (ratio * eff)  # arm gravity, 0 deg = hanging
        f = lambda ww: (Kt * (V - Ke * ww) / R - Bm * ww - tau) / Jm
        k1w = f(w);             k1t = w
        k2w = f(w + .5*dt*k1w); k2t = w + .5*dt*k1w
        k3w = f(w + .5*dt*k2w); k3t = w + .5*dt*k2w
        k4w = f(w + dt*k3w);    k4t = w + dt*k3w
        w = w + dt/6 * (k1w + 2*k2w + 2*k3w + k4w)
        th = th + dt/6 * (k1t + 2*k2t + 2*k3t + k4t)
        th_log[n] = th / ratio; v_log[n] = V
    return th_log, v_log

def step_cost(th_log, v_log, vpct, ref_deg, cfg=CFG):
    """Lower is better. Tracking (time-weighted) + overshoot + steady-state error + voltage chatter."""
    N = th_log.shape[0]; ref = np.radians(np.asarray(ref_deg, float)); refn = np.maximum(ref, np.radians(30.0))
    t = (np.arange(1, N + 1) / N)[:, None]
    e = (ref - th_log) / refn
    track = np.mean(np.abs(e) * (0.2 + t), axis=0)
    overshoot = np.maximum(0.0, th_log.max(axis=0) - ref) / refn
    sse = np.mean(np.abs(e[-N // 10:]), axis=0)
    chatter = np.mean(np.abs(np.diff(v_log, axis=0)), axis=0) / (cfg['V_rated'] * np.asarray(vpct) / 100.0)
    return np.minimum(4 * track + 3 * overshoot + 10 * sse + 1.5 * chatter, 6.0)

# ------------------------------------------------------------------ payload ranges = the sliders in the browser page
MASS_MAX, LEN_MIN, LEN_MAX, V_MIN = 2.0, 0.02, 0.15, 10.0
REF_MIN, REF_MAX = 5.0, 360.0
LO = np.array([0.5, 0.1, 0.005]); HI = np.array([50.0, 200.0, 5.0])       # Kp, Ki, Kd search range (log scale)

def sample_context(n, rng):
    mass = rng.uniform(0, MASS_MAX, n) * (rng.random(n) > 0.15)            # some no-payload cases
    length = rng.uniform(LEN_MIN, LEN_MAX, n)
    gravity = (rng.random(n) < 0.5).astype(float)
    vpct = rng.uniform(V_MIN, 100.0, n)
    ref = np.where(rng.random(n) < 0.25, 90.0, rng.uniform(REF_MIN, REF_MAX, n))   # target angle [deg]; 25% at the 90 deg default
    return mass, length, gravity, vpct, ref

def features(mass, length, gravity, vpct, ref_deg):
    """Policy input. MUST match the JS in the browser page."""
    return np.stack([mass / MASS_MAX,
                     (length - LEN_MIN) / (LEN_MAX - LEN_MIN),
                     gravity,
                     vpct / 100.0,
                     gravity * mass * length / (MASS_MAX * LEN_MAX),       # gravity torque on the arm
                     mass * length ** 2 / (MASS_MAX * LEN_MAX ** 2),            # payload inertia
                     ref_deg / 360.0,                                           # target angle
                     gravity * mass * length * np.sin(np.radians(ref_deg)) / (MASS_MAX * LEN_MAX)], axis=1)  # holding torque at target

def gains_from_z(z):
    u = 1.0 / (1.0 + np.exp(-z))
    return LO * (HI / LO) ** u

# ------------------------------------------------------------------ tiny MLP policy (6 -> 32 -> 32 -> 3) with manual backprop
class Policy:
    def __init__(self, rng, hidden=32):
        s = lambda a, b: rng.normal(0, np.sqrt(1.0 / a), (a, b))
        self.p = dict(W1=s(8, hidden), b1=np.zeros(hidden), W2=s(hidden, hidden), b2=np.zeros(hidden),
                      W3=s(hidden, 3) * 0.1,
                      b3=np.array([0.0, 0.83, -0.69]),                     # start near the default gains (5, 20, 0.05)
                      logstd=np.full(3, np.log(0.5)))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p['W1'] + p['b1']); h2 = np.tanh(h1 @ p['W2'] + p['b2'])
        return h2 @ p['W3'] + p['b3'], (X, h1, h2)

    def adam_ascent(self, g, lr):
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]; self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] += lr * mh / (np.sqrt(vh) + 1e-8)

    def grads(self, cache, dmu, dlogstd):
        X, h1, h2 = cache; p = self.p
        g = dict(W3=h2.T @ dmu, b3=dmu.sum(0), logstd=dlogstd)
        d2 = (dmu @ p['W3'].T) * (1 - h2 ** 2)
        g['W2'] = h1.T @ d2; g['b2'] = d2.sum(0)
        d1 = (d2 @ p['W2'].T) * (1 - h1 ** 2)
        g['W1'] = X.T @ d1; g['b1'] = d1.sum(0)
        return g

# ------------------------------------------------------------------ training
def train(iters=300, contexts=128, samples=8, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); pol = Policy(rng)
    t0 = time.time()
    for it in range(iters):
        m, L, g, v, R = sample_context(contexts, rng)
        X = features(m, L, g, v, R)
        mu, cache = pol.forward(X)                                            # (C,3)
        std = np.exp(pol.p['logstd'])
        eps = rng.normal(size=(contexts, samples, 3))
        z = mu[:, None, :] + std * eps                                        # (C,K,3) sampled actions
        gains = gains_from_z(z.reshape(-1, 3))
        rep = lambda a: np.repeat(a, samples)
        th, V = simulate_batch(gains[:, 0], gains[:, 1], gains[:, 2], rep(m), rep(L), rep(g), rep(v), rep(R))
        cost = step_cost(th, V, rep(v), rep(R)).reshape(contexts, samples)
        adv = -(cost - cost.mean(1, keepdims=True))                           # per-payload baseline
        adv /= adv.std() + 1e-8
        n = contexts * samples
        dmu = (adv[:, :, None] * eps / std).sum(1) / n                        # d logpi / d mu = eps/std
        dlogstd = (adv[:, :, None] * (eps ** 2 - 1)).sum((0, 1)) / n
        pol.adam_ascent(pol.grads(cache, dmu, dlogstd), lr)
        pol.p['logstd'] = np.clip(pol.p['logstd'], np.log(0.03), np.log(1.0))
        if it % 20 == 0 or it == iters - 1:
            print(f"iter {it:4d}  mean cost {cost.mean():.3f}  best-of-{samples} {cost.min(1).mean():.3f}  "
                  f"std {np.round(np.exp(pol.p['logstd']), 2)}  ({time.time() - t0:.0f}s)", flush=True)
    return pol

def policy_gains(pol, m, L, g, v, R):
    mu, _ = pol.forward(features(m, L, g, v, R))
    return gains_from_z(mu)

def evaluate(pol, n=1000, seed=123):
    rng = np.random.default_rng(seed); m, L, g, v, R = sample_context(n, rng)
    def score(G):
        th, V = simulate_batch(G[:, 0], G[:, 1], G[:, 2], m, L, g, v, R)
        return step_cost(th, V, v, R), th
    rows = {"RL auto-tune": policy_gains(pol, m, L, g, v, R),
            "default 5/20/0.05": np.tile([5, 20, 0.05], (n, 1)),
            "'Tuned' preset 15/0/0.3": np.tile([15, 0, 0.3], (n, 1))}
    print(f"\nHeld-out payloads + random targets 5-360 deg (n={n}):  mean cost | share within 2% of target at the end")
    for name, G in rows.items():
        c, th = score(G)
        ok = np.mean(np.abs(np.degrees(th[-1]) - R) < 0.02 * R)
        print(f"  {name:26s} {c.mean():6.3f} | {100 * ok:5.1f}%")
    # brute-force sanity on a few payload+target cases
    rs = np.random.default_rng(7); k = 12; m2, L2, g2, v2, R2 = sample_context(k, rs); cand = 400
    G = np.exp(rs.uniform(np.log(LO), np.log(HI), (cand, 3)))
    best = []
    for i in range(k):
        rp = lambda a: np.full(cand, a)
        th, V = simulate_batch(G[:, 0], G[:, 1], G[:, 2], rp(m2[i]), rp(L2[i]), rp(g2[i]), rp(v2[i]), rp(R2[i]))
        best.append(step_cost(th, V, rp(v2[i]), rp(R2[i])).min())
    Gp = policy_gains(pol, m2, L2, g2, v2, R2)
    th, V = simulate_batch(Gp[:, 0], Gp[:, 1], Gp[:, 2], m2, L2, g2, v2, R2)
    print(f"\nvs brute-force search ({cand} random gain sets per case, {k} cases): "
          f"brute-force best {np.mean(best):.3f}  |  RL policy {step_cost(th, V, v2, R2).mean():.3f}")


def export(pol, path="autotune_policy_360.json"):
    d = {k: np.asarray(v).tolist() for k, v in pol.p.items() if k != 'logstd'}
    d.update(lo=LO.tolist(), hi=HI.tolist(), mass_max=MASS_MAX, len_min=LEN_MIN, len_max=LEN_MAX)
    with open(path, "w") as f: json.dump(d, f)
    print("saved", path)

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    policy = train(iters=iters)
    evaluate(policy)
    export(policy)
