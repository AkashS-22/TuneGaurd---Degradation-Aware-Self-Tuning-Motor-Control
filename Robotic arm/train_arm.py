"""REINFORCE (contextual bandit) auto-tuner for the 3-axis arm.  numpy only.
Policy: (payload mass, per-joint health features, holding-load fractions) -> 9 PID gains (Kp,Ki,Kd for J1..J3).
One episode = one simulated pick-and-place cycle (arm_sim.simulate); reward = -cycle_cost.

  python train_arm.py aware   300    # health-aware: sees per-joint health, trained on degraded plants
  python train_arm.py nominal 300    # health-blind baseline: trained on new motors only (h=1), health features = 1
"""
import sys, json, time
import numpy as np
import arm_sim as S

LO1 = np.array([2.0, 0.5, 0.005]); HI1 = np.array([300.0, 600.0, 5.0])
LO = np.tile(LO1, 3); HI = np.tile(HI1, 3)
NIN = 12

def gains_from_z(z):
    u = 1.0 / (1.0 + np.exp(-z))
    return LO * (HI / LO) ** u

def sample_context(n, rng, degraded=True):
    mp = np.where(rng.random(n) < 0.35, rng.choice([m for _, m in S.PARTS], n), rng.uniform(0, S.MP_MAX, n))
    if not degraded:
        return mp, np.ones((n, 3))
    h = np.where(rng.random((n, 3)) < 0.25, 1.0, rng.uniform(S.H_MIN, 1.0, (n, 3)))
    uni = rng.random(n) < 0.25                                   # 25%: all joints aged equally
    h[uni] = rng.uniform(S.H_MIN, 1.0, (uni.sum(), 1)) * np.ones((1, 3))
    return mp, h

class Policy:
    def __init__(self, rng, hidden=48):
        s = lambda a, b: rng.normal(0, np.sqrt(1.0 / a), (a, b))
        b3 = np.tile([0.40, 0.73, -0.27], 3)                         # start near Kp~40, Ki~60, Kd~0.1
        self.p = dict(W1=s(NIN, hidden), b1=np.zeros(hidden), W2=s(hidden, hidden), b2=np.zeros(hidden),
                      W3=s(hidden, 9) * 0.1, b3=b3, logstd=np.full(9, np.log(0.5)))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0
    def forward(self, X):
        p = self.p; h1 = np.tanh(X @ p['W1'] + p['b1']); h2 = np.tanh(h1 @ p['W2'] + p['b2'])
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
        d2 = (dmu @ p['W3'].T) * (1 - h2 ** 2); g['W2'] = h1.T @ d2; g['b2'] = d2.sum(0)
        d1 = (d2 @ p['W2'].T) * (1 - h1 ** 2); g['W1'] = X.T @ d1; g['b1'] = d1.sum(0)
        return g

def policy_gains(pol, mp, h, aware=True):
    X = S.features(mp, h if aware else np.ones_like(h))
    mu, _ = pol.forward(X)
    return gains_from_z(mu).reshape(-1, 3, 3)

def train(mode, iters, contexts=64, samples=8, lr=3e-3, seed=0):
    aware = mode == 'aware'
    rng = np.random.default_rng(seed); pol = Policy(rng); t0 = time.time()
    for it in range(iters):
        mp, h = sample_context(contexts, rng, degraded=aware)
        X = S.features(mp, h)                                         # nominal: h=1 -> health features constant
        mu, cache = pol.forward(X); std = np.exp(pol.p['logstd'])
        eps = rng.normal(size=(contexts, samples, 9))
        z = mu[:, None, :] + std * eps
        gains = gains_from_z(z.reshape(-1, 9)).reshape(-1, 3, 3)
        rep = lambda a: np.repeat(a, samples, axis=0)
        q, V, ref = S.simulate(gains, rep(mp), rep(h), seed=it)
        cost = S.cycle_cost(q, V, ref).reshape(contexts, samples)
        adv = -(cost - cost.mean(1, keepdims=True)); adv /= adv.std() + 1e-8
        n = contexts * samples
        dmu = (adv[:, :, None] * eps / std).sum(1) / n
        dlogstd = (adv[:, :, None] * (eps ** 2 - 1)).sum((0, 1)) / n
        pol.adam_ascent(pol.grads(cache, dmu, dlogstd), lr)
        pol.p['logstd'] = np.clip(pol.p['logstd'], np.log(0.03), np.log(1.0))
        if it % 10 == 0 or it == iters - 1:
            print(f"[{mode}] iter {it:4d} mean cost {cost.mean():.3f} best-of-{samples} {cost.min(1).mean():.3f} "
                  f"std {np.round(np.exp(pol.p['logstd']).mean(), 2)} ({time.time()-t0:.0f}s)", flush=True)
        if it % 50 == 49: export(pol, f"arm_policy_{mode}.json")
    return pol

def export(pol, path):
    d = {k: np.asarray(v).tolist() for k, v in pol.p.items() if k != 'logstd'}
    d.update(lo=LO.tolist(), hi=HI.tolist(), mp_max=S.MP_MAX)
    json.dump(d, open(path, 'w'))

if __name__ == '__main__':
    mode = sys.argv[1]; iters = int(sys.argv[2])
    pol = train(mode, iters); export(pol, f"arm_policy_{mode}.json"); print('saved', mode)
