"""Held-out evaluation: RL health-aware vs RL tuned-when-new vs gain compensation vs fixed gains."""
import json, numpy as np, arm_sim as S, train_arm as T
def load(f):
    d = json.load(open(f)); P = T.Policy(np.random.default_rng(0)); 
    for k in ('W1','b1','W2','b2','W3','b3'): P.p[k] = np.array(d[k])
    return P
aware, nom = load('arm_policy_aware.json'), load('arm_policy_nominal.json')
fixed = T.policy_gains(nom, np.array([0.0]), np.ones((1, 3)), aware=False)[0]
def tip(q):  # (B,3) -> mm position
    p3 = q[:, 1] + q[:, 2]; r = S.ARM['L2']*np.sin(q[:, 1]) + S.ARM['L3']*np.sin(p3); y = S.ARM['z0'] + S.ARM['L2']*np.cos(q[:, 1]) + S.ARM['L3']*np.cos(p3) - S.ARM['lt']
    return np.stack([r*np.cos(q[:, 0]), y, -r*np.sin(q[:, 0])], 1)*1000
def tiperr(q, ref):
    tail = S.N_STEPS // 20; return np.linalg.norm(tip(q[-tail:].mean(0)) - tip(np.tile(ref[-1], (q.shape[1], 1))), axis=1)
def strategies(mp, h):
    n = len(mp); gn = T.policy_gains(nom, mp, h, aware=False)
    k = (S.degrade(h)['Kt'] / S.degrade(h)['R']) / (S.MOTOR['Kt'] / S.MOTOR['R'])
    return {'RL health-aware': T.policy_gains(aware, mp, h, True), 'RL tuned when new': gn,
            'Gain compensation': np.clip((gn / k[:, :, None]).reshape(n, 9), T.LO, T.HI).reshape(n, 3, 3), 'Fixed gains': np.tile(fixed, (n, 1, 1))}
rng = np.random.default_rng(5)
for label, mpf in [('random payloads 0-1.5 kg', lambda n: rng.uniform(0, 1.5, n)), ('heavy parts (1.2 / 1.5 kg)', lambda n: rng.choice([1.2, 1.5], n))]:
    print(f'\n== {label}: mean cost | % placed (<2 mm) | median tip error mm   (n=150 per health level)')
    for hv in (1.0, 0.8, 0.6, 0.4):
        n = 150; mp = mpf(n); h = np.full((n, 3), hv); out = []
        for name, G in strategies(mp, h).items():
            q, V, ref = S.simulate(G, mp, h, seed=3); c = S.cycle_cost(q, V, ref); e = tiperr(q, ref)
            out.append(f'{name}: {c.mean():.2f}|{100*np.mean(e<2):3.0f}%|{np.median(e):5.1f}')
        print(f'  h={hv}: ' + '   '.join(out))
