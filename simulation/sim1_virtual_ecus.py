"""Synthetic CAN voltage-fingerprinting simulation (NOT real hardware data).

What it does, step by step:
  1. Invents 8 "ECUs", each with its own CAN_H/CAN_L voltage level, noise and
     edge ringing (the fingerprint), and samples them like a 12-bit ADC.
  2. Cuts the samples into blocks of 1000 and computes 5 statistics
     (mean, variance, skewness, kurtosis, 5th moment) on CAN_H, CAN_L and
     their difference -> 15 features per block, as in Dini et al. [17].
  3. Trains 7 classifiers and compares accuracy, size and speed.
  4. Shifts all voltages by 0-60 mV to mimic temperature/supply drift.
  5. Adds 3 never-seen ECUs and tests open-set rejection.
  6. Simulates an attacker mixing spoofed frames with real ones and compares
     the anchor's 10-of-10 batch alarm with CUSUM.
  7. Checks the analytical model's predictions against the simulation.
  8. Tests CUSUM when some genuine frames are misclassified (false alarms).

Run:  python sim.py      (Windows)   or   python3 sim.py   (Mac/Linux)
Needs: numpy scipy scikit-learn   (matplotlib optional, for the .png charts)
Outputs: printed report (sections 1-6), results.json, results.png, drift_importance.png"""
import numpy as np, json, time, pickle
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

rng = np.random.default_rng(42)
N_ECU, OBS, BLK = 8, 400, 1000
VREF, BITS = 5.0, 12
LSB = VREF / 2**BITS

# per-ECU transceiver "fingerprint": dominant levels, noise, ringing/overshoot shape
ecu = []
for i in range(N_ECU):
    ecu.append(dict(
        vh=3.50 + rng.normal(0, 0.010), vl=1.50 + rng.normal(0, 0.010),
        sh=0.012 + rng.uniform(0, 0.0015), sl=0.012 + rng.uniform(0, 0.0015),
        ring=rng.uniform(0.045, 0.055),      # overshoot amplitude at edges
        tau=rng.uniform(7, 9)))           # ringing decay (samples)

def adc(x):
    return np.clip(np.round(x / LSB), 0, 2**BITS - 1) * LSB

def block(p, drift=0.0):
    # frame-level ground/supply jitter makes classes overlap (realism)
    g = rng.normal(0, 0.008) + drift
    n = BLK
    # dominant bit segments of ~10 samples each with edge ringing
    t = np.arange(n) % 10
    ringH = p['ring'] * np.exp(-t / p['tau']) * np.cos(1.3 * t)
    ringL = -0.8 * p['ring'] * np.exp(-t / p['tau']) * np.cos(1.3 * t)
    h = adc(p['vh'] + g + ringH + rng.normal(0, p['sh'], n))
    l = adc(p['vl'] + g * 0.6 + ringL + rng.normal(0, p['sl'], n))
    return h, l

def moments(x):
    m, v = x.mean(), x.var()
    s = x.std() + 1e-12
    hyper = np.mean(((x - m) / s) ** 5)
    return [m, v, skew(x), kurtosis(x, fisher=False), hyper]

def feats(h, l):
    return moments(h) + moments(l) + moments(h - l)

def make(params, nobs, drift=0.0):
    X, y = [], []
    for i, p in enumerate(params):
        for _ in range(nobs):
            X.append(feats(*block(p, drift))); y.append(i)
    return np.array(X), np.array(y)

X, y = make(ecu, OBS)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

models = {
    'Linear SVM': make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000)),
    'LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    'QDA': make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)),
    'Gaussian NB': make_pipeline(StandardScaler(), GaussianNB()),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0),
    'KNN (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
}
res = {}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in models.items():
    cvacc = cross_val_score(m, Xtr, ytr, cv=cv).mean()
    m.fit(Xtr, ytr)
    t0 = time.perf_counter()
    for _ in range(20): pred = m.predict(Xte)
    dt = (time.perf_counter() - t0) / 20 / len(Xte) * 1e6
    size = len(pickle.dumps(m)) / 1024
    res[name] = dict(cv=round(cvacc*100,1), test=round(accuracy_score(yte, pred)*100,1),
                     us=round(dt,2), kb=round(size,1))

# feature importance (mean+variance vs higher-order) via RF
rf = models['Random Forest']
imp = rf.feature_importances_
order = ['mean','var','skew','kurt','hyper']
imp_by_moment = {o: round(float(sum(imp[k] for k in range(15) if k % 5 == j))*100,1) for j,o in enumerate(order)}

# drift test: train at nominal, test with supply/temperature offset
lsvm = models['Linear SVM']
drift = {}
for d_mv in [0, 10, 20, 40, 60]:
    Xd, yd = make(ecu, 100, drift=d_mv/1000)
    drift[d_mv] = round(accuracy_score(yd, lsvm.predict(Xd))*100,1)

# --- Extension 1: open-set (foreign device) detection ---
# closed-set SVM forcibly labels an unknown ECU; distance-to-class-centroid rejects it
sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr)
lda = LinearDiscriminantAnalysis(store_covariance=True).fit(Ztr, ytr)
cov_inv = np.linalg.pinv(lda.covariance_)
cent = np.array([Ztr[ytr==k].mean(0) for k in range(N_ECU)])
def mdist(Z):
    d = np.array([np.einsum('ij,jk,ik->i', Z-c, cov_inv, Z-c) for c in cent]).T
    return d.min(1)
thr = np.percentile(mdist(Ztr), 99)
foreign = [dict(vh=3.5+rng.normal(0,0.010), vl=1.5+rng.normal(0,0.010),
                sh=0.012+rng.uniform(0,0.0015), sl=0.012+rng.uniform(0,0.0015),
                ring=rng.uniform(0.045,0.055), tau=rng.uniform(7,9)) for _ in range(3)]
Xf, _ = make(foreign, 150)
known_reject = float((mdist(sc.transform(Xte)) > thr).mean())
foreign_reject = float((mdist(sc.transform(Xf)) > thr).mean())
openset = dict(closed_set_foreign_detect=0.0,
               foreign_detect=round(foreign_reject*100,1), known_false_reject=round(known_reject*100,1))

# --- Extension 2: intermittent spoofing — batch L=10 buffer vs CUSUM ---
# ECU 3 is impersonated by ECU 6 on a fraction r of frames, interleaved
def frame_mismatch_prob(attacker, victim):
    Xa, _ = make([ecu[attacker]], 300); Xv, _ = make([ecu[victim]], 300)
    pa = float((lsvm.predict(Xa) != victim).mean()); pv = float((lsvm.predict(Xv) != victim).mean())
    return pa, pv
pa, pv = frame_mismatch_prob(6, 3)
def simulate(rate, trials=300, frames=500, L=10, k=0.2, h=5.0):
    det_b = det_c = 0; delay_c = []
    for _ in range(trials):
        mism = []
        for f in range(frames):
            spoof = rng.random() < rate
            mism.append(rng.random() < (pa if spoof else pv))
        # batch rule: non-overlapping batches of L, alarm if all L mismatches
        b = any(all(mism[i:i+L]) for i in range(0, frames - L + 1, L))
        # CUSUM on mismatch indicator
        S = 0; c = False
        for f, m in enumerate(mism):
            S = max(0.0, S + (1 if m else 0) - k)
            if S > h: c = True; delay_c.append(f); break
        det_b += b; det_c += c
    return round(det_b/trials*100,1), round(det_c/trials*100,1)
spoof = {}
for r in [0.0, 0.1, 0.3, 0.5, 1.0]:
    spoof[r] = simulate(r)

out = dict(models=res, imp_by_moment=imp_by_moment, drift=drift, openset=openset,
           spoof=spoof, per_frame=dict(attacker_mismatch=round(pa*100,1), legit_mismatch=round(pv*100,1)),
           n_obs=int(len(X)), n_feat=int(X.shape[1]))
cm = confusion_matrix(yte, lsvm.predict(Xte)).tolist(); out['cm'] = cm
json.dump(out, open('results.json','w'), indent=1)

# ---------------- readable report (numbers used on slides 11, 14-16, 22) ----------------
line = '-' * 72
print(line); print('SYNTHETIC CAN VOLTAGE-FINGERPRINTING SIMULATION  (not hardware data)')
print(f'{N_ECU} ECUs, {len(X)} blocks of {BLK} dominant-bit samples, {X.shape[1]} features each')
print(line); print('1) WHICH CLASSIFIER?  (slide 14)')
print(f'{"Classifier":<16}{"CV acc %":>10}{"Test acc %":>12}{"Size kB":>10}{"us/block":>10}')
for n, r in res.items():
    print(f'{n:<16}{r["cv"]:>10}{r["test"]:>12}{r["kb"]:>10}{r["us"]:>10}')
print(line); print('2) DRIFT: linear SVM trained at nominal, tested with an offset  (slide 15)')
for d, a in drift.items(): print(f'   offset {d:>3} mV -> accuracy {a}%')
print('   Feature importance by moment (random forest):', imp_by_moment)
print(line); print('3) UNKNOWN DEVICE (3 ECUs never seen in training)  (slide 16)')
print(f'   closed-set SVM flags {openset["closed_set_foreign_detect"]}% of foreign blocks')
print(f'   open-set rejection flags {openset["foreign_detect"]}%, '
      f'wrongly rejects {openset["known_false_reject"]}% of known-ECU blocks')
print(line); print('4) INTERLEAVED SPOOFING: ECU 6 impersonates ECU 3  (slide 16)')
print(f'   per-frame mismatch: spoofed frames {pa*100:.1f}%, genuine frames {pv*100:.1f}%')
print(f'{"   spoof rate":<16}{"batch L=10 %":>14}{"CUSUM %":>10}')
for r, (b, c) in spoof.items(): print(f'   {int(r*100):>3}%{"":<9}{b:>14}{c:>10}')
print(line)

# ---------------- 5) analytical model vs simulation (slide 11) ----------------
# Batch rule: P_D = 1 - (1 - q^L)^(F/L);  CUSUM: detects if q > k, delay ~ h/(q-k)
# q = rho*pa + (1-rho)*pv  is the chance a frame is flagged as a mismatch
L_, F_, k_, h_ = 10, 500, 0.2, 5.0
print('5) ANALYTICAL MODEL vs SIMULATION  (slide 11)')
print(f'   pa = {pa:.2f}, pv = {pv:.2f}, threshold rho* = (k - pv)/(pa - pv) = {(k_-pv)/(pa-pv):.2f}')
print(f'{"   spoof rate":<16}{"batch pred %":>14}{"batch sim %":>13}{"CUSUM pred":>20}{"CUSUM sim %":>13}')
for r in [0.1, 0.3, 0.5, 1.0]:
    q = r*pa + (1-r)*pv
    pb = (1 - (1 - q**L_)**(F_/L_)) * 100
    cp = f'yes, ~{h_/(q-k_):.0f} frames' if q > k_ else 'no (below rho*)'
    print(f'   {int(r*100):>3}%{"":<9}{pb:>14.2f}{spoof[r][0]:>13}{cp:>20}{spoof[r][1]:>13}')
print(line)

# ---------------- 6) robustness: genuine frames sometimes misclassified (slide 22) ----------------
# What if the classifier wrongly flags a fraction pv of GENUINE frames? Does CUSUM false-alarm?
print('6) CUSUM ROBUSTNESS when genuine frames are misclassified  (slide 22)')
print('   detection % per 500-frame run (2000 runs); spoofed frames flagged 95% of the time')
print(f'{"   pv":<10}{"k":>5}{"rho*":>7}{"0%":>8}{"10%":>8}{"20%":>8}{"30%":>8}{"50%":>8}   <- spoof rate (0% = false alarm)')
rng2 = np.random.default_rng(1)
def cusum_run(rho, pa_, pv_, k, h=5, F=500, T=2000):
    det = 0
    for _ in range(T):
        sp = rng2.random(F) < rho
        m = np.where(sp, rng2.random(F) < pa_, rng2.random(F) < pv_)
        S = 0
        for x in m:
            S = max(0, S + x - k)
            if S > h: det += 1; break
    return det / T * 100
robust = {}
for pv_ in [0.0, 0.05, 0.10]:
    for k in [0.2, 0.3]:
        row = [round(cusum_run(r, 0.95, pv_, k), 1) for r in [0, 0.1, 0.2, 0.3, 0.5]]
        robust[f'pv={pv_},k={k}'] = row
        print(f'   {pv_:<7}{k:>5}{(k-pv_)/(0.95-pv_):>7.2f}' + ''.join(f'{v:>8}' for v in row))
print(line)

# ---------------- optional charts (needs matplotlib) ----------------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = list(res); ax[0].barh(names, [res[n]['test'] for n in names], color='#2D5B8C')
    ax[0].set_xlim(80, 100); ax[0].set_xlabel('Test accuracy (%)'); ax[0].set_title('Classifier comparison')
    ax[1].plot(list(drift), list(drift.values()), 'o-', color='#B4530F')
    ax[1].set_xlabel('Common-mode offset (mV)'); ax[1].set_ylabel('Accuracy (%)'); ax[1].set_title('Drift')
    rates = [int(r*100) for r in spoof]
    ax[2].plot(rates, [v[0] for v in spoof.values()], 's-', color='#9AA5B4', label='Batch rule (L=10)')
    ax[2].plot(rates, [v[1] for v in spoof.values()], 'o-', color='#B4530F', label='CUSUM')
    ax[2].set_xlabel('Spoofed frames (%)'); ax[2].set_ylabel('Detection rate (%)')
    ax[2].set_title('Interleaved spoofing'); ax[2].legend()
    fig.suptitle('Synthetic simulation - not hardware data'); fig.tight_layout()
    fig.savefig('results.png', dpi=150); print('Charts saved to results.png')
    # second figure: drift and feature importance (slide 15)
    fig2, bx = plt.subplots(1, 2, figsize=(13, 4.2))
    offs = [f'{d} mV' for d in drift]; acc = list(drift.values())
    bx[0].barh(offs[::-1], acc[::-1], color=['#2D5B8C' if a >= 75 else '#B4530F' for a in acc[::-1]])
    for i, a in enumerate(acc[::-1]): bx[0].text(a + 1, i, f'{a}%', va='center')
    bx[0].set_xlim(0, 105); bx[0].set_xlabel('Linear SVM accuracy (%)')
    bx[0].set_title('Accuracy vs. common-mode voltage offset')
    labels = {'mean': 'Mean', 'var': 'Variance', 'skew': 'Skewness', 'kurt': 'Kurtosis', 'hyper': '5th moment'}
    items = sorted(imp_by_moment.items(), key=lambda t: t[1])
    bx[1].barh([labels[n] for n, _ in items], [v for _, v in items],
               color=['#2D5B8C' if n in ('mean', 'var') else '#9AA5B4' for n, _ in items])
    for i, (_, v) in enumerate(items): bx[1].text(v + 0.8, i, f'{v}%', va='center')
    bx[1].set_xlim(0, 55); bx[1].set_xlabel('Share of random-forest importance (%)')
    bx[1].set_title('Which statistics matter (3 channels summed)')
    fig2.suptitle('Synthetic simulation - not hardware data'); fig2.tight_layout()
    fig2.savefig('drift_importance.png', dpi=150); print('Charts saved to drift_importance.png')
except ImportError:
    print('Install matplotlib to also get results.png')
