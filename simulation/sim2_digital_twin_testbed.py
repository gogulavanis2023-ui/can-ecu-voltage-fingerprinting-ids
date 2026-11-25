"""Digital twin of the Project-1 CAN testbed + guardian (SIMULATION, not hardware data).

Review 2's sim.py modelled ECUs as random voltage blocks. This one models the
actual bench we built, bit by bit, and runs the full guardian pipeline on it:

  Bench being modelled
    * 125 kbit/s CAN bus, MCP2515 + TJA1050 modules on Arduino Nanos
    * N1 -> ID 0x100 (every 20 ms), N2 -> 0x200 (40 ms), N3 -> 0x300 (100 ms)
    * ATK  : attacker Nano (hardware the guardian has never seen)
    * FOR  : a second foreign device (e.g. an OBD dongle)
    * Guardian: CAN_H/CAN_L -> 100k/100k divider -> CA3130 buffer -> 12-bit ADC
      at 1 MS/s per channel (STM32F103 dual-ADC mode) = 8 samples per bit

  Pipeline (same 8 blocks as the Review 2 block diagram)
    1 dual ADC  2 frame sync + bit decoding (ID read from the waveform itself)
    3 15 moment features  4 linear SVM  5 open-set (Mahalanobis) check
    6 ID-owner check  7 CUSUM per ID  8 alarm log

  Timeline (65 s of bus traffic)
    0-15 s  enrolment (genuine traffic only, guardian learns fingerprints)
    15-25   normal
    25-35   ATTACK A  masquerade: N1 silenced, ATK sends 0x100
    35-45   normal
    45-55   ATTACK B  interleaved spoof: compromised N3 injects extra 0x100
            frames (30 % of all 0x100 traffic) while N1 keeps running
    55-65   ATTACK C  foreign device: N3 silenced, FOR sends 0x300

Run:  python sim_testbed.py        Needs: numpy scipy scikit-learn matplotlib
Outputs go to ./out_testbed/ (PNG figures, guardian_log.txt, results.json)

All device parameters are drawn from TJA1050 datasheet ranges; results depend
on them, so present the numbers as simulation results, not measurements.
"""
import json, os, time
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

rng = np.random.default_rng(7)
OUT = 'out_testbed'; os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- bus + ADC
BITRATE = 125_000
FS_SIM = 8_000_000                 # internal analog simulation rate
OS = FS_SIM // BITRATE             # 64 sim samples per bit
FS_ADC = 1_000_000                 # guardian ADC rate per channel
DEC = FS_SIM // FS_ADC             # 8 sim samples per ADC sample
SPB = OS // DEC                    # 8 ADC samples per bit
VR = 2.5                           # recessive level (both lines)
DIV = 0.5                          # 100k/100k divider before CA3130
VREF, BITS = 3.3, 12
LSB = VREF / 2**BITS
DOM_THR = 0.9                      # guardian: CANH-CANL above this = dominant

def make_device(name):
    """One TJA1050 module: dominant levels, ground offset and edge response."""
    return dict(name=name,
                vh=3.55 + rng.normal(0, 0.025),     # CAN_H dominant (V)
                vl=1.45 + rng.normal(0, 0.025),     # CAN_L dominant (V)
                gnd=rng.normal(0, 0.004),           # node ground offset (V)
                fn=rng.uniform(280e3, 360e3),       # edge response (Hz)
                zeta=rng.uniform(0.55, 0.75))       # edge damping

DEV = {n: make_device(n) for n in ['N1', 'N2', 'N3', 'ATK', 'FOR']}
GENUINE = ['N1', 'N2', 'N3']
OWNER = {0x100: 'N1', 0x200: 'N2', 0x300: 'N3'}

def edge_filter(dev):
    wn = 2 * np.pi * dev['fn']
    b, a = signal.bilinear([wn**2], [1, 2 * dev['zeta'] * wn, wn**2], fs=FS_SIM)
    return b, a
for d in DEV.values():
    d['ba'] = edge_filter(d)

# ---------------------------------------------------------------- CAN frames
def crc15(bits):
    crc = 0
    for b in bits:
        nxt = b ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if nxt:
            crc ^= 0x4599
    return [(crc >> i) & 1 for i in range(14, -1, -1)]

def stuff(bits):
    out, run, last = [], 0, None
    for b in bits:
        out.append(b)
        run = run + 1 if b == last else 1
        last = b
        if run == 5:
            out.append(1 - b); last = 1 - b; run = 1
    return out

def build_frame(can_id, data):
    bits = [0] + [(can_id >> i) & 1 for i in range(10, -1, -1)] + [0, 0, 0]
    bits += [(len(data) >> i) & 1 for i in range(3, -1, -1)]
    for byte in data:
        bits += [(byte >> i) & 1 for i in range(7, -1, -1)]
    bits += crc15(bits)
    st = stuff(bits)
    lead = 4
    full = [1] * lead + st + [1, 0, 1] + [1] * 10   # CRC delim, ACK, ACK delim, EOF+IFS
    return np.array(full), lead + len(st) + 1         # index of ACK slot

def frame_waveform(dev, can_id, data):
    """Analog CAN_H/CAN_L as seen by the guardian ADC (bus volts after rescaling)."""
    bits, ack = build_frame(can_id, data)
    dom = np.repeat(bits == 0, OS)
    ack_mask = np.zeros_like(dom); ack_mask[ack * OS:(ack + 1) * OS] = True
    amp = 1 + rng.normal(0, 0.0015)                  # supply ripple this frame
    g = rng.normal(0, 0.003) + dev['gnd']           # common-mode / ground shift
    th = np.where(dom, (dev['vh'] - VR) * amp, 0.0)
    tl = np.where(dom, (dev['vl'] - VR) * amp, 0.0)
    # ACK bit is driven by the receivers, not the sender
    th[ack_mask] = 1.07; tl[ack_mask] = -1.07
    b, a = dev['ba']
    h = VR + g + signal.lfilter(b, a, th)
    l = VR + g + signal.lfilter(b, a, tl)
    ph = rng.integers(0, DEC)                       # ADC not synced to bit edges
    h, l = h[ph::DEC], l[ph::DEC]
    def adc(v):                                     # divider + buffer noise + 12-bit
        code = np.clip(np.round((DIV * v + rng.normal(0, 0.0015, v.size)) / LSB), 0, 2**BITS - 1)
        return code * LSB / DIV
    return adc(h), adc(l)

# ---------------------------------------------------------------- guardian
def guardian_decode(h, l):
    """Frame sync + bit decoding from the samples. Returns ID and the samples
    that belong to dominant bits from SOF to end of CRC (ACK excluded)."""
    dom = (h - l) > DOM_THR
    sof = int(np.argmax(dom))
    raw, bits, run, last, k, need = [], [], 0, None, 0, None
    while True:
        c = sof + SPB // 2 + SPB * k
        if c >= dom.size:
            return None
        b = 0 if dom[c] else 1
        raw.append(b); k += 1
        if run == 5:                                # stuff bit: skip
            run, last = 1, b; continue
        bits.append(b)
        run = run + 1 if b == last else 1
        last = b
        if len(bits) == 19 and need is None:
            dlc = min(int(''.join(map(str, bits[15:19])), 2), 8)
            need = 19 + 8 * dlc + 15
        if need and len(bits) == need:
            break
    if crc15(bits[:-15]) != bits[-15:]:
        return None                                 # CRC error -> drop frame
    can_id = int(''.join(map(str, bits[1:12])), 2)
    idx = np.concatenate([np.arange(sof + SPB * i, sof + SPB * (i + 1))
                          for i, b in enumerate(raw) if b == 0])
    idx = idx[idx < h.size]
    return can_id, h[idx], l[idx]

def moments(x):
    m, s = x.mean(), x.std() + 1e-12
    return [m, x.var(), skew(x), kurtosis(x, fisher=False), np.mean(((x - m) / s) ** 5)]

def features(h, l):
    return moments(h) + moments(l) + moments(h - l)

# ---------------------------------------------------------------- traffic
PHASES = [(0, 15, 'Enrolment'), (15, 25, 'Normal'), (25, 35, 'Attack A: masquerade'),
          (35, 45, 'Normal'), (45, 55, 'Attack B: interleaved spoof'),
          (55, 65, 'Attack C: foreign device')]
T_END = 65.0

def schedule():
    ev = []  # (time, sender, id)
    for sender, cid, per in [('N1', 0x100, 0.020), ('N2', 0x200, 0.040), ('N3', 0x300, 0.100)]:
        t = rng.uniform(0, per)
        while t < T_END:
            silenced = (sender == 'N1' and 25 <= t < 35) or (sender == 'N3' and 55 <= t < 65)
            if not silenced:
                ev.append((t, sender, cid))
            t += per
    t = 25.0                                        # A: ATK takes over 0x100
    while t < 35:
        ev.append((t, 'ATK', 0x100)); t += 0.020
    t = 45.0                                        # B: N3 injects 0x100 (30 % share)
    while t < 55:
        ev.append((t, 'N3', 0x100)); t += rng.exponential(1 / 21.4)
    t = 55.0                                        # C: FOR sends 0x300
    while t < 65:
        ev.append((t, 'FOR', 0x300)); t += 0.100
    return sorted(ev)

print('Simulating bus traffic and running the guardian on every frame...')
t0 = time.time()
events = schedule()
frames = []
counter = {}
decode_ok = 0
for t, snd, cid in events:
    counter[cid] = (counter.get(cid, 0) + 1) & 0xFF
    data = [counter[cid]] + list(rng.integers(0, 256, 7))
    h, l = frame_waveform(DEV[snd], cid, data)
    dec = guardian_decode(h, l)
    if dec is None:
        continue
    rid, hs, ls = dec
    decode_ok += (rid == cid)
    frames.append(dict(t=t, sender=snd, id=rid, f=features(hs, ls), n=hs.size))
print(f'  {len(frames)} frames simulated and decoded in {time.time()-t0:.1f} s')
ID_ACC = decode_ok / len(frames) * 100

X = np.array([f['f'] for f in frames]); T = np.array([f['t'] for f in frames])
S = np.array([f['sender'] for f in frames]); I = np.array([f['id'] for f in frames])

# ---------------------------------------------------------------- training (enrolment)
tr = T < 15
clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=50000, class_weight='balanced')).fit(X[tr], S[tr])
sc = clf.named_steps['standardscaler']
Ztr = sc.transform(X[tr])
lda = LinearDiscriminantAnalysis(store_covariance=True).fit(Ztr, S[tr])
cov_inv = np.linalg.pinv(lda.covariance_)
cent = {c: Ztr[S[tr] == c].mean(0) for c in GENUINE}
def mdist(Z):
    return np.min([np.einsum('ij,jk,ik->i', Z - cent[c], cov_inv, Z - cent[c]) for c in GENUINE], axis=0)
THR = np.percentile(mdist(Ztr), 99.5)

pred = clf.predict(X)
dist = mdist(sc.transform(X))
unknown = dist > THR
owner = np.array([OWNER[i] for i in I])
mismatch = unknown | (pred != owner)

normal = ((T >= 15) & (T < 25)) | ((T >= 35) & (T < 45))
genuine_test = (T >= 15) & np.isin(S, GENUINE) & (S == owner) & ~((T >= 45) & (T < 55) & (S == 'N3') & (I == 0x100))
closed_acc = accuracy_score(S[genuine_test], pred[genuine_test]) * 100
cm = confusion_matrix(S[genuine_test], pred[genuine_test], labels=GENUINE)
foreign = np.isin(S, ['ATK', 'FOR'])
open_detect = unknown[foreign].mean() * 100
false_reject = unknown[genuine_test].mean() * 100

# ---------------------------------------------------------------- CUSUM + batch rule, log
K, H, L = 0.25, 6.0, 10
CAP = 2 * H                         # bounded CUSUM so alarms clear after an attack
cus = {i: 0.0 for i in OWNER}; run = {i: 0 for i in OWNER}
cus_trace = {i: [] for i in OWNER}
alarms_c, alarms_b = [], []
in_alarm = {i: False for i in OWNER}
log = []
for j in np.argsort(T):
    if T[j] < 15:
        continue
    cid = I[j]; m = mismatch[j]
    cus[cid] = min(CAP, max(0.0, cus[cid] + (1 if m else 0) - K))
    cus_trace[cid].append((T[j], cus[cid]))
    run[cid] = run[cid] + 1 if m else 0
    who = 'UNKNOWN' if unknown[j] else pred[j]
    line = f'{T[j]:7.3f} s  ID 0x{cid:03X}  sender={who:<7} owner={OWNER[cid]}  CUSUM={cus[cid]:5.2f}'
    if cus[cid] > H and not in_alarm[cid]:
        in_alarm[cid] = True
        kind = 'UNKNOWN DEVICE' if unknown[j] else f'SENDER MISMATCH (physically {pred[j]})'
        alarms_c.append(dict(t=float(T[j]), id=int(cid), kind=kind))
        line += f'   *** ALERT: {kind} on ID 0x{cid:03X} ***'
    elif cus[cid] == 0 and in_alarm[cid]:
        in_alarm[cid] = False
        line += '   (alarm cleared)'
    if run[cid] == L:
        alarms_b.append(dict(t=float(T[j]), id=int(cid)))
    log.append(line)
with open(f'{OUT}/guardian_log.txt', 'w') as fh:
    fh.write('Simulated guardian output (one line per CAN frame)\n')
    fh.write('\n'.join(log))

def first_after(alarms, t0, t1, cid):
    hits = [a['t'] for a in alarms if t0 <= a['t'] < t1 + 0.5 and a['id'] == cid]
    return round(hits[0] - t0, 3) if hits else None

attacks = {
    'A: masquerade (ATK as N1, 0x100)': (25, 35, 0x100),
    'B: interleaved spoof (N3 injects 0x100, 30%)': (45, 55, 0x100),
    'C: foreign device (FOR as N3, 0x300)': (55, 65, 0x300),
}
det = {k: dict(cusum_s=first_after(alarms_c, *v), batch_s=first_after(alarms_b, *v)) for k, v in attacks.items()}
false_alarms = sum(1 for a in alarms_c if (15 <= a['t'] < 25) or (35.5 <= a['t'] < 45))

# ---------------------------------------------------------------- embedded footprint
n_feat = X.shape[1]
svm = clf.named_steps['linearsvc']
params = 2 * n_feat + svm.coef_.size + svm.intercept_.size + len(GENUINE) * n_feat + n_feat * n_feat + 1
model_bytes = params * 4
mean_samples = float(np.mean([f['n'] for f in frames]))
ops = mean_samples * 3 * 9 + 3 * 40 + svm.coef_.size + len(GENUINE) * (n_feat * n_feat + n_feat)
fps = len([f for f in frames if 15 <= f['t'] < 25]) / 10
cpu = ops * fps * 4 / 72e6 * 100        # assume ~4 cycles/op fixed point on Cortex-M3

res = dict(frames=len(frames), id_decode_acc=round(ID_ACC, 2),
           closed_set_acc=round(closed_acc, 2), open_set_detect=round(open_detect, 2),
           genuine_false_reject=round(false_reject, 2), false_alarms_normal=false_alarms,
           detection=det, alarms=alarms_c, model_bytes=int(model_bytes),
           dominant_samples_per_frame=round(mean_samples), ops_per_frame=int(ops),
           frames_per_s=fps, est_cpu_load_pct=round(cpu, 1), confusion=cm.tolist(),
           devices={k: {kk: round(float(vv), 4) for kk, vv in v.items() if kk in ('vh', 'vl', 'gnd', 'fn', 'zeta')} for k, v in DEV.items()})
json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)

# ---------------------------------------------------------------- report
ln = '-' * 74
print(ln); print('DIGITAL TWIN OF THE CAN TESTBED + GUARDIAN  (simulation, not hardware data)'); print(ln)
print(f'Frames on the bus                     {len(frames)}  ({fps:.0f} frames/s)')
print(f'ID read from waveform (no MCP2515)    {ID_ACC:.2f} % correct')
print(f'Sender identification (N1/N2/N3)     {closed_acc:.2f} %')
print(f'Unknown hardware flagged (ATK, FOR)   {open_detect:.2f} % of frames')
print(f'Genuine frames wrongly flagged        {false_reject:.2f} %')
print(f'False alarms in normal phases         {false_alarms}')
print(ln); print(f'{"Attack":<48}{"CUSUM":>12}{"10-of-10":>12}')
for k, v in det.items():
    c = f'{v["cusum_s"]} s' if v['cusum_s'] is not None else 'missed'
    b = f'{v["batch_s"]} s' if v['batch_s'] is not None else 'missed'
    print(f'{k:<48}{c:>12}{b:>12}')
print(ln)
print(f'Model size on MCU                     {model_bytes} bytes ({params} float parameters)')
print(f'Work per frame                        ~{int(ops)} ops on ~{mean_samples:.0f} dominant samples')
print(f'Estimated CPU load, STM32F103 72 MHz  ~{cpu:.1f} % (fixed point, ~4 cycles/op assumed)')
print(ln)

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.8,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
COL = {'N1': '#2a78d6', 'N2': '#eb6834', 'N3': '#1baf7a', 'ATK': '#0b0b0b', 'FOR': '#8a8984'}
MK = {'N1': 'o', 'N2': 'o', 'N3': 'o', 'ATK': 'x', 'FOR': '^'}
TAG = 'Simulation - digital twin of the bench, not hardware data'

# Fig 1: same frame (0x100, same data) from N1 and from the attacker
data = [0x2A, 1, 2, 3, 4, 5, 6, 7]
ph_rng = rng.bit_generator.state
h1, l1 = frame_waveform(DEV['N1'], 0x100, data)
ha, la = frame_waveform(DEV['ATK'], 0x100, data)
tus = np.arange(h1.size) / FS_ADC * 1e6
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
for h, l, n in [(h1, l1, 'N1'), (ha, la, 'ATK')]:
    ax[0].plot(tus, h, color=COL[n], lw=1.6, label=f'{n} CAN_H')
    ax[0].plot(tus, l, color=COL[n], lw=1.6, ls='--', label=f'{n} CAN_L')
ax[0].set_xlim(20, 160); ax[0].set_xlabel('Time (µs)'); ax[0].set_ylabel('Bus voltage (V)')
ax[0].set_title('Same message ID 0x100, same data: genuine N1 vs attacker', loc='left')
ax[0].legend(ncol=4, frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, -0.32))
seg = (tus > 36) & (tus < 62)
for h, n in [(h1, 'N1'), (ha, 'ATK')]:
    ax[1].plot(tus[seg], h[seg] * 1000, color=COL[n], lw=1.8, marker='o', ms=4, label=n)
ax[1].set_xlabel('Time (µs)'); ax[1].set_ylabel('CAN_H (mV)')
ax[1].set_title('Zoom: dominant levels differ', loc='left')
ax[1].legend(frameon=False)
fig.text(0.99, 0.01, TAG, ha='right', fontsize=8, color='#52514e')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_waveforms.png', dpi=160); plt.close(fig)

# Fig 2: fingerprint space (LDA projection learnt on genuine ECUs)
P = lda.transform(sc.transform(X))
fig, ax = plt.subplots(figsize=(7, 5.4))
for n in ['N1', 'N2', 'N3', 'ATK', 'FOR']:
    m = (S == n) & ((T >= 15) | np.isin(n, GENUINE))
    m &= ~((S == 'N3') & (I == 0x100))
    sel = np.where(m)[0][:400]
    ax.scatter(P[sel, 0], P[sel, 1], s=14 if MK[n] == 'o' else 22, marker=MK[n],
               color=COL[n], alpha=0.75, linewidths=1.2,
               label={'ATK': 'ATK (unknown attacker)', 'FOR': 'FOR (unknown device)'}.get(n, f'{n} (genuine)'),
               edgecolors='#fcfcfb' if MK[n] == 'o' else None)
ax.set_xlabel('Fingerprint axis 1 (LDA)'); ax.set_ylabel('Fingerprint axis 2 (LDA)')
ax.set_title('Each module forms its own voltage cluster; strangers fall outside', loc='left', fontsize=10.5)
ax.legend(frameon=False, fontsize=9)
fig.text(0.99, 0.01, TAG, ha='right', fontsize=8, color='#52514e')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_fingerprints.png', dpi=160); plt.close(fig)

# Fig 3: attack timeline with CUSUM
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
for t0_, t1_, name in PHASES:
    for a in ax:
        if name.startswith('Attack'):
            a.axvspan(t0_, t1_, color='#f3d9d9', lw=0)
        elif name == 'Enrolment':
            a.axvspan(t0_, t1_, color='#ecebe6', lw=0)
    ax[0].text((t0_ + t1_) / 2, 1.02, name, transform=ax[0].get_xaxis_transform(),
               ha='center', va='bottom', fontsize=8.5, color='#52514e')
lanes = {'N1': 3, 'N2': 2, 'N3': 1, 'UNKNOWN': 0}
m100 = I == 0x100
for lab, y in lanes.items():
    sel = m100 & (np.where(unknown, 'UNKNOWN', pred) == lab)
    ax[0].scatter(T[sel], np.full(sel.sum(), y), s=6, marker='|',
                  color=COL.get(lab, '#e34948'))
ax[0].set_yticks(list(lanes.values())); ax[0].set_yticklabels(['N1 (owner)', 'N2', 'N3', 'Unknown'])
ax[0].set_ylim(-0.6, 3.6); ax[0].grid(axis='y', visible=False)
ax[0].set_title('Who the guardian says physically sent each 0x100 frame', loc='left', pad=18)
for cid, n in [(0x100, '0x100 (N1)'), (0x300, '0x300 (N3)')]:
    tt, ss = zip(*cus_trace[cid])
    ax[1].plot(tt, ss, lw=1.6, color=COL['N1'] if cid == 0x100 else COL['N3'], label=f'CUSUM for ID {n}')
ax[1].axhline(H, color='#52514e', lw=1, ls='--'); ax[1].text(15.2, H + 0.4, f'alarm threshold h = {H}', fontsize=8.5, color='#52514e')
for a in alarms_c:
    ax[1].annotate('ALERT', (a['t'], H), xytext=(a['t'] + 0.3, CAP + 1.5), fontsize=8.5,
                   arrowprops=dict(arrowstyle='-', color='#e34948'), color='#e34948', fontweight='bold')
ax[1].set_ylim(0, CAP + 4); ax[1].set_ylabel('CUSUM statistic'); ax[1].set_xlabel('Time (s)')
ax[1].legend(frameon=False, loc='upper left', fontsize=9)
ax[1].set_title('Evidence builds up only during attacks', loc='left')
ax[1].set_xlim(0, T_END)
fig.text(0.99, 0.005, TAG, ha='right', fontsize=8, color='#52514e')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_attack_timeline.png', dpi=160); plt.close(fig)

# Fig 4: confusion matrix (genuine frames after enrolment)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
cmn = cm / cm.sum(1, keepdims=True) * 100
ax.imshow(cmn, cmap='Blues', vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cmn[i, j]:.1f}%', ha='center', va='center',
                color='#fcfcfb' if cmn[i, j] > 60 else '#0b0b0b', fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(GENUINE); ax.set_yticks(range(3)); ax.set_yticklabels(GENUINE)
ax.set_xlabel('Predicted sender'); ax.set_ylabel('True sender'); ax.grid(False)
ax.set_title(f'Per-frame sender ID: {closed_acc:.1f}%', loc='left', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_confusion.png', dpi=160); plt.close(fig)
print(f'Figures, log and results saved in ./{OUT}/')
