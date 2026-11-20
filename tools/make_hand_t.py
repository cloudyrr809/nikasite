"""Makes hand_t/NN.png: the maps that let "Ну и самое главное" be written by hand on the page.

Each map belongs to one letter. For every ink pixel of hand2.png it holds
  grey  = when the pen passes that pixel (0 = the letter's first moment, 255 = its last),
  alpha = how much of the pixel is left after the pencil's pressure (lighter press = thinner, paler line).
A pixel belongs to ONE pass of the pen (the first one that covers it), so at crossings and where the pen runs
back over itself nothing shows up before the pen gets there.

Run from the site folder:  python tools/make_hand_t.py .
It prints the HAND_T line for index.html (box of each map, start and length of each letter, ms)."""
import json, sys, os
import numpy as np
from PIL import Image
from scipy import ndimage as nd
from scipy.spatial import cKDTree

root = sys.argv[1] if len(sys.argv) > 1 else '.'
HAND = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_route.json')))
# pause before each letter, ms: ♡ | Н у | и | с а м о е | г л а в н о е
GAP = [0, 320, 90, 220, 220, 70, 70, 70, 70, 300, 70, 70, 70, 70, 70, 70]

V = 520.0      # pen speed on straight runs, px/s
STEP = 0.5     # route sampling, px


def smooth(a, r, passes):
    a = np.asarray(a, float).copy(); n = len(a)
    for _ in range(passes):
        c = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), a]), axis=0)
        i = np.arange(n); k = np.minimum(np.minimum(i, n - 1 - i), r)   # symmetric window, shrinking at the ends
        a = ((c[i + k + 1] - c[i - k]).T / (2 * k + 1)).T
    return a


def press(d):   # the same pencil pressure as the line in "The process"
    return .55 * np.sin(d / 9.1 + 1) + .4 * np.sin(d / 5.3 + 2.3) + .3 * np.sin(d / 17.7 + .5) + .5 * np.sin(d / 60 + .7)


def edge(d, ph):   # each edge of the line has its own small ups and downs
    return .26 * np.sin(d / 3.3 + ph) + .18 * np.sin(d / 8.1 + ph * 2.3) + .12 * np.sin(d / 1.9 + ph * .7) + .14 * np.sin(d / 23 + ph * 1.7)


def letter(a):
    P = np.array(a, float).reshape(-1, 3)
    pts, fl = [P[0, :2]], [0.0]
    for i in range(1, len(P)):
        n = max(1, int(np.ceil(np.hypot(*(P[i, :2] - P[i - 1, :2])) / STEP)))
        for j in range(1, n + 1):
            pts.append(P[i - 1, :2] + (P[i, :2] - P[i - 1, :2]) * j / n); fl.append(P[i, 2])
    pts = smooth(np.array(pts), 5, 3)            # irons out the 1-px staircase of the tracing
    fl = np.array(fl)
    seg = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]; s = np.cumsum(seg); L = s[-1]; n = len(pts)
    # bend: change of direction over ±4 px of route
    bend = np.zeros(n)
    for i in range(n):
        a0 = np.searchsorted(s, s[i] - 4); a1 = min(n - 1, np.searchsorted(s, s[i] + 4))
        if a0 >= i or a1 <= i: continue
        d1 = pts[i] - pts[a0]; d2 = pts[a1] - pts[i]
        bend[i] = abs(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)) / max(1e-3, s[a1] - s[a0]) * 2
    v = V / (1 + 7 * bend)
    v = np.where(fl > 0, V * 2.2, v)                              # sliding back over its own ink: quicker
    ei = np.clip(s / 30, 0, 1); ei = ei * ei * (3 - 2 * ei)       # sets off softly
    eo = np.clip((L - s) / 22, 0, 1); eo = eo * eo * (3 - 2 * eo)  # and eases at the end
    v = v * (.3 + .7 * ei) * (.55 + .45 * eo)
    v = smooth(v, max(1, int(round(12 / STEP))), 2)               # the hand changes speed gently
    tm = np.r_[0, np.cumsum(seg[1:] / ((v[1:] + v[:-1]) / 2) * 1000)]
    # tangent, for telling the two edges of the line apart
    tg = np.gradient(pts, axis=0); tg /= np.maximum(1e-6, np.hypot(*tg.T))[:, None]
    # pass number: a new pass starts where the pen turns back on itself
    ps = np.r_[0, np.cumsum(np.abs(np.diff(fl)) > 0)]
    return dict(p=pts, tm=tm, s=s, L=L, tg=tg, ps=ps, v=v)


letters = [letter(a) for a in HAND]
T = 0.0; S = 0.0
cols = {k: [] for k in 'p t li u s tg ps lsn lse'.split()}
meta = []
for li, l in enumerate(letters):
    T += GAP[li]
    meta.append([round(T), round(l['tm'][-1])])
    n = len(l['p'])
    cols['p'].append(l['p']); cols['t'].append(T + l['tm']); cols['li'].append(np.full(n, li))
    cols['u'].append(l['tm'] / l['tm'][-1]); cols['s'].append(S + l['s']); cols['tg'].append(l['tg'])
    cols['ps'].append(l['ps'] + li * 1000); cols['lsn'].append(l['s']); cols['lse'].append(l['L'] - l['s'])
    T += l['tm'][-1]; S += l['L'] + 40
C = {k: np.concatenate(v) if k not in ('p', 'tg') else np.vstack(v) for k, v in cols.items()}
tree = cKDTree(C['p'])

# ---- pressure along the route: lighter press = a thinner line (each edge wanders on its own) and a paler one ----
d = C['s']
pr = press(d) / 1.75                                              # about -1..1
taper = np.minimum(np.clip(C['lsn'] / 14, 0, 1), np.clip(C['lse'] / 18, 0, 1))
taper = .5 + .5 * taper * taper * (3 - 2 * taper)                # the pencil lands and lifts lightly
hw = 2.75 * (1 + .42 * pr) * taper                                # half-width, px (the drawing's own line is ~2.8)
HWL = np.clip(hw * (1 + .55 * edge(d, .4)), .9, 6)
HWR = np.clip(hw * (1 + .55 * edge(d, 2.6)), .9, 6)
SH = .62 + .38 * np.clip(.5 + .5 * pr, 0, 1) * (.55 + .45 * taper)

alpha = np.array(Image.open(os.path.join(root, 'hand2.png')).convert('RGBA'))[:, :, 3].astype(float) / 255
area = nd.binary_dilation(alpha > 0, iterations=2)               # the grain filter's soft edge spills ~1 px
ys, xs = np.nonzero(area); q = np.c_[xs, ys].astype(float)
_, own = tree.query(q)
A = np.zeros(len(q))
gap_pts = int(2 * 6 / STEP) + 4
for k, cand in enumerate(tree.query_ball_point(q, 6.0)):
    if not cand: continue
    c = np.sort(np.array(cand))
    # the nearby route points fall into passes of the pen (runs of neighbouring points); in each pass the nearest point
    # says how far the pixel is from that pass's centre line and when the pen is there
    br = np.r_[0, np.nonzero((np.diff(c) > gap_pts) | (np.diff(C['ps'][c]) != 0))[0] + 1, len(c)]
    best_t, cov_all, sh = None, 0.0, 1.0
    for i in range(len(br) - 1):
        r = c[br[i]:br[i + 1]]
        dd = np.hypot(*(C['p'][r] - q[k]).T); j = r[np.argmin(dd)]; dist = dd.min()
        side = (q[k, 0] - C['p'][j, 0]) * C['tg'][j, 1] - (q[k, 1] - C['p'][j, 1]) * C['tg'][j, 0]
        cov = np.clip((HWL[j] if side >= 0 else HWR[j]) - dist + .7, 0, 1)
        if cov <= .02: continue
        if cov > cov_all: cov_all, sh = cov, SH[j]
        if best_t is None or C['t'][j] < best_t: best_t, own[k] = C['t'][j], j   # the pen's first pass over it
    A[k] = cov_all * sh

lid = C['li'][own]; u = C['u'][own]
os.makedirs(os.path.join(root, 'hand_t'), exist_ok=True)
out = []
for li in range(len(HAND)):
    sel = lid == li
    x0, x1, y0, y1 = xs[sel].min(), xs[sel].max() + 1, ys[sel].min(), ys[sel].max() + 1
    im = np.zeros((y1 - y0, x1 - x0, 2), np.uint8)
    im[ys[sel] - y0, xs[sel] - x0, 0] = np.clip(np.round(u[sel] * 255), 0, 255)
    im[ys[sel] - y0, xs[sel] - x0, 1] = np.clip(np.round(A[sel] * 255), 0, 255)
    Image.fromarray(im, 'LA').save(os.path.join(root, 'hand_t', '%02d.png' % li), optimize=True)
    out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)] + meta[li])
print('  var HAND_T=' + json.dumps(out, separators=(',', ':')) + ';')
print('total ms', round(T), file=sys.stderr)
