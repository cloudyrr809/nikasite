"""Makes hand_t/NN.png: the maps that let "Ну и самое главное" be written by hand on the page.

Each map belongs to one letter. For every ink pixel of hand2.png it holds
  grey  = when the pen passes that pixel (0 = the letter's first moment, 255 = its last),
  alpha = how much of the pixel the pencil's pressure leaves (pressed lightly: a thinner, paler line; never wider
          than the drawing's own line, so nothing grows after it is written).
A pixel belongs only to the pass of the pen whose line it lies on; where two lines really overlap, to the first pass.
So at crossings and where the pen runs back over itself nothing shows up before the pen gets there.

The route (tools/hand_route.json): one list per letter of x, y, f — f = 1 where the pen slides back over its own ink.
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
GAP = [0, 220, 60, 150, 150, 45, 45, 45, 45, 200, 45, 45, 45, 45, 45, 45]

V = 820.0      # pen speed on straight runs, px/s
STEP = 0.5     # route sampling, px
INK = 3.0      # half-width of the drawing's line (measured ~2.8) — a pixel further than this from a pass is not its ink


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
    pts = np.array(pts); fl = np.array(fl)
    ps = np.r_[0, np.cumsum(np.diff(fl) != 0)]                    # pass number: a new pass where the pen turns back
    # iron out the 1-px staircase of the tracing, pass by pass (so the turning points keep their tips)
    for p in np.unique(ps):
        m = np.nonzero(ps == p)[0]; lo = max(0, m[0] - 1)             # shares its first point with the pass before
        pts[lo:m[-1] + 1] = smooth(pts[lo:m[-1] + 1], 8, 4)
    seg = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]; s = np.cumsum(seg); L = s[-1]; n = len(pts)
    bend = np.zeros(n)                                            # change of direction over ±4 px of route
    for i in range(n):
        a0 = np.searchsorted(s, s[i] - 4); a1 = min(n - 1, np.searchsorted(s, s[i] + 4))
        if a0 >= i or a1 <= i: continue
        d1 = pts[i] - pts[a0]; d2 = pts[a1] - pts[i]
        bend[i] = abs(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)) / max(1e-3, s[a1] - s[a0]) * 2
    v = V / (1 + 6 * bend)
    v = np.where(fl > 0, V * 2.0, v)                              # sliding back over its own ink: quicker
    ei = np.clip(s / 26, 0, 1); ei = ei * ei * (3 - 2 * ei)       # sets off softly
    eo = np.clip((L - s) / 20, 0, 1); eo = eo * eo * (3 - 2 * eo)  # and eases at the end
    v = v * (.35 + .65 * ei) * (.6 + .4 * eo)
    v = smooth(v, max(1, int(round(12 / STEP))), 2)               # the hand changes speed gently
    tm = np.r_[0, np.cumsum(seg[1:] / ((v[1:] + v[:-1]) / 2) * 1000)]
    tg = np.gradient(pts, axis=0); tg /= np.maximum(1e-6, np.hypot(*tg.T))[:, None]
    return dict(p=pts, tm=tm, s=s, L=L, tg=tg, ps=ps)


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

# ---- pressure along the route: pressed lightly = a thinner line (each edge wanders on its own) and a paler one.
#      The line is only ever worn thinner than the drawing's own, never made wider ----
d = C['s']
pr = press(d) / 1.75                                              # about -1..1
taper = np.minimum(np.clip(C['lsn'] / 12, 0, 1), np.clip(C['lse'] / 16, 0, 1))
taper = .5 + .5 * taper * taper * (3 - 2 * taper)                # the pencil lands and lifts lightly
hw = 2.45 * (1 + .36 * pr) * taper
alpha = np.array(Image.open(os.path.join(root, 'hand2.png')).convert('RGBA'))[:, :, 3].astype(float) / 255
# the drawing's own half-width along the route, read away from crossings (where two lines make the ink look wider):
# depth of the ink at the route, the lower part of it over ±10 px of route
inside = nd.distance_transform_edt(alpha > .5)
rp = np.clip(np.round(C['p']).astype(int), 0, [alpha.shape[1] - 1, alpha.shape[0] - 1])
wl = nd.percentile_filter(inside[rp[:, 1], rp[:, 0]], 25, size=41, mode='nearest')
cap = np.clip(wl + .2, 1.5, INK)
HWL = np.clip(hw * (1 + .4 * edge(d, .4)), .9, cap)
HWR = np.clip(hw * (1 + .4 * edge(d, 2.6)), .9, cap)
SH = .64 + .36 * np.clip(.5 + .5 * pr, 0, 1) * (.55 + .45 * taper)

area = nd.binary_dilation(alpha > 0, iterations=1)
ys, xs = np.nonzero(area); q = np.c_[xs, ys].astype(float)
_, own = tree.query(q)
A = np.zeros(len(q))
gap_pts = int(2 * (INK + 2) / STEP) + 4
for k, cand in enumerate(tree.query_ball_point(q, INK + 1.5)):
    if not cand: continue
    c = np.sort(np.array(cand))
    # the nearby route points fall into passes of the pen (runs of neighbouring points); in each pass the nearest point
    # says how far the pixel is from that pass's centre line and when the pen is there
    br = np.r_[0, np.nonzero((np.diff(c) > gap_pts) | (np.diff(C['ps'][c]) != 0))[0] + 1, len(c)]
    runs = []
    for i in range(len(br) - 1):
        r = c[br[i]:br[i + 1]]
        dd = np.hypot(*(C['p'][r] - q[k]).T); runs.append((dd.min(), r[np.argmin(dd)]))
    dmin = min(r[0] for r in runs)
    # every pass is a band of its own width round its (smoothed, straight-through) centre line; the first band that really covers the pixel owns it
    mine = [r for r in runs if r[0] <= INK]
    got = []
    for dist, j in mine:
        side = (q[k, 0] - C['p'][j, 0]) * C['tg'][j, 1] - (q[k, 1] - C['p'][j, 1]) * C['tg'][j, 0]
        cov = np.clip((HWL[j] if side >= 0 else HWR[j]) - dist + .6, 0, 1)
        if cov > .02: got.append((C['t'][j], cov, j))
    if not got: own[k] = min(runs)[1]; continue
    got.sort()
    # the first pass over it takes it if that pass really paints it; a pixel only grazed by the first pass waits for
    # the pass that paints it most — so the second line never shows up at a crossing with the first
    t_, cov, j = next((g for g in got if g[1] >= .5), None) or max(got, key=lambda g: g[1])
    # where the pen runs up a line and slides back down it, the two passes are one line, not two thin ones crossing:
    # the pixel takes the fuller of the two (it shows with the first, which is the same line)
    for dist, jj in runs:
        if abs(C['ps'][jj] - C['ps'][j]) == 1:
            side = (q[k, 0] - C['p'][jj, 0]) * C['tg'][jj, 1] - (q[k, 1] - C['p'][jj, 1]) * C['tg'][jj, 0]
            cov = max(cov, np.clip((HWL[jj] if side >= 0 else HWR[jj]) - dist + .6, 0, 1))
    own[k] = j; A[k] = cov * SH[j]

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
