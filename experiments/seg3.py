import sys, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from cozmo.capture import Capture; from cozmo.cache import fused_cloud
from cozmo import geometry as g
RES=0.02
def disk(r_m):
    r = int(round(r_m/RES)); y,x = np.mgrid[-r:r+1,-r:r+1]; return x*x+y*y<=r*r
def grow(seeds, interior, iters=2000):
    lab = seeds.copy()
    for _ in range(iters):
        d = ndi.grey_dilation(lab, size=(3,3))
        new = (lab==0) & interior & (d>0)
        if not new.any(): break
        lab[new] = d[new]
    return lab
def sweep_seeds(interior, min_seed=0.3):
    """Morphological sweep: erode progressively; a region becomes several seeds only
    when it genuinely splits into >=2 parts of meaningful size."""
    dist = ndi.distance_transform_edt(interior)*RES
    seeds, n = ndi.label(dist>0.10)
    for t in np.arange(0.14, 1.5, 0.04):
        comps, nc = ndi.label(dist>t)
        if nc==0: break
        areas = ndi.sum(np.ones_like(dist), comps, range(1,nc+1))*RES*RES
        new = seeds.copy(); nxt = seeds.max()+1
        for sid in np.unique(seeds)[1:]:
            inside = np.unique(comps[(seeds==sid)&(comps>0)])
            big = [c for c in inside if areas[c-1]>=min_seed]
            if len(big)>=2:
                region = seeds==sid
                new[region] = 0
                for c in big:
                    new[comps==c] = nxt; nxt += 1
        seeds = new
    return seeds
def boundary_is_door(ws, a, b, wuv, wn, frame):
    """True if the a|b boundary sits in a wall line (wall faces parallel to the cut
    continue beyond both ends)."""
    ma = ws==a; mb = ws==b
    cut = (ndi.binary_dilation(ma, np.ones((3,3))) & mb) | (ndi.binary_dilation(mb, np.ones((3,3))) & ma)
    rr, cc = np.nonzero(cut)
    pts = frame.cell_center(rr, cc)
    ext = pts.max(0)-pts.min(0)
    ax = int(np.argmax(ext))
    width = ext[ax]
    lo, hi = pts[:,ax].min(), pts[:,ax].max()
    mid = np.median(pts[:,1-ax])
    face_parallel = np.abs(wn[:, 1-ax]) > 0.8
    ends = []
    for sgn, e in ((-1, lo), (1, hi)):
        along = (wuv[:,ax]-e)*sgn
        m = face_parallel & (along>0.02) & (along<0.40) & (np.abs(wuv[:,1-ax]-mid)<0.25)
        span = np.ptp(wuv[m,ax]) if m.sum()>20 else 0.0
        ends.append(span)
    door = min(ends) > 0.08 or (max(ends) > 0.08 and width < 1.3)
    return door, width, ends
def merge_non_doors(ws, wuv, wn, frame, verbose=False):
    changed = True
    while changed:
        changed = False
        for a in np.unique(ws)[1:]:
            ma = ws==a; ring = ndi.binary_dilation(ma, np.ones((3,3))) & ~ma
            nb, cnt = np.unique(ws[ring], return_counts=True)
            for b,cn in zip(nb,cnt):
                if b<=a or cn<5: continue
                door, width, ends = boundary_is_door(ws, a, b, wuv, wn, frame)
                if verbose: print(f"     {a}|{b} width={width:.2f} ends={np.round(ends,2)} door={door}")
                if not door:
                    ws[ws==b] = a; changed = True; break
            if changed: break
    return ws
def visited_only(ws, frame, cap):
    traj = frame.to_plan(np.array([f.T_wc[:3,3] for f in cap.frames]))
    r,c = frame.to_cell(traj)
    ok = (r>=0)&(r<ws.shape[0])&(c>=0)&(c<ws.shape[1])
    ids, cnt = np.unique(ws[r[ok],c[ok]], return_counts=True)
    keep = ids[(ids>0)&(cnt>=30)]
    return np.where(np.isin(ws, keep), ws, 0), traj
names = sys.argv[1:] or ['c00a170fe1','1a8384c3f6','c7d28f72c6']
fig, axs = plt.subplots(1, len(names), figsize=(7*len(names), 7), squeeze=False)
for k, n in enumerate(names):
    cap = Capture('samples/'+n); P,N,C = fused_cloud(cap)
    fy = g.find_floor(P,N); cy = g.find_ceiling(P,N,fy); h=P[:,1]-fy
    vert = np.abs(N[:,1])<g.VERTICAL_NY
    ang = g.manhattan_angle(N, vert)
    uv = g.PlanFrame(ang, np.zeros(2), RES, (1,1)).to_plan(P)
    frame = g.PlanFrame.fit(uv[np.abs(h)<0.05], ang, RES)
    wall = frame.rasterize(uv[vert & (h>0.2) & (h<2.0)])>=3
    fs = np.load(f'out/cache/{n}_free.npy')
    envelope = ndi.binary_fill_holes(ndi.binary_closing(wall, disk(0.5), border_value=0))
    interior = (fs>=2) & ~wall
    interior = ndi.binary_opening(interior, disk(0.04))
    seeds = sweep_seeds(interior)
    ws = grow(seeds, interior)
    ids, cnt = np.unique(ws[ws>0], return_counts=True)
    small = ids[cnt*RES*RES<1.0]
    ws[np.isin(ws, small)] = 0; ws = grow(ws, interior)
    wsel = vert & (h>0.3) & (h<2.0)
    wuv = uv[wsel]; wn = np.stack([frame.to_plan(np.c_[N[wsel,0],0*N[wsel,0],N[wsel,2]])[:,0], frame.to_plan(np.c_[N[wsel,0],0*N[wsel,0],N[wsel,2]])[:,1]],1)
    ws = merge_non_doors(ws, wuv, wn, frame)
    ws, traj = visited_only(ws, frame, cap)
    ids = np.unique(ws)[1:]
    print(f"{n}: rooms={len(ids)} envelope area={envelope.sum()*RES*RES:.1f} m2")
    for i in ids: print(f"   room {i}: area {np.sum(ws==i)*RES*RES:.2f} m2")
    ax=axs[0,k]; ax.imshow(np.ma.masked_equal(ws,0), origin='lower', cmap='tab20'); ax.imshow(np.ma.masked_equal(wall,0),origin='lower',cmap='gray_r',alpha=.6)
    for i in ids:
        r,c = np.argwhere(ws==i).mean(0); ax.text(c,r,str(i),fontsize=9)
    ax.plot(*((traj-frame.origin)/RES).T,'k-',lw=.4); ax.set_title(n)
plt.tight_layout(); plt.savefig('out/seg3.png', dpi=70)
