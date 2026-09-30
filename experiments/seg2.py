import sys, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from cozmo.capture import Capture; from cozmo.cache import fused_cloud
from cozmo import geometry as g
RES=0.02
def free_space(cap, frame, fy, cy, step=5, nbins=180):
    acc = np.zeros(frame.shape, np.uint16)
    top = (cy - fy - 0.15) if cy else 2.2
    for i in range(0, len(cap), step):
        P,_ = cap.points_world(i, min_conf=1, max_depth=5.0, stride=2)
        h = P[:,1]-fy; P = P[(h>0.15)&(h<top)]
        if len(P)<50: continue
        cam = frame.to_plan(cap.frames[i].T_wc[None,:3,3])[0]
        d = frame.to_plan(P)-cam
        az = np.arctan2(d[:,1], d[:,0]); rng = np.hypot(d[:,0],d[:,1])
        b = ((az+np.pi)/(2*np.pi)*nbins).astype(int)%nbins
        far = np.zeros(nbins); np.maximum.at(far, b, rng)
        occ = np.flatnonzero(far>0)
        if len(occ)<3: continue
        a = (occ+0.5)/nbins*2*np.pi-np.pi
        pts = cam + np.stack([np.cos(a),np.sin(a)],1)*far[occ,None]
        mask = np.zeros(frame.shape, np.uint8)
        cc = ((cam-frame.origin)/RES)
        pp = ((pts-frame.origin)/RES)
        for j in range(len(occ)-1):
            if occ[j+1]-occ[j]==1:
                tri = np.array([cc, pp[j], pp[j+1]])*16
                cv2.fillConvexPoly(mask, tri.astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        acc += mask
    return acc
def grow(seeds, interior, iters=1000):
    lab = seeds.copy()
    for _ in range(iters):
        d = ndi.grey_dilation(lab, size=(3,3))
        new = (lab==0) & interior & (d>0)
        if not new.any(): break
        lab[new] = d[new]
    return lab
names = sys.argv[1:] or ['c00a170fe1','1a8384c3f6','c7d28f72c6']
fig, axs = plt.subplots(2, len(names), figsize=(7*len(names), 13), squeeze=False)
for k, n in enumerate(names):
    cap = Capture('samples/'+n); P,N,C = fused_cloud(cap)
    fy = g.find_floor(P,N); cy = g.find_ceiling(P,N,fy); h=P[:,1]-fy
    vert = np.abs(N[:,1])<g.VERTICAL_NY
    ang = g.manhattan_angle(N, vert)
    uv = g.PlanFrame(ang, np.zeros(2), RES, (1,1)).to_plan(P)
    frame = g.PlanFrame.fit(uv[np.abs(h)<0.05], ang, RES)
    wall = frame.rasterize(uv[vert & (h>0.2) & (h<2.0)])>=3
    fs = free_space(cap, frame, fy, cy)
    np.save(f'out/cache/{n}_free.npy', fs)
    interior = (fs>=2) & ~wall
    interior = ndi.binary_opening(interior, np.ones((3,3)))
    dist = ndi.distance_transform_edt(interior)*RES
    seeds,ns = ndi.label(dist>0.42)
    sizes = ndi.sum(np.ones_like(dist), seeds, range(1,ns+1))*RES*RES
    seeds[np.isin(seeds, np.flatnonzero(sizes<0.15)+1)] = 0
    ws = grow(seeds, interior)
    ids = np.unique(ws)[1:]
    print(f"{n}: rooms={len(ids)} ceil_h={None if cy is None else round(cy-fy,3)}")
    for i in ids: print(f"   room {i}: area {np.sum(ws==i)*RES*RES:.2f} m2")
    axs[0,k].imshow(np.log1p(fs), origin='lower'); axs[0,k].imshow(np.ma.masked_equal(wall,0),origin='lower',cmap='gray_r'); axs[0,k].set_title(n+' free space')
    axs[1,k].imshow(np.ma.masked_equal(ws,0), origin='lower', cmap='tab20'); axs[1,k].imshow(np.ma.masked_equal(wall,0),origin='lower',cmap='gray_r',alpha=.6)
    for i in ids:
        r,c = np.argwhere(ws==i).mean(0); axs[1,k].text(c,r,str(i),fontsize=9)
plt.tight_layout(); plt.savefig('out/seg2.png', dpi=70)
