import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from cozmo.capture import Capture; from cozmo.cache import fused_cloud
from cozmo import geometry as g
RES=0.02
names = sys.argv[1:] or ['c00a170fe1','1a8384c3f6','c7d28f72c6']
fig, axs = plt.subplots(2, len(names), figsize=(7*len(names), 13), squeeze=False)
for k, n in enumerate(names):
    cap = Capture('samples/'+n); P,N,C = fused_cloud(cap)
    fy = g.find_floor(P,N); cy = g.find_ceiling(P,N,fy)
    h = P[:,1]-fy
    vert = np.abs(N[:,1])<g.VERTICAL_NY; horiz = np.abs(N[:,1])>g.HORIZONTAL_NY
    ang = g.manhattan_angle(N, vert)
    uv = np.zeros((len(P),2)); frame = None
    tmp = g.PlanFrame(ang, np.zeros(2), RES, (1,1)); uv = tmp.to_plan(P)
    frame = g.PlanFrame.fit(uv[np.abs(h)<0.05], ang, RES)
    floor = frame.rasterize(uv[horiz & (np.abs(h)<0.03)])>0
    wall = frame.rasterize(uv[vert & (h>0.2) & (h<2.0)])>=3
    traj = frame.to_plan(np.array([f.T_wc[:3,3] for f in cap.frames]))
    # interior: floor seen, closed and hole-filled, walls removed
    interior = ndi.binary_closing(floor, np.ones((7,7)))
    interior = ndi.binary_fill_holes(interior) & ~ndi.binary_dilation(wall, np.ones((3,3)))
    lab,_ = ndi.label(interior); r,c = frame.to_cell(traj)
    keep = np.unique(lab[r,c]); keep = keep[keep>0]
    interior = np.isin(lab, keep)
    dist = ndi.distance_transform_edt(interior)*RES
    seeds,ns = ndi.label(dist>0.45)
    sizes = ndi.sum(np.ones_like(dist), seeds, range(1,ns+1))*RES*RES
    for i,s in enumerate(sizes,1):
        if s<0.3: seeds[seeds==i]=0
    # geodesic assignment: nearest seed via watershed on -dist
    import cv2
    markers = seeds.astype(np.int32); markers[~interior]=-1
    img = np.dstack([(255*(1-dist/dist.max())).astype(np.uint8)]*3)
    ws = cv2.watershed(img, markers.copy()); ws[~interior]=0; ws[ws<0]=0
    nroom = len(np.unique(ws))-1
    print(f"{n}: floor_y={fy:.3f} ceil_h={None if cy is None else round(cy-fy,3)} angle={np.degrees(ang):.2f}deg grid={frame.shape} rooms={nroom}")
    for i in np.unique(ws)[1:]:
        print(f"   room {i}: area {np.sum(ws==i)*RES*RES:.2f} m2")
    ax=axs[0,k]; ax.imshow(floor*1+wall*2, origin='lower', cmap='gray_r'); ax.plot(*((traj-frame.origin)/RES).T,'r-',lw=.5); ax.set_title(n+' floor/wall')
    ax=axs[1,k]; ax.imshow(np.ma.masked_equal(ws,0), origin='lower', cmap='tab20'); ax.imshow(np.ma.masked_equal(wall,0),origin='lower',cmap='gray_r',alpha=.6); ax.set_title('rooms')
plt.tight_layout(); plt.savefig('out/seg.png', dpi=70)
