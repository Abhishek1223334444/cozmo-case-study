import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from cozmo.capture import Capture; from cozmo.cache import fused_cloud
from cozmo import geometry as g, rooms, layout
RES=0.02
names = sys.argv[1:] or ['c00a170fe1','1a8384c3f6','c7d28f72c6']
fig, axs = plt.subplots(1, len(names), figsize=(8*len(names), 8), squeeze=False)
for k, n in enumerate(names):
    cap = Capture('samples/'+n); P,N,C = fused_cloud(cap)
    fy = g.find_floor(P,N); cy = g.find_ceiling(P,N,fy); h=P[:,1]-fy
    vert = np.abs(N[:,1])<g.VERTICAL_NY; horiz = np.abs(N[:,1])>g.HORIZONTAL_NY
    ang = g.manhattan_angle(N, vert)
    uv = g.PlanFrame(ang, np.zeros(2), RES, (1,1)).to_plan(P)
    frame = g.PlanFrame.fit(uv[np.abs(h)<0.05], ang, RES)
    wsel = vert & (h>0.2) & (h<2.0)
    wall = frame.rasterize(uv[wsel])>=3
    fpath = Path(f'out/cache/{n}_free.npy')
    free = np.load(fpath) if fpath.exists() else rooms.free_space(cap, frame, fy, cy)
    nuv = frame.to_plan(np.c_[N[wsel,0], 0*N[wsel,0], N[wsel,2]])
    traj = frame.to_plan(np.array([f.T_wc[:3,3] for f in cap.frames]))
    lab, unv = rooms.segment_rooms(free, wall, uv[wsel], nuv, traj, frame)
    ax = axs[0,k]
    ax.scatter(uv[wsel][::3,0], uv[wsel][::3,1], s=0.02, c='0.6')
    print(f"== {n}  rooms={lab.max()}")
    for rid in range(1, lab.max()+1):
        R = layout.fit_room(rid, lab==rid, frame, P, N, uv, fy, vert, horiz)
        V = np.vstack([R.vertices, R.vertices[:1]]); ax.plot(V[:,0], V[:,1], '-', lw=1.5)
        ch = R.ceiling_height
        print(f"  room {rid}: area {R.area:5.2f} m2, perim {R.perimeter:5.2f} m, walls {len(R.walls)}, "
              f"snapped {sum(w.snapped for w in R.walls)}/{len(R.walls)}, ceiling {'-' if ch is None else f'{ch:.3f}'} "
              f"(spread {R.ceiling_spread*100:.1f}cm)")
        print("     wall lengths:", np.round(R.wall_lengths, 3).tolist())
        ops = layout.find_openings(R, P, uv, free, frame)
        for o in ops:
            a = R.vertices[o.wall]; b = R.vertices[(o.wall+1)%len(R.vertices)]
            d = (b-a)/np.linalg.norm(b-a); p0 = a+d*o.start; p1 = a+d*(o.start+o.width)
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]], '-', lw=5, c='red' if o.kind=='door' else 'deepskyblue', alpha=.7)
            print(f"     {o.kind}: wall {o.wall} width {o.width:.3f} m  h {o.bottom:.2f}-{o.top:.2f}  seen {o.seen_through:.2f}")
        ax.text(*R.vertices.mean(0), str(rid), fontsize=11, weight='bold')
    ax.set_aspect('equal'); ax.set_title(n)
plt.tight_layout(); plt.savefig('out/fit.png', dpi=70)
