"""Score and re-anchor the relative-mode inner-product cost law.

Reads the two calibration CSVs written by ``calibrate_rel_ip_cost.py``
(Python) and ``calibrateRelIpCost.m`` (MATLAB), reconstructs the
selector's predictors from their columns, and reports

* the routing regret (held-out over random halves) of the shipped
  ``_REL_COST_LAW`` / ``relRouteCostMs`` constants,
* the same for a plain log-log refit of all nine laws (which, on the
  6 September 2026 calibration, scores several times worse -- the
  shipped exponents stand), and
* a per-order multiplicative correction to the Bulger prediction chosen
  to minimise regret, which is what the shipped intercepts at r = 3, 4
  (Python) and r = 2, 3 (MATLAB) now carry.

Usage::

    python tools/fit_rel_ip_cost.py rel_ip_cost_python.csv rel_ip_cost_matlab.csv

The shipped constants below are the ones *before* the re-anchoring, so
the script reproduces the comparison that justified it.
"""
import numpy as np, math, sys, csv
PATHS={'python': sys.argv[1], 'matlab': sys.argv[2]}
LAW = {
 "python": {"bulger":{2:(-8.5858,0.8245),3:(-8.1774,0.7890),4:(-6.7951,0.7278)},
            "centres":{2:(-8.8001,0.8195),3:(-10.0569,0.9269),4:(-9.7867,0.9251)},
            "grid":{2:(-5.0440,0.4817),3:(-3.6893,0.5881),4:(-3.2639,0.7894)}},
}
FLOOR = {"python": {2:(0.05,0.067),3:(0.09,0.17),4:(0.0,2.65)},
         "matlab": {2:(0.137,0.086),3:(0.280,0.353),4:(0.497,7.956)}}
def load(fn):
    rows=[]
    with open(fn) as f:
        rdr=csv.DictReader(l for l in f if not l.startswith('#'))
        for d in rdr:
            r=int(d['r']);Kx=int(d['K_x']);Ky=int(d['K_y']);N=int(d['N'])
            M=lambda K: math.factorial(r)*math.comb(K,r)
            C=lambda K: math.comb(K,r)
            pairs=N*N
            nJx,nKx,nJy,nKy=N*M(Kx),N*C(Kx),N*M(Ky),N*C(Ky)
            d['term_bulger']=nJx*nKy+nJx*nKx+nJy*nKy
            d['term_centres']=pairs*(M(Kx)*M(Ky)+M(Kx)**2+M(Ky)**2)
            d['term_grid']=pairs*float(d['nu'])*max(Kx,Ky)
            for k in ('t_bulger','t_centres','t_grid','pred_bulger','pred_mobius'):
                d[k]=float(d[k]) if d[k] not in ('','-') else float('nan')
            d['r']=r
            rows.append(d)
    return rows
def cost(law,route,r,term): a,b=law[route][min(max(r,2),4)]; return math.exp(a)*max(term,1.0)**b
def pred(law,floor,d):
    r=d['r']; pb=cost(law,'bulger',r,d['term_bulger'])
    pm=cost(law,'grid',r,d['term_grid'])
    if d['gate_route']=='centres' or True:  # centres admissible unless gate blocked; CSV gate says which route the orchestrator takes
        pass
    pc=cost(law,'centres',r,d['term_centres']) if math.isfinite(d['t_centres']) or d['declined_centres'] in ('','-','memory','budget') else float('inf')
    pm=min(pm,pc); f,p=floor[min(max(r,2),4)]; pm=max(pm,f+3*p)
    return pb,pm
def t_mobius(d):  # actual time of the Möbius method = the route the gate takes
    return d['t_centres'] if d['gate_route']=='centres' else d['t_grid']
def regret(law,floor,rows):
    tot=0.0; wrong=0; n=0
    for d in rows:
        pb,pm=pred(law,floor,d); tb=d['t_bulger']; tm=t_mobius(d)
        if math.isnan(tb) or math.isnan(tm): continue
        choose_b = pb<=pm
        tc = tb if choose_b else tm
        best=min(tb,tm)
        if math.isinf(tc) and math.isinf(best): continue
        n+=1
        if tc>best*1.0000001: wrong+=1
        tot+= (tc-best) if math.isfinite(tc) else 5000.0  # censored: budget 5 s
    return tot,wrong,n
def fit(rows):
    law={"bulger":{},"centres":{},"grid":{}}
    for route,tk,term in (("bulger","t_bulger","term_bulger"),("centres","t_centres","term_centres"),("grid","t_grid","term_grid")):
        for r in (2,3,4):
            x=[];y=[]
            for d in rows:
                if d['r']==r and math.isfinite(d[tk]) and d[tk]>0:
                    x.append(math.log(max(d[term],1.0))); y.append(math.log(d[tk]))
            X=np.vstack([np.ones(len(x)),x]).T; a,b=np.linalg.lstsq(X,np.array(y),rcond=None)[0]
            law[route][r]=(a,b)
    return law
for lang in ("python","matlab"):
    rows=load(PATHS[lang]); floor=FLOOR[lang]
    # check reproduction of csv predictions with the python law (both languages carry own laws; only python known here)
    if lang=="python":
        err=[abs(pred(LAW['python'],floor,d)[0]-d['pred_bulger'])/d['pred_bulger'] for d in rows[:200]]
        errm=[abs(pred(LAW['python'],floor,d)[1]-d['pred_mobius'])/d['pred_mobius'] for d in rows[:200]]
        print("repro bulger max rel err", max(err), "mobius", max(errm))
    full=fit(rows)
    print(f"\n== {lang}: {len(rows)} cells")
    if lang=="python":
        print(" shipped law regret (all cells): %.0f ms, %d/%d wrong" % regret(LAW['python'],floor,rows))
    print(" refit law regret (in-sample): %.0f ms, %d/%d wrong" % regret(full,floor,rows))
    rng=np.random.default_rng(0); res=[]
    for rep in range(200):
        idx=rng.permutation(len(rows)); h=len(rows)//2
        tr=[rows[i] for i in idx[:h]]; te=[rows[i] for i in idx[h:]]
        law=fit(tr)
        rf=regret(law,floor,te)
        rs=regret(LAW['python'],floor,te) if lang=="python" else (float('nan'),0,0)
        res.append((rf[0],rf[1],rs[0],rs[1]))
    res=np.array(res)
    print(" CV held-out (random halves, 200 reps): refit regret %.0f ms (%.1f wrong) ; shipped %.0f ms (%.1f wrong)" % (res[:,0].mean(),res[:,1].mean(),res[:,2].mean(),res[:,3].mean()))
    for route in full:
        print("  ",route,{r:(round(a,4),round(b,4)) for r,(a,b) in full[route].items()})
    LAW[lang+"_fit"]=full
# cross-language transfer
py=load(PATHS["python"]); ml=load(PATHS["matlab"])
print("\ntransfer: python-fit on matlab cells: %.0f ms %d/%d wrong" % regret(LAW['python_fit'],FLOOR['matlab'],ml))
print("transfer: matlab-fit on python cells: %.0f ms %d/%d wrong" % regret(LAW['matlab_fit'],FLOOR['python'],py))
print("oracle-free baseline: always bulger / always mobius:")
for lang,rows,fl in (("python",py,FLOOR['python']),("matlab",ml,FLOOR['matlab'])):
    B={"bulger":{r:(-100,0) for r in (2,3,4)},"centres":{r:(100,0) for r in (2,3,4)},"grid":{r:(100,0) for r in (2,3,4)}}
    Mo={"bulger":{r:(100,0) for r in (2,3,4)},"centres":{r:(-100,0) for r in (2,3,4)},"grid":{r:(-100,0) for r in (2,3,4)}}
    print(" ",lang,"always bulger %.0f ms %d/%d wrong" % regret(B,fl,rows), "| always mobius %.0f ms %d/%d wrong" % regret(Mo,fl,rows))

LAW["matlab"]={"bulger":{2:(-7.2149,0.6970),3:(-8.0168,0.7493),4:(-7.7822,0.7500)},
               "centres":{2:(-7.7429,0.6800),3:(-8.6263,0.7781),4:(-8.8269,0.8029)},
               "grid":{2:(-4.1388,0.3916),3:(-2.1985,0.5021),4:(-0.5409,0.5768)}}
print("\nmatlab shipped law regret (all cells): %.0f ms, %d/%d wrong" % regret(LAW['matlab'],FLOOR['matlab'],ml))
rng=np.random.default_rng(1); acc=[]
for rep in range(200):
    idx=rng.permutation(len(ml)); h=len(ml)//2; te=[ml[i] for i in idx[h:]]; tr=[ml[i] for i in idx[:h]]
    acc.append(regret(LAW['matlab'],FLOOR['matlab'],te)[:2]+regret(fit(tr),FLOOR['matlab'],te)[:2])
acc=np.array(acc); print("matlab CV held-out: shipped %.0f ms (%.1f wrong) vs refit %.0f ms (%.1f wrong)"%(acc[:,0].mean(),acc[:,1].mean(),acc[:,2].mean(),acc[:,3].mean()))
# where are the shipped-law misroutes? by r and N and censoring
import collections
for lang,rows,fl in (("python",py,FLOOR['python']),("matlab",ml,FLOOR['matlab'])):
    c=collections.Counter(); reg=collections.Counter()
    for d in rows:
        pb,pm=pred(LAW[lang],fl,d); tb=d['t_bulger']; tm=t_mobius(d)
        if math.isnan(tb) or math.isnan(tm): continue
        tc=tb if pb<=pm else tm; best=min(tb,tm)
        if math.isinf(tc) and math.isinf(best): continue
        if tc>best*1.0000001:
            key=(d['r'],d['N'],'toB' if pb<=pm else 'toM', 'cens' if math.isinf(tc) else 'fin')
            c[key]+=1; reg[key]+= (tc-best) if math.isfinite(tc) else 5000
    print(lang, "misroutes by (r,N,chosen,censored): count / regret ms")
    for k in sorted(c): print("   ",k,c[k],round(reg[k]))

def scaled(law, s):  # multiply bulger prediction by s[r] (shift a)
    out={k:dict(v) for k,v in law.items()}
    out['bulger']={r:(a+math.log(s[r]),b) for r,(a,b) in law['bulger'].items()}
    return out
grid_s=[0.25,0.35,0.5,0.7,1.0,1.4,2.0,2.8,4.0]
for lang,rows,fl in (("python",py,FLOOR['python']),("matlab",ml,FLOOR['matlab'])):
    rng=np.random.default_rng(2); acc=[]
    for rep in range(100):
        idx=rng.permutation(len(rows)); h=len(rows)//2; tr=[rows[i] for i in idx[:h]]; te=[rows[i] for i in idx[h:]]
        best={}
        for r in (2,3,4):
            trr=[d for d in tr if d['r']==r]
            best[r]=min(grid_s, key=lambda s: regret(scaled(LAW[lang],{2:1,3:1,4:1}|{r:s}),fl,trr)[0])
        acc.append((regret(LAW[lang],fl,te)[0], regret(scaled(LAW[lang],best),fl,te)[0], best[2],best[3],best[4]))
    acc=np.array(acc)
    print(lang,"held-out: shipped %.0f ms vs per-r bulger scale %.0f ms; median scales r2=%.2f r3=%.2f r4=%.2f"%(acc[:,0].mean(),acc[:,1].mean(),np.median(acc[:,2]),np.median(acc[:,3]),np.median(acc[:,4])))

print("\n--- refined per-r Bulger intercept shift ---")
grid_s=list(np.exp(np.linspace(np.log(1/8),np.log(8),33)))
for lang,rows,fl in (("python",py,FLOOR['python']),("matlab",ml,FLOOR['matlab'])):
    rng=np.random.default_rng(3); acc=[]
    for rep in range(200):
        idx=rng.permutation(len(rows)); h=len(rows)//2; tr=[rows[i] for i in idx[:h]]; te=[rows[i] for i in idx[h:]]
        best={}
        for r in (2,3,4):
            trr=[d for d in tr if d['r']==r]
            best[r]=min(grid_s, key=lambda s: regret(scaled(LAW[lang],{2:1,3:1,4:1}|{r:s}),fl,trr)[0])
        rs=regret(LAW[lang],fl,te); rn=regret(scaled(LAW[lang],best),fl,te)
        acc.append((rs[0],rs[1],rn[0],rn[1],best[2],best[3],best[4]))
    acc=np.array(acc)
    full={r:min(grid_s, key=lambda s: regret(scaled(LAW[lang],{2:1,3:1,4:1}|{r:s}),fl,[d for d in rows if d['r']==r])[0]) for r in (2,3,4)}
    print(lang,"held-out shipped %.0f ms (%.1f wrong) -> shifted %.0f ms (%.1f wrong); median CV scales r2=%.2f r3=%.2f r4=%.2f; full-data scales r2=%.3f r3=%.3f r4=%.3f"%(acc[:,0].mean(),acc[:,1].mean(),acc[:,2].mean(),acc[:,3].mean(),np.median(acc[:,4]),np.median(acc[:,5]),np.median(acc[:,6]),full[2],full[3],full[4]))
    print("   in-sample with full-data scales: %.0f ms %d/%d wrong"%regret(scaled(LAW[lang],full),fl,rows))
    # flatness: regret over the scale grid for r=2 (full data)
    for r in (2,3,4):
        rr=[d for d in rows if d['r']==r]
        prof=[(round(s,2),round(regret(scaled(LAW[lang],{2:1,3:1,4:1}|{r:s}),fl,rr)[0])) for s in grid_s[::4]]
        print("   r=%d profile"%r,prof)
    LAW[lang+"_shift"]=full
