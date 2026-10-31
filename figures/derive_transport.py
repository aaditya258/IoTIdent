import json, math, os, statistics as st

# Run from anywhere: paths resolve against the repository root.
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES=os.path.join(ROOT,"results")
from collections import defaultdict

def agg(v):
    n=len(v); m=st.mean(v); s=st.stdev(v) if n>1 else 0.0
    return dict(mean=round(m,4), std=round(s,4), ci95=round(1.96*s/math.sqrt(n),4) if n>1 else 0.0, n=n)

FIELDS=("known_acceptance","delta_A","unknown_rejection","unknown_rejection_at_oracle",
        "auroc_unknown","oscr","tau_used","tau_oracle",
        "known_macro_f1_after_rejection","known_macro_f1_after_rejection_at_oracle","closed_set_macro_f1")

def harvest(path):
    d=json.load(open(path))
    buck=defaultdict(lambda: defaultdict(list)); vd=defaultdict(lambda: defaultdict(int))
    nw=defaultdict(lambda: defaultdict(list))
    for rk,r in d["runs"].items():
        axis=rk.split("|")[0].split(":")[0]; meth=rk.split("|")[1]
        for cn,cv in r["conditions"].items():
            kind="control" if cn=="source_control" else cn
            for sc,m in cv["scorers"].items():
                key=f"{axis}|{meth}|{sc}|{kind}"
                for f in FIELDS:
                    if m.get(f) is not None: buck[key][f].append(m[f])
                nw[key]["known"].append(m["n_known"]); nw[key]["unknown"].append(m["n_unknown"])
    for vk,v in d["verdicts"].items():
        p=vk.split("|"); axis=p[0].split(":")[0]; meth=p[1]; sc=p[3]; kind=p[4]
        vd[f"{axis}|{meth}|{sc}|{kind}"][v["verdict"]]+=1
    out={}
    for key,b in buck.items():
        o={f:agg(b[f]) for f in FIELDS if b[f]}
        o["reps"]=len(b["delta_A"])
        o["windows_known_mean"]=int(st.mean(nw[key]["known"]))
        o["windows_unknown_mean"]=int(st.mean(nw[key]["unknown"]))
        if key in vd: o["verdicts"]=dict(vd[key])
        out[key]=o
    return dict(alpha=d["alpha"], method=d["methods"][0], views=d["views"], repeats=d["repeats"], cells=out)

res={"forest":harvest(os.path.join(RES,"cf_threshold_transport_rf_FHC_r5.json")),
     "gated":harvest(os.path.join(RES,"cf_threshold_transport_gated_FHC_r5.json"))}
json.dump(res, open(os.path.join(RES,"derived_transport.json"),"w"), indent=1)

# print the headline cells
for mdl in ("forest","gated"):
    print("#"*70); print(mdl.upper())
    for k in sorted(res[mdl]["cells"]):
        c=res[mdl]["cells"][k]
        if "|msp|" not in k and "|proto|" not in k: continue
        print(f"\n{k}  reps={c['reps']}  nK={c['windows_known_mean']} nU={c['windows_unknown_mean']}")
        for f in ("known_acceptance","delta_A","unknown_rejection","unknown_rejection_at_oracle","auroc_unknown","oscr","tau_used","tau_oracle","known_macro_f1_after_rejection","known_macro_f1_after_rejection_at_oracle","closed_set_macro_f1"):
            if f in c: print("   %-42s %.4f +- %.4f"%(f,c[f]["mean"],c[f]["ci95"]))
        if "verdicts" in c: print("   verdicts", c["verdicts"])
