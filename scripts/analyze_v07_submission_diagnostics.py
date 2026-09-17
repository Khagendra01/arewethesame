from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ARCHS=("qwen","mistral")
OWNERS=("self","other","focal")
SEEDS=(31,42,73,128,256)
ORDER=("first","second")
HIST=(0,1)
METRICS=("delta_ownership_sensitivity","delta_self_vs_focal_sensitivity","delta_focal_vs_other_sensitivity")


def load_json(p:Path):
    with p.open() as f: return json.load(f)

def realization_paths(root:Path, arch:str):
    return [root/arch/"base.json"], [root/arch/f"mixed_s{s}.json" for s in SEEDS]

def item_abs_metrics(row, owner):
    m=row["mass"]
    probs=[]; acc=[]
    for order in ORDER:
        for h in HIST:
            r=m[f"rel_{owner}_{order}_{h}"]
            probs.append(float(r["correct_prob"]))
            acc.append(float(bool(r["correct"])))
    s=row["metrics_mass"]["S"]
    sens=0.5*(float(s[f"{owner}_first"])+float(s[f"{owner}_second"]))
    return np.array([np.mean(probs),np.mean(acc),sens],dtype=float)

def item_switch(row,a,b):
    m=row["mass"]; vals=[]
    for order in ORDER:
        for h in HIST:
            vals.append(float(m[f"rel_{a}_{order}_{h}"]["choice"] != m[f"rel_{b}_{order}_{h}"]["choice"]))
    return float(np.mean(vals))

def bootstrap_matrix(mat, n_boot, rng):
    # mat [seed, item, metric] or [1,item,metric]
    ns,ni,nm=mat.shape
    draws=np.empty((n_boot,nm),dtype=float)
    for b in range(n_boot):
        si = rng.integers(0,ns,size=ns) if ns>1 else np.array([0])
        ii = rng.integers(0,ni,size=ni)
        draws[b] = mat[si][:,ii,:].mean(axis=(0,1))
    return np.percentile(draws,[2.5,97.5],axis=0).T

def summarize_abs(files,n_boot,rng):
    datas=[load_json(p) for p in files]
    ids=sorted(datas[0]["items"])
    assert all(sorted(d["items"])==ids for d in datas)
    out={}
    for owner in OWNERS:
        mat=np.array([[item_abs_metrics(d["items"][iid],owner) for iid in ids] for d in datas])
        mean=mat.mean(axis=(0,1)); ci=bootstrap_matrix(mat,n_boot,rng)
        out[owner]={
            "mean_designated_score":float(mean[0]),"ci95_designated_score":ci[0].tolist(),
            "policy_agreement_accuracy":float(mean[1]),"ci95_policy_agreement_accuracy":ci[1].tolist(),
            "mean_S_sensitivity":float(mean[2]),"ci95_mean_S_sensitivity":ci[2].tolist(),
        }
    return out

def summarize_switch(files,n_boot,rng):
    datas=[load_json(p) for p in files]; ids=sorted(datas[0]["items"])
    out={}
    for a,b in (("self","other"),("self","focal")):
        mat=np.array([[[item_switch(d["items"][iid],a,b)] for iid in ids] for d in datas])
        mean=float(mat.mean()); ci=bootstrap_matrix(mat,n_boot,rng)[0].tolist()
        out[f"{a}_vs_{b}"]={"switch_rate":mean,"ci95":ci,"matched_variants_per_realization":len(ids)*4,"realizations":len(datas),"total_matched_variants_descriptive":len(ids)*4*len(datas)}
    return out

def seed_sensitivity(root,arch):
    base=load_json(root/arch/"base.json")
    mixed=[load_json(root/arch/f"mixed_s{s}.json") for s in SEEDS]
    ids=sorted(base["items"])
    base_means={m:float(np.mean([base["items"][iid]["metrics_mass"][m] for iid in ids])) for m in METRICS}
    seed_means={}
    for s,d in zip(SEEDS,mixed):
        seed_means[str(s)]={m:float(np.mean([d["items"][iid]["metrics_mass"][m] for iid in ids])) for m in METRICS}
    loo={}
    for omit in SEEDS:
        keep=[seed_means[str(s)] for s in SEEDS if s!=omit]
        rec={m:float(np.mean([x[m] for x in keep])) for m in METRICS}
        rec["self_focal_change_vs_base"]=rec["delta_self_vs_focal_sensitivity"]-base_means["delta_self_vs_focal_sensitivity"]
        rec["focal_other_change_vs_base"]=rec["delta_focal_vs_other_sensitivity"]-base_means["delta_focal_vs_other_sensitivity"]
        loo[str(omit)]=rec
    ranges={}
    for k in [*METRICS,"self_focal_change_vs_base","focal_other_change_vs_base"]:
        vals=[v[k] for v in loo.values()]
        ranges[k]={"min":float(min(vals)),"max":float(max(vals)),"signs":sorted(set("positive" if x>0 else "negative" if x<0 else "zero" for x in vals))}
    return {"base_means":base_means,"seed_means":seed_means,"leave_one_seed_out":loo,"loo_ranges":ranges}

def scoring_equivalence(root):
    out={}
    for arch in ARCHS:
        maxdiff=0.0; token_sets={}
        for fn in ["base.json",*[f"mixed_s{s}.json" for s in SEEDS]]:
            d=load_json(root/arch/fn)
            for iid,row in d["items"].items():
                for key,mv in row["mass"].items():
                    xv=row["max"][key]
                    for a in "ABC": maxdiff=max(maxdiff,abs(float(mv["probs"][a])-float(xv["probs"][a])))
                rt=row["raw_token_logp"]["rel_self_first_0"]
                sig=tuple((a,tuple(rt[a]["eligible_ids"])) for a in "ABC")
                token_sets[str(sig)]=token_sets.get(str(sig),0)+1
        out[arch]={"max_abs_probability_difference_mass_vs_max":maxdiff,"eligible_token_set_signatures":token_sets}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=20260917)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    result={"analysis":"v0.7 submission diagnostics","status":"exploratory_posthoc","date":"2026-09-17","bootstrap_draws":a.bootstrap,"bootstrap_rng_seed":a.seed,"scoring":"mass"}
    for arch in ARCHS:
        basefiles,mixedfiles=realization_paths(a.root,arch)
        result[arch]={
            "base":{"absolute_performance":summarize_abs(basefiles,a.bootstrap,np.random.default_rng(a.seed+11)),"switch_rates":summarize_switch(basefiles,a.bootstrap,np.random.default_rng(a.seed+12))},
            "mixed":{"absolute_performance":summarize_abs(mixedfiles,a.bootstrap,np.random.default_rng(a.seed+21)),"switch_rates":summarize_switch(mixedfiles,a.bootstrap,np.random.default_rng(a.seed+22))},
            "seed_sensitivity":seed_sensitivity(a.root,arch),
        }
    result["scoring_convention_check"]=scoring_equivalence(a.root)
    jp=a.out_dir/"submission_diagnostics.json"; jp.write_text(json.dumps(result,indent=2))
    rows=[]
    for arch in ARCHS:
        for phase in ("base","mixed"):
            for owner,rec in result[arch][phase]["absolute_performance"].items():
                rows.append({"architecture":arch,"phase":phase,"ownership":owner,**rec})
    with (a.out_dir/"absolute_performance.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    sw=[]
    for arch in ARCHS:
        for phase in ("base","mixed"):
            for pair,rec in result[arch][phase]["switch_rates"].items(): sw.append({"architecture":arch,"phase":phase,"pair":pair,**rec})
    with (a.out_dir/"switch_rates.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sw[0].keys()); w.writeheader(); w.writerows(sw)
    ss_rows=[]
    for arch in ARCHS:
        ss=result[arch]["seed_sensitivity"]
        for seed,rec in ss["seed_means"].items():
            ss_rows.append({"architecture":arch,"kind":"seed_mean","seed":seed,**rec})
        for omit,rec in ss["leave_one_seed_out"].items():
            ss_rows.append({"architecture":arch,"kind":"leave_one_seed_out","seed":omit,**rec})
    fields=[]
    for row in ss_rows:
        for k in row:
            if k not in fields: fields.append(k)
    with (a.out_dir/"seed_sensitivity.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(ss_rows)
    lines=["# Exploratory v0.7 submission diagnostics","","Generated 2026-09-17 from frozen per-item outputs only; no new model inference.",""]
    for arch in ARCHS:
        lines += [f"## {arch.upper()}",""]
        for phase in ("base","mixed"):
            lines += [f"### {phase}","","| ownership | designated score | policy accuracy | mean S |","|---|---:|---:|---:|"]
            for owner,r in result[arch][phase]["absolute_performance"].items():
                lines.append(f"| {owner} | {r['mean_designated_score']:.4f} [{r['ci95_designated_score'][0]:.4f}, {r['ci95_designated_score'][1]:.4f}] | {r['policy_agreement_accuracy']:.4f} [{r['ci95_policy_agreement_accuracy'][0]:.4f}, {r['ci95_policy_agreement_accuracy'][1]:.4f}] | {r['mean_S_sensitivity']:.4f} [{r['ci95_mean_S_sensitivity'][0]:.4f}, {r['ci95_mean_S_sensitivity'][1]:.4f}] |")
            lines += ["","Matched switch rates:"]
            for pair,r in result[arch][phase]["switch_rates"].items(): lines.append(f"- {pair}: {r['switch_rate']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}], denominator {r['total_matched_variants_descriptive']} across {r['realizations']} realization(s).")
            lines.append("")
        ss=result[arch]["seed_sensitivity"]
        lines += ["### Leave-one-training-seed-out ranges",""]
        for k,v in ss["loo_ranges"].items(): lines.append(f"- {k}: {v['min']:+.4f} to {v['max']:+.4f}; signs {', '.join(v['signs'])}.")
        lines.append("")
    lines += ["## Scoring convention token check",""]
    for arch,r in result["scoring_convention_check"].items():
        lines.append(f"- {arch}: maximum absolute normalized probability difference between mass and legacy max records = {r['max_abs_probability_difference_mass_vs_max']:.3e}; observed eligible-token signatures = {len(r['eligible_token_set_signatures'])}.")
    (a.out_dir/"SUBMISSION_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
