"""Canonical Phenix validation for GT, AF3, AT-AF3 and corrected structures.

Runs on the Linux Phenix environment.  It is read-only with respect to input
structures and writes an independent results/gt_comparison_phenix_ccdfix
directory. Prediction structures use the highest ranking sample per case/seed.
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, subprocess
from pathlib import Path
from collections import defaultdict
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
GT_ROOT=ROOT/"data"/"gt_reference"
DEFAULT_PHENIX_PYTHON=os.environ.get("PHENIX_PYTHON", "phenix.python")
DEFAULT_MOLPROBITY=os.environ.get("PHENIX_MOLPROBITY", "phenix.molprobity")
ALL=("7P5R_2","7QDI_1","8AVR_2","8BFD_2","8GQP_1","8JH6_1","8P6Q_2")
LOCAL=("7OOJ_1","7P0A_3","7P0T_3","7W40_6","8HR3_1","8HR4_1","8U78_1","8UM9_1","9BIC_1")
SEEDS=(1373158607,239081664,53710185,1592467582,590620972,525901257,479341424,299655413,1581559893,220106708,1453201079,1590571866,1915941033,1171165723,186699714,1268073013,906070221,68252794,63989048,201209006,469521478,499635469,1085242217,1292825379,56985562,1205264596,427000597,1537640409,1395616197,1506083911)
METRICS=("molprobity_score","clashscore","ramachandran_favored_percent","ramachandran_outliers_percent","rotamer_outliers_percent","cbeta_deviations_count","bond_length_rmsd","bond_angle_rmsd","bond_rmsz","bond_outlier_rate_pct")

def choose(root,case,seed):
    sd=root/case/f"seed_{seed}"; ranks=list(sd.rglob("*_ranking_scores.csv")); models=list(sd.rglob("*_model.cif"))
    if len(ranks)!=1 or len(models)!=6: raise RuntimeError(f"inventory {sd}: ranking={len(ranks)} models={len(models)}")
    with ranks[0].open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
    sample=max(rows,key=lambda r:float(r["ranking_score"]))["sample"]
    found=[p for p in models if f"_sample-{sample}_model.cif" in p.name]
    if len(found)!=1: raise RuntimeError(f"selected sample not unique: {sd}")
    return found[0]

def parse_log(text):
    def f(pattern):
        m=re.search(pattern,text,re.I|re.M); return float(m.group(1)) if m else math.nan
    return {
      "molprobity_score":f(r"MolProbity\s+score\s*=\s*([0-9.]+)"), "clashscore":f(r"Clashscore\s*=\s*([0-9.]+)"),
      "ramachandran_favored_percent":f(r"^\s*favored\s*=\s*([0-9.]+)\s*%"), "ramachandran_outliers_percent":f(r"Ramachandran outliers\s*=\s*([0-9.]+)\s*%"),
      "rotamer_outliers_percent":f(r"Rotamer outliers\s*=\s*([0-9.]+)\s*%"), "cbeta_deviations_count":f(r"C-beta deviations\s*=\s*([0-9.]+)"),
      "bond_length_rmsd":f(r"RMS\(bonds\)\s*=\s*([0-9.]+)"), "bond_angle_rmsd":f(r"RMS\(angles\)\s*=\s*([0-9.]+)"),
    }

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=ROOT/"results/gt_comparison_phenix_ccdfix"); ap.add_argument("--phenix-python",default=DEFAULT_PHENIX_PYTHON); ap.add_argument("--molprobity",default=DEFAULT_MOLPROBITY); ap.add_argument("--all-d-standard-root",type=Path,default=ROOT/"runs_master42_n30/standard_af3"); ap.add_argument("--all-d-at-root",type=Path,default=ROOT/"runs_master42_n30/at_af3"); ap.add_argument("--all-d-corrected-root",type=Path,default=ROOT/"runs_master42_n30/at_af3_chirality_corrected"); a=ap.parse_args()
    out=a.output_dir; pdbdir=out/"canonical_pdb"; logs=out/"molprobity_logs"; pdbdir.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
    # CCTBX conversion is executed inside Phenix Python because the host venv
    # does not provide the canonical hierarchy writer.
    manifest=[]
    for ds,cases,std,at,corr in [("All-D",ALL,a.all_d_standard_root,a.all_d_at_root,a.all_d_corrected_root),("Local-D",LOCAL,ROOT/"runs/local_d_curated_v1_1_post_cutoff/standard_af3",ROOT/"runs/local_d_curated_v1_1_post_cutoff/at_af3_first_pass",ROOT/"runs/local_d_curated_v1_1_post_cutoff/corrected")]:
      for case in cases:
        gt=GT_ROOT/("all_d" if ds=="All-D" else "local_d")/(case.split("_")[0]+".cif")
        manifest.append((ds,case,0,"GT",gt))
        for seed in SEEDS:
          corrected=corr/case/f"seed_{seed}"/"corrected_all_errors.cif"
          if not corrected.is_file(): raise SystemExit(f"MISSING corrected structure: {corrected}")
          manifest += [(ds,case,seed,"AF3",choose(std,case,seed)),(ds,case,seed,"AT-AF3",choose(at,case,seed)),(ds,case,seed,"Corrected AT-AF3",corrected)]
    man=out/"gt_phenix_manifest.csv"; write_csv(man,[{"dataset_type":d,"case":c,"seed":s,"condition":k,"input_cif":str(p)} for d,c,s,k,p in manifest])
    convert=f'''import sys, json, math\nfrom pathlib import Path\nfrom scripts.analyze_bond_geometry import make_geometry_manager\nman=Path(sys.argv[1]); out=Path(sys.argv[2]); bond_out={{}}\nimport csv\ndef stats(path):\n h,proxies=make_geometry_manager(path); sites=h.atoms().extract_xyz(); zs=[]\n for p in proxies:\n  i,j=map(int,p.i_seqs)\n  if i>=len(sites) or j>=len(sites): continue\n  w=float(p.weight); ideal=float(p.distance_ideal)\n  if w<=0: continue\n  d=math.sqrt(sum((float(sites[i][q])-float(sites[j][q]))**2 for q in range(3))); zs.append((d-ideal)*math.sqrt(w))\n if not zs: return {{"bond_rmsz":float("nan"),"bond_outlier_rate_pct":float("nan"),"bond_n":0}}\n return {{"bond_rmsz":math.sqrt(sum(z*z for z in zs)/len(zs)),"bond_outlier_rate_pct":100.0*sum(abs(z)>5 for z in zs)/len(zs),"bond_n":len(zs)}}\nfor r in csv.DictReader(man.open(newline='',encoding='utf-8')):\n p=Path(r['input_cif']); target=out/f"{{r['dataset_type'].replace('-', '_')}}_{{r['case']}}_seed_{{r['seed']}}_{{r['condition'].replace(' ','_').replace('-','_')}}.pdb"\n if not p.is_file(): raise SystemExit(f"MISSING {{p}}")\n if not target.exists():\n  h,_=make_geometry_manager(p); h.write_pdb_file(file_name=str(target))\n if target.stat().st_size==0: raise SystemExit(f"EMPTY {{target}}")\n bond_out[str(target)]=stats(p)\n(out/"bond_stats.json").write_text(json.dumps(bond_out))\nprint('canonical PDB conversion complete')\n'''
    subprocess.run([a.phenix_python,"-c",convert,str(man),str(pdbdir)],check=True)
    # Rewrite every canonical PDB from its CIF source and remove ANISOU lines
    # before MolProbity.
    rewrite = '''import csv, sys, json
from pathlib import Path
from scripts.analyze_bond_geometry import make_geometry_manager
man=Path(sys.argv[1]); out=Path(sys.argv[2])
alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
mapping={}
for r in csv.DictReader(man.open(newline="", encoding="utf-8")):
 p=Path(r["input_cif"])
 target=out/f"{r['dataset_type'].replace('-', '_')}_{r['case']}_seed_{r['seed']}_{r['condition'].replace(' ','_').replace('-','_')}.pdb"
 h,_=make_geometry_manager(p)
 ids=[]
 for model in h.models():
  for chain in model.chains():
   if chain.id not in ids: ids.append(chain.id)
 if len(ids)>len(alphabet): raise RuntimeError(f"too many chains for PDB single-character IDs: {p}")
 local={old:alphabet[i] for i,old in enumerate(ids)}
 for model in h.models():
  for chain in model.chains(): chain.id=local[chain.id]
 mapping[target.name]={"source":str(p),"chain_id_map":local}
 h.write_pdb_file(file_name=str(target))
 lines=target.read_text(errors="replace").splitlines(True)
 target.write_text("".join(line for line in lines if not line.startswith("ANISOU")), encoding="utf-8")
(out.parent/"canonical_chain_id_mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
print("canonical PDB rewrite complete")
'''
    subprocess.run([a.phenix_python,"-c",rewrite,str(man),str(pdbdir)],check=True)
    stats_path=out/"bond_stats.json"
    # The embedded conversion writes bond_stats.json beside the canonical PDBs.
    if not stats_path.is_file() and (pdbdir/"bond_stats.json").is_file():
        stats_path=pdbdir/"bond_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"STOP canonical conversion did not write {out/'bond_stats.json'}")
    bond_stats=json.loads(stats_path.read_text())
    rows=[]
    for d,c,s,k,p in manifest:
      stem=f"{d.replace('-','_')}_{c}_seed_{s}_{k.replace(' ','_').replace('-','_')}"; pdb=pdbdir/(stem+".pdb"); log=logs/(stem+".log")
      if not log.exists() or "Summary" not in log.read_text(errors="replace"):
        with log.open("w",encoding="utf-8") as h: subprocess.run([a.molprobity,str(pdb)],stdout=h,stderr=subprocess.STDOUT,check=False,timeout=1200)
      vals=parse_log(log.read_text(errors="replace")); vals.update(bond_stats[str(pdb)])
      # Some standard MolProbity categories are legitimately unavailable for
      # structures containing only non-standard/D residues.  Keep those fields
      # as NaN instead of marking the entire structure as a parse failure.
      parsed = sum(math.isfinite(float(vals[x])) for x in METRICS if x in vals)
      vals.update({"dataset_type":d,"case":c,"seed":s,"condition":k,"structure_file":str(pdb),"log_file":str(log),"status":"success" if parsed else "failed_parse"})
      rows.append(vals)
    write_csv(out/"per_structure_molprobity.csv",rows)
    by={(r["dataset_type"],r["case"],int(r["seed"]),r["condition"]):r for r in rows}; gt_by={(r["dataset_type"],r["case"]):r for r in rows if r["condition"]=="GT"}; paired=[]
    prediction_keys=sorted({(r["dataset_type"],r["case"],int(r["seed"])) for r in rows if r["condition"]!="GT"})
    for key in prediction_keys:
      base={"dataset_type":key[0],"case":key[1],"seed":key[2]}; gt=gt_by[key[:2]]
      for cond in ("AF3","AT-AF3","Corrected AT-AF3"):
        q=by[key+(cond,)]
        for metric in vals:
          if metric in {"dataset_type","case","seed","condition","structure_file","log_file","status"}: continue
          base[f"{cond}_{metric}"]=q[metric]; base[f"{cond}_{metric}_abs_error_vs_gt"]=abs(float(q[metric])-float(gt[metric])) if math.isfinite(float(q[metric])) and math.isfinite(float(gt[metric])) else math.nan
      for metric in METRICS:
        base[f"AF3_minus_AT-AF3_{metric}_error"]=float(base[f"AF3_{metric}_abs_error_vs_gt"])-float(base[f"AT-AF3_{metric}_abs_error_vs_gt"])
        base[f"AT-AF3_minus_Corrected_{metric}_error"]=float(base[f"AT-AF3_{metric}_abs_error_vs_gt"])-float(base[f"Corrected AT-AF3_{metric}_abs_error_vs_gt"])
      paired.append(base)
    write_csv(out/"paired_vs_gt.csv",paired)
    groups=defaultdict(list)
    for r in paired: groups[(r["dataset_type"],r["case"])].append(r)
    summary=[]
    for (ds,c),g in sorted(groups.items()):
      z={"dataset_type":ds,"case":c,"n_seeds":len(g)}
      for metric in METRICS:
       for label in ("AF3","AT-AF3","Corrected AT-AF3"):
        key=f"{label}_{metric}_abs_error_vs_gt"
        x=[float(r[key]) for r in g if r[key] not in ("", "nan", "NaN", None) and math.isfinite(float(r[key]))]
        z[key+"_n_valid"]=len(x)
        z[key+"_mean"]=statistics.mean(x) if x else math.nan
        z[key+"_sd"]=statistics.stdev(x) if len(x)>1 else (0.0 if x else math.nan)
       for label in ("AF3_minus_AT-AF3","AT-AF3_minus_Corrected"):
        key=f"{label}_{metric}_error"
        x=[float(r[key]) for r in g if r[key] not in ("", "nan", "NaN", None) and math.isfinite(float(r[key]))]
        z[key+"_n_valid"]=len(x)
        z[key+"_mean"]=statistics.mean(x) if x else math.nan
        z[key+"_sd"]=statistics.stdev(x) if len(x)>1 else (0.0 if x else math.nan)
      summary.append(z)
    write_csv(out/"per_case_summary.csv",summary)
    (out/"summary.json").write_text(json.dumps({"comparison":"GT vs AF3, AT-AF3 and corrected AT-AF3","n_validation_structures":len(rows),"n_paired_seed_rows":len(paired),"metrics":METRICS,"error_definition":"absolute prediction metric minus GT metric","improvement_definition":"AF3 error - AT-AF3 error; AT-AF3 error - corrected error","warning":"D-residue Ramachandran, rotamer and C-beta values may be not evaluable in standard MolProbity; zero is not interpreted as perfect without evaluable-residue counts"},indent=2)+"\n")
    print(f"GT Phenix validation complete: {len(rows)} structures, {len(paired)} pairs")
if __name__=="__main__": raise SystemExit(main())
