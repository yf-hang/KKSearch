"""Run six-point searches, inspect preserved results and isolate the numeric learner."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits
from physics import POINTS, SECTORS, catalog, generate, formula
from validation import checks, evidence, review
from plot_tool import plot_training, write_report
from utils import PROJECT_ROOT, OUTPUT_FILES, load_config, sha256_file, write_json

def _isolated_training(work, matrix, settings):
    """Copy only the neural model and numeric trainer; keep physics outside."""
    package = work/"package"
    source_names = ("model.py", "train.py")
    for name in source_names:
        destination = package/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT/name, destination)
    inputs = work/"input"
    inputs.mkdir()
    np.savez_compressed(inputs/"matrix.npz", train=matrix)
    write_json(inputs/"config.json", settings)
    bootstrap = work/"worker.py"
    bootstrap.write_text('''import json, os, sys
from pathlib import Path
root = Path(__file__).resolve().parent
project = Path(sys.argv[1]).resolve()
dependencies = project/".venv"
# The installed editable package adds a virtual namespace path. The child
# uses private copies, so remove this project-specific import hook path.
sys.path[:] = [p for p in sys.path if not p.startswith("__editable__.neural_relations-")]
sys.path.insert(0, str(root/"package"))
opened, blocked = set(), []
allowed_roots = (root/"package", root/"input", root/"neural")
allowed_files = (root/"worker.py", root/"access.json")
def audit(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    original_project = path.is_relative_to(project) and not path.is_relative_to(dependencies)
    controller_files = path.is_relative_to(root) and path not in allowed_files and not any(path.is_relative_to(p) for p in allowed_roots)
    if original_project or controller_files:
        blocked.append(str(path))
        raise PermissionError("Training cannot read project or controller-only files")
    if path.is_relative_to(root):
        opened.add(str(path.relative_to(root)))
sys.addaudithook(audit)
try:
    from train import train_worker
    train_worker(root/"input", root/"neural")
finally:
    modules = {n: str(Path(m.__file__).resolve().relative_to(root))
               for n,m in sys.modules.copy().items()
               if n.split('.')[0] in {"model","train","main","physics","validation","plot_tool","utils","test_core"}
               and getattr(m,"__file__",None)}
    (root/"access.json").write_text(json.dumps({"opened_paths":sorted(opened),
        "blocked_accesses":blocked,"project_modules":modules},indent=2))
''')
    subprocess.run([sys.executable,"-I","-B",str(bootstrap),str(PROJECT_ROOT)],
                   cwd=work, check=True)
    search = json.loads((work/"neural/search.json").read_text())
    audit = json.loads((work/"access.json").read_text())
    audit.update({"passed":not audit["blocked_accesses"] and search["forbidden_modules_absent"],
                  "mechanism":"Minimal source package and Python file-access audit; not an OS sandbox",
                  "source_sha256":{name:sha256_file(package/name) for name in source_names},
                  "training_input_sha256":{p.name:sha256_file(p) for p in inputs.iterdir()},
                  "bootstrap_sha256":sha256_file(bootstrap),
                  "target_coefficients_supplied":False,"target_support_supplied":False,
                  "previous_checkpoint_loaded":False,"physics_or_proof_modules_available":False})
    if not audit["passed"]:
        raise RuntimeError("Training isolation audit failed")
    return search, audit


def run(out='results',config=None):
    cfg=load_config() if config is None else config
    if cfg.get('points')!=POINTS or cfg['data'].get('points')!=POINTS:
        raise ValueError('Only the current six-point configuration is supported')
    terms=catalog(cfg['warp_weights'],cfg['catalog_seed']);start=time.perf_counter()
    source_files=[PROJECT_ROOT/name for name in ('main.py','model.py','train.py','physics.py','validation.py','plot_tool.py','utils.py')]
    source_hashes={str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in source_files}
    # Matter-conserving mixed sectors: one or two scalar pairs and the remaining gravity states.
    # All profile sectors remain in the generated bank and initial coverage.
    eligible=[i for i,states in enumerate(SECTORS) if states.count('s') in (2,4)]
    settings={**cfg['search'],'seed':cfg['seed'],'alphabet':cfg['coefficients'],'max_terms':cfg['max_terms'],
              'eligible_contexts':eligible}
    with tempfile.TemporaryDirectory(prefix='neural-search-') as tmp,threadpool_limits(limits=1):
        work=Path(tmp)
        manifest=generate(work/'data',cfg['data'],terms)
        with np.load(work/'data/data.npz') as data: train=data['train_matrix']
        search,audit=_isolated_training(work,train,settings)
        selected=[]
        with np.load(work/'data/data.npz') as data:
            for c in search['candidates']:
                if not c['passed'] or c['context'] not in eligible: continue
                val=evidence(data['validation_matrix'][c['context']],c['coefficients'])
                if val['passed']: selected.append({**c,'validation':val})
        frozen={'candidates':selected,'dataset_sha256':manifest['npz_sha256']}
        write_json(work/'frozen.json',frozen);frozen_hash=sha256_file(work/'frozen.json')
        verified=[]
        for c in selected:
            tests=checks(work/'data/data.npz',c,terms)
            if all(t['passed'] for t in tests.values()): verified.append({**c,'formula':formula(terms,c['context'],c['coefficients']),'checks':tests,'novelty_review':review(terms,SECTORS[c['context']],c['coefficients'])})
        status='literature_review_required' if verified else 'no_verified_candidate'
        result={'schema':8,'task':'bosonic_six_point_search','status':status,
                'accepted':False,'numerically_verified':bool(verified),'novel_candidates':[],
                'config':cfg,'catalog':terms,'dataset':manifest,'isolation_audit':audit,
                'search':{k:v for k,v in search.items() if k not in ('history','candidates')},'best_training_proposals':search['candidates'][:8],
                'frozen_candidates':frozen,'frozen_candidates_sha256':frozen_hash,'verified_candidates':verified,
                'scope':'Matter and gravity profiles covered; each candidate combines terms within one fixed external-profile sector; no full amplitudes or spectral sums claimed',
                'source_sha256':source_hashes,
                'total_seconds':time.perf_counter()-start,'literature_comparison':'Post-freeze derivability analysis does not decide whether the explicit six-point equations appear in the references. Literature occurrence requires a separate comparison; deriving a conclusion from published premises is allowed.'}
        stage=work/'publish';stage.mkdir()
        shutil.copy2(work/'data/data.npz',stage/'dataset.npz');shutil.copy2(work/'neural/policy.pt',stage/'model.pt')
        result['model_sha256']=sha256_file(stage/'model.pt');write_json(stage/'result.json',result)
        with (stage/'training.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(search['history'][0]));writer.writeheader();writer.writerows(search['history'])
        os.environ['MPLCONFIGDIR']=str(work/'mpl')
        plot_training(search['history'],stage);write_report(stage,result)
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        for name in OUTPUT_FILES: shutil.copy2(stage/name,out/name)
    return result


def current_result(folder):
    result=json.loads((Path(folder)/'result.json').read_text())
    if result.get('task')!='bosonic_six_point_search':
        raise ValueError('This result belongs to a different search. Run six-point training first')
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent bosonic six-point search")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run", help="Generate data, search and verify; keep only the latest result")
    train.add_argument("--out", default="results", help="Output directory; the six current output files are replaced")
    sample = sub.add_parser("sample", help="Sample the current model")
    sample.add_argument("--checkpoint", default="results/model.pt")
    sample.add_argument("--count", type=int, default=20)
    sample.add_argument("--seed", type=int, default=2026)
    verify = sub.add_parser("verify", help="Recheck a frozen candidate on held-out data")
    verify.add_argument("--run", default="results")
    verify.add_argument("--coefficients", help="Optional JSON vector; defaults to the first verified candidate")
    verify.add_argument("--high-precision", action="store_true")
    verify.add_argument("--context", type=int, help="External-profile sector for an explicit coefficient vector")
    report = sub.add_parser("report", help="Regenerate the current report")
    report.add_argument("--run", default="results")
    arguments=sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ["run"])
    try:
        if args.command == "run":
            print(f"Output: {Path(args.out).resolve()}", flush=True)
            result = run(args.out)
            print(result["status"])
            for candidate in result["verified_candidates"]:
                print(candidate["formula"])
        elif args.command == "sample":
            if args.count < 1:
                parser.error("--count must be positive")
            import torch
            from model import load_policy
            torch.set_num_threads(1);torch.manual_seed(args.seed)
            model, _ = load_policy(args.checkpoint)
            result = current_result(Path(args.checkpoint).parent)
            if model.terms!=len(result['catalog']) or model.contexts!=len(result['dataset']['sectors']):
                raise ValueError('Checkpoint dimensions do not match the six-point result')
            # Re-sample with context-bearing tokens so each printed formula has
            # the actual external-state content used by the network.
            with torch.no_grad():
                c,tokens=model.sample(args.count)
            counts=Counter((int(t[-1]),tuple(row)) for row,t in zip(c.tolist(),tokens.tolist()))
            print(json.dumps([{"context":ctx,"coefficients":list(row),"formula":formula(result['catalog'],ctx,row),"count":n}
                              for (ctx,row),n in counts.most_common()],indent=2))
        elif args.command == "verify":
            from threadpoolctl import threadpool_limits
            folder = Path(args.run)
            result = current_result(folder)
            if sha256_file(folder/"dataset.npz") != result["dataset"]["npz_sha256"]:
                raise ValueError("Dataset hash mismatch")
            if args.coefficients is None:
                if not result["verified_candidates"]:
                    raise ValueError("No verified candidate; provide --coefficients to examine another vector")
                c = result["verified_candidates"][0]["coefficients"]
            else:
                c = json.loads(args.coefficients)
            if not isinstance(c,list) or len(c) != len(result["catalog"]):
                raise ValueError("Expected a coefficient vector matching the current catalog")
            context=args.context
            if context is None:
                if not result['verified_candidates']:
                    raise ValueError('Provide --context for an explicit vector')
                context=result['verified_candidates'][0]['context']
            with threadpool_limits(limits=1):
                check=checks(folder/'dataset.npz',{'context':context,'coefficients':c},result['catalog'],independent=args.high_precision)
            print(json.dumps(check,indent=2))
        elif args.command == "report":
            folder = Path(args.run)
            write_report(folder,current_result(folder))
            print(folder/"report.md")
    except (FileNotFoundError,ValueError) as error:
        parser.error(str(error)+"\nRun training first with: python main.py")


if __name__ == "__main__":
    main()
