"""Compact PNG plots of the current training history."""
from pathlib import Path


def plot_training(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([r["step"] for r in history], [r["loss"] for r in history], label="Total loss")
    axis.set(xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=.2)
    axis.legend()
    fig.savefig(Path(out)/"loss.png", dpi=160)
    plt.close(fig)


def write_report(out,result):
    sectors=result['dataset']['sectors']
    lines=['# Bosonic six-point neural search','',f"Status: {result['status']}",
           f"Profiles: matter s; gravity u, v, w; {len(sectors)} external-profile sectors; {len(result['catalog'])} terms per sector.",
           'Search is for overlap-integral identities, not full helicity amplitudes. A candidate currently stays within one external-profile sector.',
           'Training includes allowed mode-zero states and the radion. Independent zero-mode checks apply to the profiles in the accepted candidate.',
           f"Training: {result['search']['completed_steps']} steps, {result['search']['seconds']:.2f} seconds.",
           f"Data generation, search and verification: {result['total_seconds']:.2f} seconds (excludes output publication and earlier tuning).",
           f"First successful proposal: {result['search'].get('first_success_proposal','unrecorded')}.",
           'Numerical validity, derivability from published premises, and explicit occurrence in the papers are separate questions.',
           f"Literature-confirmed new relations: {len(result.get('novel_candidates',[]))}; unreviewed candidates are pending, not rejected.",
           'Literature comparison: '+str(result['literature_comparison']),'']
    for c in result['verified_candidates']:
        lines += [str(sectors[c['context']]),'Review: '+c.get('novelty_review',{}).get('status','pending'),'```text',c['formula'],'```',
                  'Mass labels M0...M5 are zero-based external slots; field labels 1...6 are one-based. Derivatives and masses in this machine formula are dimensionless.',
                  '', '| Check | Max relative residual | Active fraction | Passed |', '|---|---:|---:|---|']
        for name,score in c['checks'].items():
            lines.append(f"| {name} | {score.get('max_relative',float('nan')):.3e} | {score.get('active_fraction',0):.3f} | {score['passed']} |")
        lines += ['', 'Zero-mode checks count degenerate 0=0 rows separately and also check the floored absolute residual.', '']
    lines+=['![loss](loss.png)','']
    (Path(out)/'report.md').write_text('\n'.join(lines))

