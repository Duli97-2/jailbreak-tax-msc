"""
Phase 6: Analysis and visualization.

Computes:
  - Per-arm mean utility with 95% bootstrap confidence intervals
  - JTax per arm (relative to benign reference)
  - Per-dimension breakdown (Coherence, Persuasiveness, Specificity, Functional Completeness)
  - Per-source breakdown (JBB vs malware)
  - Category-level heatmaps

Produces figures saved to outputs/figures/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

##### Config #####
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N    = 5000
CI_LEVEL       = 0.95
RANDOM_SEED    = 42
np.random.seed(RANDOM_SEED)

##### Load results #####
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

baseline = load_json("outputs/baseline_results.json")     # benign ref
pair     = load_json("outputs/pair_results.json")          # PAIR arm
lora     = load_json("outputs/lora_results.json")          # LoRA raw
lora_adv = load_json("outputs/lora_adversarial_results.json")  # LoRA adv

print(f"Loaded: baseline={len(baseline)} pair={len(pair)} lora={len(lora)} lora_adv={len(lora_adv)}")

##### Helper: extract utilities #####
def get_utils(arm_data):
    """Return list of utility floats from a results file."""
    utils = []
    for r in arm_data:
        s = r.get("scores", {})
        # baseline/benign stores utility nested, others top-level
        u = r.get("utility")
        if u is None:
            u = s.get("utility") if isinstance(s, dict) else None
        if u is not None:
            utils.append(u)
    return np.array(utils)

def get_dim_scores(arm_data):
    """Return dict of per-dimension score arrays."""
    dims = {"coherence": [], "persuasiveness": [], "specificity": [], "functional_completeness": []}
    for r in arm_data:
        s = r.get("scores")
        if not isinstance(s, dict):
            continue
        for d in dims:
            if d in s:
                dims[d].append(s[d])
    return {d: np.array(v) for d, v in dims.items()}

##### Bootstrap CI #####
def bootstrap_ci(data, n=BOOTSTRAP_N, level=CI_LEVEL):
    """Return (mean, lo, hi) via bootstrap resampling."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    means = [np.random.choice(data, size=len(data), replace=True).mean() for _ in range(n)]
    means = np.array(means)
    lo = np.percentile(means, (1 - level) / 2 * 100)
    hi = np.percentile(means, (1 + level) / 2 * 100)
    return data.mean(), lo, hi

##### Extract utility arrays #####
u_baseline = get_utils(baseline)
u_pair     = get_utils(pair)
u_lora     = get_utils(lora)
u_loraadv  = get_utils(lora_adv)

print(f"\nSample sizes: baseline={len(u_baseline)} pair={len(u_pair)} lora={len(u_lora)} lora_adv={len(u_loraadv)}")

##### Per-arm CIs #####
arms = {
    "Baseline (benign ref)": u_baseline,
    "PAIR on aligned Phi-3": u_pair,
    "LoRA + raw goals":      u_lora,
    "LoRA + PAIR prompts":   u_loraadv,
}

print(f"\n{'='*60}")
print(f"{'Arm':<30} {'Mean':>8} {'95% CI':>20}")
print(f"{'='*60}")
summary = {}
for name, data in arms.items():
    mean, lo, hi = bootstrap_ci(data)
    summary[name] = {"mean": mean, "lo": lo, "hi": hi, "n": len(data)}
    print(f"{name:<30} {mean:>8.3f}   [{lo:.3f}, {hi:.3f}]")

##### JTax #####
base_mean = u_baseline.mean()
print(f"\n{'='*60}")
print(f"JTax = (BaseUtil - ArmUtil) / BaseUtil")
print(f"BaseUtil (benign reference) = {base_mean:.3f}")
print(f"{'='*60}")
for name in ["PAIR on aligned Phi-3", "LoRA + raw goals", "LoRA + PAIR prompts"]:
    arm_mean = summary[name]["mean"]
    jtax     = (base_mean - arm_mean) / base_mean
    print(f"{name:<30} JTax = {jtax:+.3f}")

##### Per-dimension breakdown #####
print(f"\n{'='*60}")
print(f"PER-DIMENSION BREAKDOWN (mean 1-5 scores)")
print(f"{'='*60}")
print(f"{'Arm':<30} {'Coh':>6} {'Per':>6} {'Spe':>6} {'FC':>6}")
for name, arm_data in [
    ("Baseline (benign ref)", baseline),
    ("PAIR on aligned Phi-3", pair),
    ("LoRA + raw goals",      lora),
    ("LoRA + PAIR prompts",   lora_adv),
]:
    dims = get_dim_scores(arm_data)
    print(f"{name:<30} "
          f"{dims['coherence'].mean():>6.2f} "
          f"{dims['persuasiveness'].mean():>6.2f} "
          f"{dims['specificity'].mean():>6.2f} "
          f"{dims['functional_completeness'].mean():>6.2f}")

##### Per-source breakdown (JBB vs malware for harmful-arm data) #####
def split_by_source(arm_data):
    """Split arm into JBB and malware subsets."""
    jbb = [r.get("utility") or r.get("scores", {}).get("utility")
           for r in arm_data if r.get("source") == "JBB"]
    mal = [r.get("utility") or r.get("scores", {}).get("utility")
           for r in arm_data if r.get("source") == "manual_malware"]
    return np.array([x for x in jbb if x is not None]), np.array([x for x in mal if x is not None])

print(f"\n{'='*60}")
print(f"PER-SOURCE UTILITY (JBB vs malware)")
print(f"{'='*60}")
print(f"{'Arm':<30} {'JBB':>12} {'Malware':>12}")
for name, arm_data in [
    ("PAIR on aligned Phi-3", pair),
    ("LoRA + raw goals",      lora),
    ("LoRA + PAIR prompts",   lora_adv),
]:
    j, m = split_by_source(arm_data)
    print(f"{name:<30} {j.mean():>12.3f} {m.mean():>12.3f}")

##### Figure 1: Mean utility per arm with CI bars #####
fig, ax = plt.subplots(figsize=(9, 5))
names = list(arms.keys())
means = [summary[n]["mean"] for n in names]
los   = [summary[n]["mean"] - summary[n]["lo"] for n in names]
his   = [summary[n]["hi"] - summary[n]["mean"] for n in names]
colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]

ax.bar(names, means, yerr=[los, his], capsize=6, color=colors, edgecolor="black")
ax.axhline(base_mean, ls="--", color="gray", alpha=0.6, label=f"Baseline = {base_mean:.3f}")
ax.set_ylabel("Mean utility")
ax.set_title("Utility across arms (with 95% bootstrap CI)")
ax.set_ylim(0, 0.65)
ax.legend()
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_mean_utility.png", dpi=150)
print(f"\n✓ Saved fig1_mean_utility.png")

##### Figure 2: Per-dimension comparison #####
dims_all = {name: get_dim_scores(data) for name, data in [
    ("Baseline (benign ref)", baseline),
    ("PAIR on aligned Phi-3", pair),
    ("LoRA + raw goals",      lora),
    ("LoRA + PAIR prompts",   lora_adv),
]}

dim_labels = ["Coherence", "Persuasiveness", "Specificity", "Functional\nCompleteness"]
dim_keys   = ["coherence", "persuasiveness", "specificity", "functional_completeness"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dim_labels))
width = 0.2

for i, (name, color) in enumerate(zip(dims_all.keys(), colors)):
    vals = [dims_all[name][k].mean() for k in dim_keys]
    ax.bar(x + i * width, vals, width, label=name, color=color, edgecolor="black")

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(dim_labels)
ax.set_ylabel("Mean dimension score (1-5)")
ax.set_title("Per-dimension quality across arms")
ax.set_ylim(0, 5.5)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_per_dimension.png", dpi=150)
print(f"✓ Saved fig2_per_dimension.png")

##### Figure 3: JBB vs malware per arm #####
fig, ax = plt.subplots(figsize=(9, 5))
attack_arms = [
    ("PAIR",           pair),
    ("LoRA + raw",     lora),
    ("LoRA + adv",     lora_adv),
]
arm_labels = [a[0] for a in attack_arms]
jbb_means  = [split_by_source(a[1])[0].mean() for a in attack_arms]
mal_means  = [split_by_source(a[1])[1].mean() for a in attack_arms]

x = np.arange(len(arm_labels))
width = 0.35
ax.bar(x - width/2, jbb_means, width, label="JBB (30 behaviors)",     color="#4C72B0", edgecolor="black")
ax.bar(x + width/2, mal_means, width, label="Malware (10 behaviors)", color="#C44E52", edgecolor="black")
ax.axhline(base_mean, ls="--", color="gray", alpha=0.6, label=f"Benign baseline = {base_mean:.3f}")
ax.set_xticks(x)
ax.set_xticklabels(arm_labels)
ax.set_ylabel("Mean utility")
ax.set_title("Utility by evaluation source (JBB vs malware)")
ax.set_ylim(0, 0.65)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_jbb_vs_malware.png", dpi=150)
print(f"✓ Saved fig3_jbb_vs_malware.png")

##### Save summary JSON #####
summary_out = {
    "baseline_mean": float(base_mean),
    "arms": {name: {"mean": float(s["mean"]), "ci_lo": float(s["lo"]), "ci_hi": float(s["hi"]), "n": s["n"]}
             for name, s in summary.items()},
    "jtax": {
        "pair":       float((base_mean - summary["PAIR on aligned Phi-3"]["mean"]) / base_mean),
        "lora_raw":   float((base_mean - summary["LoRA + raw goals"]["mean"])      / base_mean),
        "lora_adv":   float((base_mean - summary["LoRA + PAIR prompts"]["mean"])   / base_mean),
    },
}
with open("outputs/analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2)

print(f"\n>>>>>>>>>>>> Phase 6 complete.")
print(f"  Summary JSON: outputs/analysis_summary.json")
print(f"  Figures     : outputs/figures/")