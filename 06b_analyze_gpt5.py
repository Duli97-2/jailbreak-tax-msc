"""
Phase 6b: Analysis and visualization using gpt-5 as judge.

Mirrors 06_analyze.py but reads from the _gpt5 result files.
Also produces side-by-side comparison figures showing gpt-5-mini vs gpt-5
judgments, which is a judge-robustness contribution for the report.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

##### Config #####
FIGURES_DIR = Path("outputs/figures_gpt5")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N    = 5000
CI_LEVEL       = 0.95
RANDOM_SEED    = 42
np.random.seed(RANDOM_SEED)

##### Helpers #####
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_utils(arm_data, key="utility"):
    utils = []
    for r in arm_data:
        u = r.get(key)
        if u is None:
            s = r.get("scores", {})
            u = s.get("utility") if isinstance(s, dict) else None
        if u is not None:
            utils.append(u)
    return np.array(utils)

def get_dim_scores(arm_data):
    dims = {"coherence": [], "persuasiveness": [], "specificity": [], "functional_completeness": []}
    for r in arm_data:
        s = r.get("scores")
        if not isinstance(s, dict):
            continue
        for d in dims:
            if d in s:
                dims[d].append(s[d])
    return {d: np.array(v) for d, v in dims.items()}

def bootstrap_ci(data, n=BOOTSTRAP_N, level=CI_LEVEL):
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    means = [np.random.choice(data, size=len(data), replace=True).mean() for _ in range(n)]
    means = np.array(means)
    lo = np.percentile(means, (1 - level) / 2 * 100)
    hi = np.percentile(means, (1 + level) / 2 * 100)
    return data.mean(), lo, hi

def split_by_source(arm_data, key="utility"):
    jbb, mal = [], []
    for r in arm_data:
        u = r.get(key)
        if u is None:
            s = r.get("scores", {})
            u = s.get("utility") if isinstance(s, dict) else None
        if u is None:
            continue
        if r.get("source") == "JBB":
            jbb.append(u)
        elif r.get("source") == "manual_malware":
            mal.append(u)
    return np.array(jbb), np.array(mal)

##### Load gpt-5 files #####
baseline_g5 = load_json("outputs/baseline_results_gpt5.json")
pair_g5     = load_json("outputs/pair_results_gpt5.json")
lora_g5     = load_json("outputs/lora_results_gpt5.json")
loraadv_g5  = load_json("outputs/lora_adversarial_results_gpt5.json")

# Also load gpt-5-mini files for comparison
baseline_mini = load_json("outputs/baseline_results.json")
pair_mini     = load_json("outputs/pair_results.json")
lora_mini     = load_json("outputs/lora_results.json")
loraadv_mini  = load_json("outputs/lora_adversarial_results.json")

print(f"Loaded gpt-5 files:     baseline={len(baseline_g5)} pair={len(pair_g5)} lora={len(lora_g5)} lora_adv={len(loraadv_g5)}")

##### Extract utilities (gpt-5) #####
u_baseline = get_utils(baseline_g5)
u_pair     = get_utils(pair_g5)
u_lora     = get_utils(lora_g5)
u_loraadv  = get_utils(loraadv_g5)

# gpt-5-mini utilities (for comparison)
m_baseline = get_utils(baseline_mini)
m_pair     = get_utils(pair_mini)
m_lora     = get_utils(lora_mini)
m_loraadv  = get_utils(loraadv_mini)

##### Per-arm CIs under gpt-5 #####
arms = {
    "Baseline (benign ref)": u_baseline,
    "PAIR on aligned Phi-3": u_pair,
    "LoRA + raw goals":      u_lora,
    "LoRA + PAIR prompts":   u_loraadv,
}

print(f"\n{'='*60}")
print(f"GPT-5 JUDGE RESULTS")
print(f"{'Arm':<30} {'Mean':>8} {'95% CI':>20}")
print(f"{'='*60}")
summary = {}
for name, data in arms.items():
    mean, lo, hi = bootstrap_ci(data)
    summary[name] = {"mean": mean, "lo": lo, "hi": hi, "n": len(data)}
    print(f"{name:<30} {mean:>8.3f}   [{lo:.3f}, {hi:.3f}]")

##### JTax (gpt-5) #####
base_mean = u_baseline.mean()
print(f"\n{'='*60}")
print(f"JTax = (BaseUtil - ArmUtil) / BaseUtil  (gpt-5)")
print(f"BaseUtil = {base_mean:.3f}")
print(f"{'='*60}")
jtax_g5 = {}
for name in ["PAIR on aligned Phi-3", "LoRA + raw goals", "LoRA + PAIR prompts"]:
    arm_mean = summary[name]["mean"]
    jtax     = (base_mean - arm_mean) / base_mean
    jtax_g5[name] = jtax
    print(f"{name:<30} JTax = {jtax:+.3f}")

##### JTax comparison (gpt-5-mini vs gpt-5) #####
base_mini = m_baseline.mean()
jtax_mini = {
    "PAIR on aligned Phi-3": (base_mini - m_pair.mean())    / base_mini,
    "LoRA + raw goals":      (base_mini - m_lora.mean())    / base_mini,
    "LoRA + PAIR prompts":   (base_mini - m_loraadv.mean()) / base_mini,
}

print(f"\n{'='*60}")
print(f"JTax COMPARISON ACROSS JUDGES")
print(f"{'Arm':<30} {'gpt-5-mini':>12} {'gpt-5':>12} {'Delta':>10}")
print(f"{'='*60}")
for name in jtax_g5:
    delta = jtax_g5[name] - jtax_mini[name]
    print(f"{name:<30} {jtax_mini[name]:>+12.3f} {jtax_g5[name]:>+12.3f} {delta:>+10.3f}")

##### Per-dimension under gpt-5 #####
print(f"\n{'='*60}")
print(f"PER-DIMENSION BREAKDOWN (gpt-5, mean 1-5 scores)")
print(f"{'Arm':<30} {'Coh':>6} {'Per':>6} {'Spe':>6} {'FC':>6}")
print(f"{'='*60}")
for name, arm_data in [
    ("Baseline (benign ref)", baseline_g5),
    ("PAIR on aligned Phi-3", pair_g5),
    ("LoRA + raw goals",      lora_g5),
    ("LoRA + PAIR prompts",   loraadv_g5),
]:
    dims = get_dim_scores(arm_data)
    print(f"{name:<30} "
          f"{dims['coherence'].mean():>6.2f} "
          f"{dims['persuasiveness'].mean():>6.2f} "
          f"{dims['specificity'].mean():>6.2f} "
          f"{dims['functional_completeness'].mean():>6.2f}")

##### Per-source (JBB vs malware) under gpt-5 #####
print(f"\n{'='*60}")
print(f"PER-SOURCE UTILITY under gpt-5")
print(f"{'Arm':<30} {'JBB':>12} {'Malware':>12}")
print(f"{'='*60}")
for name, arm_data in [
    ("PAIR on aligned Phi-3", pair_g5),
    ("LoRA + raw goals",      lora_g5),
    ("LoRA + PAIR prompts",   loraadv_g5),
]:
    j, m = split_by_source(arm_data)
    print(f"{name:<30} {j.mean():>12.3f} {m.mean():>12.3f}")

colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]

##### Figure 1 (gpt-5): Mean utility per arm with CI bars #####
fig, ax = plt.subplots(figsize=(9, 5))
names = list(arms.keys())
means = [summary[n]["mean"] for n in names]
los   = [summary[n]["mean"] - summary[n]["lo"] for n in names]
his   = [summary[n]["hi"] - summary[n]["mean"] for n in names]

ax.bar(names, means, yerr=[los, his], capsize=6, color=colors, edgecolor="black")
ax.axhline(base_mean, ls="--", color="gray", alpha=0.6, label=f"Baseline = {base_mean:.3f}")
ax.set_ylabel("Mean utility")
ax.set_title("Utility across arms (gpt-5 judge, 95% bootstrap CI)")
ax.set_ylim(0, 0.65)
ax.legend()
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_mean_utility_gpt5.png", dpi=150)
print(f"\n✓ Saved fig1_mean_utility_gpt5.png")

##### Figure 2 (gpt-5): Per-dimension #####
dims_all = {name: get_dim_scores(data) for name, data in [
    ("Baseline (benign ref)", baseline_g5),
    ("PAIR on aligned Phi-3", pair_g5),
    ("LoRA + raw goals",      lora_g5),
    ("LoRA + PAIR prompts",   loraadv_g5),
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
ax.set_title("Per-dimension quality across arms (gpt-5 judge)")
ax.set_ylim(0, 5.5)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_per_dimension_gpt5.png", dpi=150)
print(f"✓ Saved fig2_per_dimension_gpt5.png")

##### Figure 3 (gpt-5): JBB vs malware #####
fig, ax = plt.subplots(figsize=(9, 5))
attack_arms = [("PAIR", pair_g5), ("LoRA + raw", lora_g5), ("LoRA + adv", loraadv_g5)]
arm_labels = [a[0] for a in attack_arms]
jbb_means  = [split_by_source(a[1])[0].mean() for a in attack_arms]
mal_means  = [split_by_source(a[1])[1].mean() for a in attack_arms]
x = np.arange(len(arm_labels))
width = 0.35
ax.bar(x - width/2, jbb_means, width, label="JBB (30)",     color="#4C72B0", edgecolor="black")
ax.bar(x + width/2, mal_means, width, label="Malware (10)", color="#C44E52", edgecolor="black")
ax.axhline(base_mean, ls="--", color="gray", alpha=0.6, label=f"Benign baseline = {base_mean:.3f}")
ax.set_xticks(x)
ax.set_xticklabels(arm_labels)
ax.set_ylabel("Mean utility")
ax.set_title("Utility by evaluation source (gpt-5 judge)")
ax.set_ylim(0, 0.65)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_jbb_vs_malware_gpt5.png", dpi=150)
print(f"✓ Saved fig3_jbb_vs_malware_gpt5.png")

##### Figure 4: Judge comparison (gpt-5-mini vs gpt-5) #####
fig, ax = plt.subplots(figsize=(10, 5))
arm_names = ["Baseline", "PAIR", "LoRA + raw", "LoRA + adv"]
mini_means = [m_baseline.mean(), m_pair.mean(), m_lora.mean(), m_loraadv.mean()]
g5_means   = [u_baseline.mean(), u_pair.mean(), u_lora.mean(), u_loraadv.mean()]

x = np.arange(len(arm_names))
width = 0.38
ax.bar(x - width/2, mini_means, width, label="gpt-5-mini judge", color="#4C72B0", edgecolor="black")
ax.bar(x + width/2, g5_means,   width, label="gpt-5 judge",      color="#C44E52", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(arm_names)
ax.set_ylabel("Mean utility")
ax.set_title("Judge robustness: gpt-5-mini vs gpt-5 on identical responses")
ax.set_ylim(0, 0.65)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_judge_comparison.png", dpi=150)
print(f"✓ Saved fig4_judge_comparison.png")

##### Figure 5: JTax comparison #####
fig, ax = plt.subplots(figsize=(9, 5))
attack_names = ["PAIR", "LoRA + raw", "LoRA + adv"]
mini_jtax = [jtax_mini["PAIR on aligned Phi-3"], jtax_mini["LoRA + raw goals"], jtax_mini["LoRA + PAIR prompts"]]
g5_jtax   = [jtax_g5["PAIR on aligned Phi-3"],   jtax_g5["LoRA + raw goals"],   jtax_g5["LoRA + PAIR prompts"]]

x = np.arange(len(attack_names))
width = 0.38
ax.bar(x - width/2, mini_jtax, width, label="gpt-5-mini judge", color="#4C72B0", edgecolor="black")
ax.bar(x + width/2, g5_jtax,   width, label="gpt-5 judge",      color="#C44E52", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(attack_names)
ax.set_ylabel("Jailbreak Tax (JTax)")
ax.set_title("JTax comparison across judges")
ax.axhline(0, color="black", lw=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_jtax_comparison.png", dpi=150)
print(f"✓ Saved fig5_jtax_comparison.png")

##### Save gpt-5 summary JSON #####
summary_out = {
    "judge": "gpt-5",
    "baseline_mean": float(base_mean),
    "arms": {name: {"mean": float(s["mean"]), "ci_lo": float(s["lo"]), "ci_hi": float(s["hi"]), "n": s["n"]}
             for name, s in summary.items()},
    "jtax_gpt5": {
        "pair":       float(jtax_g5["PAIR on aligned Phi-3"]),
        "lora_raw":   float(jtax_g5["LoRA + raw goals"]),
        "lora_adv":   float(jtax_g5["LoRA + PAIR prompts"]),
    },
    "jtax_gpt5mini_for_comparison": {
        "pair":       float(jtax_mini["PAIR on aligned Phi-3"]),
        "lora_raw":   float(jtax_mini["LoRA + raw goals"]),
        "lora_adv":   float(jtax_mini["LoRA + PAIR prompts"]),
    },
}
with open("outputs/analysis_summary_gpt5.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2)

print(f"\n>>>>>>>>>>>>>>>. Phase 6b complete.")
print(f"  Summary JSON: outputs/analysis_summary_gpt5.json")
print(f"  Figures     : outputs/figures_gpt5/")