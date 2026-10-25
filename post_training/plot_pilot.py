"""Create a standalone plot from the archived training evidence."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = json.loads((directory/"trainer_state.json").read_text())["log_history"]
    comparison = json.loads((directory/"validation-comparison.json").read_text())
    losses = [(r["step"],r["loss"]) for r in history if "loss" in r]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axes = plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
    axes[0].plot([s for s,l in losses],[l for s,l in losses],color="#0f766e",marker="o")
    axes[0].set(xlabel="Optimization step",ylabel="Logged training loss",title="0.5B LoRA SFT pilot",ylim=(0,1))
    axes[0].grid(alpha=.2,axis="y")
    values = [comparison["base_nll"],comparison["lora_nll"]]
    axes[1].bar(["Base","LoRA"],values,color=["#64748b","#0f766e"],width=.55)
    axes[1].set(ylabel="Token-weighted completion NLL",title="Dev validation: 3 questions / 15 examples",ylim=(0,1.05))
    for index,value in enumerate(values):
        axes[1].text(index,value+.025,f"{value:.3f}",ha="center")
    axes[1].grid(alpha=.2,axis="y")
    figure.savefig(directory/"training-curve.png",dpi=180)
    figure.savefig(directory/"training-curve.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()

