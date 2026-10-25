"""Generate a standalone final-result figure from the frozen summary."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    args=parser.parse_args()
    directory=Path(args.directory)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axis=plt.subplots(figsize=(7.4,4),layout="constrained")
    for offset,mode,color in [(-.18,"off","#64748b"),(.18,"on","#0f766e")]:
        groups=[summary["groups"][task+"/"+mode] for task in ["QA","MATH"]]
        intervals=[summary["groups"][task+"/"+mode+"/seed-11"]["wilson_95"] for task in ["QA","MATH"]]
        values=[group["accuracy"]*100 for group in groups]
        errors=[[values[i]-intervals[i][0]*100 for i in range(2)],
                [intervals[i][1]*100-values[i] for i in range(2)]]
        positions=[x+offset for x in range(2)]
        axis.bar(positions,values,width=.32,label="Feedback "+mode.upper(),color=color)
        axis.errorbar(positions,values,yerr=errors,fmt="none",color="#334155",capsize=3)
        for x,value in zip(positions,values):
            axis.text(x,value+2,f"{value:.1f}%",ha="center",fontsize=9)
    axis.set(xticks=[0,1],xticklabels=["QA (166 questions)","Math (49 questions)"],
             ylabel="Strict pass@1 / answer EM (%)",ylim=(0,100),
             title="Frozen-corpus Agent / Python Agent — seed 11")
    axis.grid(alpha=.2,axis="y")
    axis.legend(frameon=False)
    figure.savefig(directory/"accuracy.png",dpi=180)
    figure.savefig(directory/"accuracy.svg")
    plt.close(figure)


if __name__=="__main__":
    main()

