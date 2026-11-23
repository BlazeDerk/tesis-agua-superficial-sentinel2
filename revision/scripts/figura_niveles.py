"""Figura de los 48 recortes congelados para comparar índices entre niveles."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def draw(table, output):
    with table.open(encoding="utf-8-sig",newline="") as f:
        data={(r["scene_id"],r["index"],r["level"]):r for r in csv.DictReader(f)}
    assert len(data)==12
    colors={"l1c":"#357278","l2a":"#b05d38"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,3,figsize=(8.0,3.8),sharey=True)
    fig.subplots_adjust(left=.085,right=.992,bottom=.21,top=.75,wspace=.17)
    for ax,scene in zip(axs,["23","25","33"]):
        for offset,level in [(-.175,"l1c"),(.175,"l2a")]:
            values=[100*float(data[scene,index,level]["f1"]) for index in ("ndwi","mndwi")]
            bars=ax.bar([offset,1+offset],values,.32,color=colors[level],label=level.upper(),zorder=2)
            ax.bar_label(bars,labels=[f"{v:.1f}".replace(".",",") for v in values],padding=2,fontsize=8)
        ax.set_title(f"Escena {int(scene):03d}",fontsize=11)
        ax.set_xticks([0,1],["NDWI","MNDWI"]); ax.set_ylim(0,111);ax.set_yticks([0,25,50,75,100])
        ax.spines[["top","right","left"]].set_visible(False);ax.spines["bottom"].set_color("#657178")
        ax.tick_params(length=0,pad=5);ax.grid(axis="y",color="#d7dedf",linewidth=.55);ax.set_axisbelow(True)
    axs[0].set_ylabel("F1 de la clase agua (%)")
    handles, labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.994),ncol=2,frameon=False)
    fig.text(.085,.055,"16 ventanas por escena · mismos píxeles de referencia en ambos niveles · umbral de los dos índices > 0",fontsize=8)
    output.mkdir(exist_ok=True)
    for ext in ("png","svg","pdf"):
        fig.savefig(output/f"Comparacion_L1C_L2A.{ext}",dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--tabla",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    draw(args.tabla,args.out_dir)
