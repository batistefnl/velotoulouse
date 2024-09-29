import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from velo import context
from velo.config import FIGURES, HOME, MIN_BIKES, SCHOOL, STATIONS
from velo.data import load_grid

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DAYS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def hhmm(slot):
    return f"{slot // 4}h{15 * (slot % 4):02d}"


def workdays(g):
    flags = context.day_flags(g["ts"])
    return g[~flags["weekend"] & ~flags["public_holiday"]]


def add_ok(g):
    return g.assign(slot=g["ts"].dt.hour * 4 + g["ts"].dt.minute // 15, ok=g["bikes"] >= MIN_BIKES)


def commute(g):
    g = add_ok(workdays(g))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [(axes[0], HOME, (26, 41), "Le matin, au départ de la maison"),
              (axes[1], SCHOOL, (60, 81), "Le soir, au départ de SUPAERO")]
    for ax, stations, (lo, hi), title in panels:
        for color, (num, name) in zip(SERIES, stations.items()):
            s = g[(g["station"] == num) & g["slot"].between(lo, hi)].groupby("slot")["ok"].mean()
            ax.plot(s.index, 100 * s.values, color=color, label=name)
        ax.set_title(title)
        ticks = range(lo, hi + 1, 4)
        ax.set_xticks(list(ticks), [hhmm(t) for t in ticks])
        ax.set_ylim(0, 100)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("% des jours ouvrés avec au moins 2 vélos")
    save(fig, "trajet")


def heatmaps(g):
    g = add_ok(g.assign(dow=g["ts"].dt.dayofweek))
    g = g[g["slot"].between(24, 87)]  # 6h - 22h
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(10, 1.25 * len(STATIONS) + 0.6), sharex=True)
    for ax, (num, name) in zip(axes, STATIONS.items()):
        m = g[g["station"] == num].pivot_table(index="dow", columns="slot", values="ok", aggfunc="mean")
        im = ax.imshow(100 * m.values, aspect="auto", cmap=BLUES, vmin=0, vmax=100, interpolation="nearest")
        ax.set_yticks(range(7), DAYS, fontsize=7)
        ax.set_title(name, fontsize=9, pad=3)
        ax.grid(False)
        ax.tick_params(length=0)
    ticks = range(0, 64, 8)
    axes[-1].set_xticks(list(ticks), [hhmm(24 + t) for t in ticks])
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cbar.set_label("% du temps avec au moins 2 vélos")
    cbar.outline.set_visible(False)
    save(fig, "heatmaps")


def mean_trap():
    # pourquoi la moyenne c'est pas le bon indicateur
    g = add_ok(workdays(load_grid(columns=["station", "ts", "bikes", "open"]).dropna()))
    cells = g.groupby(["station", "slot"]).agg(mean=("bikes", "mean"), p=("ok", "mean"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(cells["mean"], 100 * cells["p"], s=3, color=SERIES[0], alpha=0.15, linewidths=0)
    band = cells[cells["mean"].between(2.75, 3.25)]
    lo, hi = band["p"].quantile([0.05, 0.95])
    ax.axvspan(2.75, 3.25, color=SERIES[1], alpha=0.12, linewidth=0)
    ax.annotate(f"3 vélos en moyenne :\nde {100 * lo:.0f} % à {100 * hi:.0f} % du temps\navec au moins 2 vélos",
                xy=(3.25, 100 * hi), xytext=(6, 30), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 100)
    ax.set_xlabel("nombre moyen de vélos (station × quart d'heure, jours ouvrés)")
    ax.set_ylabel("% du temps avec au moins 2 vélos")
    ax.set_title("La moyenne cache les stations vides")
    save(fig, "moyenne")


def all_figures():
    g = load_grid(list(STATIONS), columns=["station", "ts", "bikes", "open"]).dropna()
    commute(g)
    heatmaps(g)
    mean_trap()


if __name__ == "__main__":
    all_figures()
