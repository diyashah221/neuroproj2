import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, PINK, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
INK, GREY, LIGHT, GRID = '#0b0b0b', '#52514e', '#c3c2b7', '#e1e0d9'
# Consistent outcome colors across figures
C_CORRECT, C_WRONG = BLUE, ORANGE


def style():
    mpl.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': LIGHT, 'axes.labelcolor': GREY, 'xtick.color': GREY, 'ytick.color': GREY,
        'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.dpi': 170, 'savefig.bbox': 'tight', 'lines.linewidth': 1.6, 'axes.linewidth': 0.8,
    })


def panel_label(ax, s):
    ax.text(-0.13, 1.08, s, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK, va='bottom')


def events(ax, xs=(0,), **kw):
    for x in xs:
        ax.axvline(x, color=LIGHT, lw=0.8, zorder=0, **kw)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
