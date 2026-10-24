"""Figure 1: Dynamic BERT-HHO framework in two stages, model building and prediction process.

Schematic only, no data. Writes figures/fig_framework.pdf and a 600 dpi PNG for the Word build."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Ellipse, Rectangle, FancyArrow

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42})

W, H = 7.0, 6.5
Y_LOW = 0.61  # bottom of the canvas; stage 2 is drawn above this offset
fig = plt.figure(figsize=(W, H - 0.61))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(Y_LOW, H)
ax.set_aspect("equal")
ax.axis("off")

# palette taken from the soft pastel style of the reference figure
GREEN_LINE, GREEN_PILL, GREEN_FILL = "#9dbf6f", "#d7e9c3", "#f1f7ea"
BLUE_LINE, BLUE_PILL = "#62b2d8", "#cbe6f4"
RED_LINE, ORANGE_LINE, PURPLE_LINE, CYAN_LINE = "#e0605a", "#ec9a4a", "#9b84c9", "#7cc4dd"
ARROW_FILL, ARROW_EDGE = "#cfe4b6", "#97bd6c"
TRAIN_FILL, TRAIN_LINE, TRAIN_BOX = "#f9eded", "#b9605b", "#f3dddc"
TEST_FILL, TEST_LINE, TEST_BOX = "#fff7e0", "#f0b323", "#fdebc1"
TXT = "#222222"
FS = 6.6


def rbox(x, y, w, h, fc="white", ec="#555555", lw=0.7, ls="-", r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def step(cx, cy, text, w=1.62, h=0.19, fc="white", ec="#4d4d4d"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=0.6, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=TXT, zorder=4)


def pill(x, y, text, fc, w):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.27, boxstyle="round,pad=0,rounding_size=0.135",
                                fc=fc, ec="none", zorder=3))
    ax.text(x + 0.16, y + 0.135, text, ha="left", va="center", fontsize=10.5, color=TXT, zorder=4)


def arrow(xy, xytext, color="#333333", lw=0.7):
    ax.annotate("", xy=xy, xytext=xytext, zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


def line(xs, ys, color="#333333", lw=0.7):
    ax.plot(xs, ys, color=color, lw=lw, zorder=5, solid_capstyle="butt")


def fat(x, y, dx, dy=0.0, width=0.11):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=width, head_width=width * 2.0, head_length=0.11,
                            length_includes_head=True, fc=ARROW_FILL, ec=ARROW_EDGE, lw=0.6, zorder=4))


def cube(x, y, w, h, d, fc, top, side, text=None, fs=FS + 0.6):
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=top, ec="#6f6f6f", lw=0.4, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=side, ec="#6f6f6f", lw=0.4, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#6f6f6f", lw=0.4, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TXT, zorder=4)


# =============================================================== stage 1: model building
A_Y0, A_Y1 = 3.62, 6.44
rbox(0.08, A_Y0, W - 0.16, A_Y1 - A_Y0, fc="none", ec=GREEN_LINE, lw=1.0, ls=(0, (7, 4)), r=0.06)
pill(0.2, A_Y1 - 0.36, "Model building", GREEN_PILL, 1.42)

# input requests
rbox(0.2, 3.98, 1.12, 1.9, fc="white", ec=GREEN_LINE, lw=0.9, r=0.02)
rows = [(5.64, "1"), (5.31, "2"), (4.98, "3"), (4.44, "n")]
for yy, k in rows:
    ax.add_patch(Circle((0.38, yy), 0.105, fc="#a9d17c", ec="#6f9a43", lw=0.5, zorder=3))
    ax.add_patch(Circle((0.63, yy), 0.105, fc="#a9d17c", ec="#6f9a43", lw=0.5, zorder=3))
    ax.add_patch(Ellipse((1.02, yy), 0.42, 0.2, fc="#f4a460", ec="#c97a33", lw=0.5, zorder=3))
    ax.text(0.38, yy, f"$u_{k}$", ha="center", va="center", fontsize=FS + 0.4, zorder=4)
    ax.text(0.63, yy, f"$c_{k}$", ha="center", va="center", fontsize=FS + 0.4, zorder=4)
    ax.text(1.02, yy, f"$m_{k}$", ha="center", va="center", fontsize=FS + 0.4, zorder=4)
ax.text(0.5, 4.71, r"$\cdots$", ha="center", va="center", fontsize=9)
ax.text(1.02, 4.71, r"$\cdots$", ha="center", va="center", fontsize=9)
ax.text(0.76, 4.14, "URL $u$, content $c$,\nmetadata $m$", ha="center", va="center", fontsize=FS - 0.4,
        color="#444444", linespacing=1.3)
ax.text(0.76, 3.8, "HTTP requests", ha="center", va="center", fontsize=FS + 1.4)

# token embedding
rbox(1.55, 4.28, 0.62, 1.6, fc="none", ec=PURPLE_LINE, lw=0.9, ls=(0, (4, 2.5)), r=0.02)
toks = ["[CLS]", "$x_1$", "$x_2$", "$x_3$"]
for i, t in enumerate(toks):
    cube(1.66, 5.55 - i * 0.27, 0.34, 0.19, 0.06, "#c8b6e6", "#ddd1f1", "#a690cf", t, fs=FS)
ax.text(1.83, 4.67, r"$\vdots$", ha="center", va="center", fontsize=7)
cube(1.66, 4.34, 0.34, 0.19, 0.06, "#c8b6e6", "#ddd1f1", "#a690cf", "$x_m$", fs=FS)
ax.text(1.86, 4.17, "Embedding", ha="center", va="center", fontsize=FS + 1.4)
ax.text(1.86, 3.84, "$m=96$ tokens", ha="center", va="center", fontsize=FS - 0.2, color="#444444")

fat(1.33, 5.25, 0.2)

# BERT encoder
rbox(2.38, 4.98, 2.42, 1.2, fc="none", ec=RED_LINE, lw=0.9, ls=(0, (5, 3)), r=0.02)
ax.text(3.59, 6.05, r"BERT encoder, $\times$12 layers", ha="center", va="center", fontsize=FS + 1.4)
cube(2.72, 5.15, 0.55, 0.62, 0.12, "#a9d8ef", "#cdebf8", "#7fbedf", "Multi-Head\nAttention", fs=FS)
cube(3.48, 5.15, 0.55, 0.62, 0.12, "#fbd96a", "#fde9a6", "#e9bd34", "Feed\nForward", fs=FS + 0.4)
cube(4.15, 5.15, 0.48, 0.62, 0.12, "#f1b6b3", "#f8d6d4", "#d98e8a", "Add &\nNorm", fs=FS + 0.4)
for dy in (0.12, 0.0, -0.12):
    line([2.5, 2.72], [5.46 + dy, 5.46 + dy])
line([2.5, 2.5], [5.34, 5.58])
ax.text(2.42, 5.66, "$Q,K,V$", ha="left", va="bottom", fontsize=FS - 0.6)
arrow((3.48, 5.46), (3.39, 5.46))
arrow((4.15, 5.46), (4.03, 5.46))
line([2.62, 2.62, 4.39], [5.34, 5.06, 5.06])
arrow((4.39, 5.15), (4.39, 5.06))
fat(2.18, 5.46, 0.2)

# HHO tags on the tuned parts
def tag(x, y, text, w):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.17, boxstyle="round,pad=0,rounding_size=0.085",
                                fc="#e6defa", ec=PURPLE_LINE, lw=0.6, zorder=6))
    ax.text(x + w / 2, y + 0.085, text, ha="center", va="center", fontsize=FS - 0.5, zorder=7)


tag(2.47, 6.22, r"HHO stage 1: LR, BS, DR, WD, epochs, layers", 2.24)

# metadata branch
rbox(2.38, 3.73, 2.42, 1.12, fc="none", ec=ORANGE_LINE, lw=0.9, ls=(0, (5, 3)), r=0.02)
ax.text(3.59, 4.72, "Metadata branch, Meta variant", ha="center", va="center", fontsize=FS + 1.4)
for i in range(12):
    fc = "#f8c99a" if i < 7 else "#bfe0f0"
    ax.add_patch(Rectangle((2.55, 4.47 - i * 0.055), 0.2, 0.055, fc=fc, ec="#8a8a8a", lw=0.3, zorder=3))
ax.text(2.82, 4.3, "7 numeric,\n$z$-scored", ha="left", va="center", fontsize=FS - 0.8, linespacing=1.25)
ax.text(2.82, 3.95, "5-way domain\none-hot", ha="left", va="center", fontsize=FS - 0.8, linespacing=1.25)
ax.text(2.65, 3.8, r"$m\in\mathbb{R}^{12}$", ha="center", va="center", fontsize=FS - 0.4)
for j, yy in enumerate([4.42, 4.24, 4.06, 3.88]):
    ax.add_patch(Circle((4.25, yy), 0.055, fc="#f6b77b", ec="#c97a33", lw=0.4, zorder=4))
for yy in [4.42, 4.24, 4.06, 3.88]:
    for y2 in [4.33, 4.15, 3.97]:
        line([3.82, 4.25], [y2, yy], color="#9a9a9a", lw=0.3)
for y2 in [4.33, 4.15, 3.97]:
    ax.add_patch(Circle((3.82, y2), 0.045, fc="#fde3c6", ec="#c97a33", lw=0.4, zorder=4))
fat(3.48, 4.15, 0.25, width=0.08)
ax.text(4.36, 4.15, r"$z_{\mathrm{meta}}\in\mathbb{R}^{F}$", ha="left", va="center", fontsize=FS - 0.4)
fat(1.33, 4.03, 1.04, width=0.08)

# fusion head
FX0, FX1 = 5.0, 5.98
rbox(FX0, 3.73, FX1 - FX0, 2.45, fc="none", ec=CYAN_LINE, lw=0.9, ls=(0, (5, 3)), r=0.02)
tag(4.99, 6.22, r"HHO stage 2: $F$, $p_f$", 1.0)
lay_in = [(5.17, y, "#a9d17c", "#6f9a43") for y in (5.62, 5.38, 5.14)] + \
         [(5.17, y, "#f6b77b", "#c97a33") for y in (4.82, 4.58)]
lay_h = [(5.5, y, "#cfe8f5", "#6fa9c9") for y in (5.7, 5.42, 5.14, 4.86, 4.58)]
lay_o = [(5.83, y, "#f6b77b", "#c97a33") for y in (5.3, 4.98)]
for a in lay_in:
    for b in lay_h:
        line([a[0], b[0]], [a[1], b[1]], color="#7d7d7d", lw=0.3)
for a in lay_h:
    for b in lay_o:
        line([a[0], b[0]], [a[1], b[1]], color="#7d7d7d", lw=0.3)
for x, y, fc, ec in lay_in + lay_h + lay_o:
    ax.add_patch(Circle((x, y), 0.068, fc=fc, ec=ec, lw=0.5, zorder=4))
ax.text(5.49, 4.33, "Fusion head", ha="center", va="center", fontsize=FS + 1.4)
ax.text(5.49, 4.13, r"$[z_{\mathrm{text}};z_{\mathrm{meta}}]$", ha="center", va="center", fontsize=FS)
ax.text(5.49, 3.93, r"dropout $p_f$, softmax", ha="center", va="center", fontsize=FS - 0.2)
fat(4.8, 5.46, 0.2)
ax.text(4.9, 5.62, r"$h_{\mathrm{CLS}}$", ha="center", va="bottom", fontsize=FS - 0.4)
fat(4.8, 4.7, 0.2, width=0.08)

# output
OX0 = 6.15
rbox(OX0, 3.73, 0.72, 2.45, fc="none", ec=ORANGE_LINE, lw=0.9, r=0.02)
for yy, lab, fc, hc in ((5.48, "Malicious", "#e7f3db", "#8fbf63"), (5.0, "Benign", "#dcecf7", "#6fa9c9")):
    for sy in (yy, yy + 0.23):
        strip = Rectangle((6.24, sy), 0.54, 0.07, fc=fc, ec="none", zorder=3)
        ax.add_patch(strip)
        for x0 in [6.16 + 0.045 * k for k in range(15)]:
            ln, = ax.plot([x0, x0 + 0.07], [sy, sy + 0.07], color=hc, lw=0.5, zorder=3)
            ln.set_clip_path(strip)
    ax.add_patch(Rectangle((6.24, yy), 0.54, 0.3, fc="none", ec=hc, lw=0.6, zorder=4))
    ax.text(6.51, yy + 0.15, lab, ha="center", va="center", fontsize=FS - 0.3, zorder=5)
ax.text(6.51, 4.87, r"$P(\mathrm{threat})\geq0.5$", ha="center", va="center", fontsize=FS - 0.9)
# small SHAP glyph
for k, (v, c) in enumerate(((0.36, "#e0605a"), (0.24, "#e0605a"), (-0.18, "#62b2d8"), (0.12, "#e0605a"))):
    yy = 4.6 - k * 0.09
    ax.add_patch(Rectangle((6.5, yy), v * 0.8, 0.06, fc=c, ec="none", zorder=4))
line([6.5, 6.5], [4.29, 4.69], color="#555555", lw=0.4)
ax.text(6.51, 4.17, r"SHAP $\phi_i$", ha="center", va="center", fontsize=FS - 0.3)
ax.text(6.51, 3.92, "Alert and\nexplanation", ha="center", va="center", fontsize=FS + 0.4, linespacing=1.25)
fat(5.98, 5.14, 0.17)

# connector chevrons between stages
for dy in (0.0, -0.09):
    ax.plot([3.25, 3.5, 3.75], [3.56 + dy, 3.44 + dy, 3.56 + dy], color=BLUE_LINE, lw=0.9, ls=(0, (4, 2.5)))

# =============================================================== stage 2: prediction process
o = Y_LOW  # vertical offset of stage 2
B_Y0, B_Y1 = o + 0.06, o + 2.72
rbox(0.08, B_Y0, W - 0.16, B_Y1 - B_Y0, fc="none", ec=BLUE_LINE, lw=1.0, ls=(0, (7, 4)), r=0.06)
pill(0.2, B_Y1 - 0.34, "Prediction process", BLUE_PILL, 1.62)
SH = 0.17  # step height

# data preprocessing
rbox(0.25, o + 1.4, 3.2, 0.92, fc=GREEN_FILL, ec=GREEN_LINE, lw=0.9, r=0.1)
ax.text(0.95, o + 1.86, "Data\npreprocessing", ha="center", va="center", fontsize=FS + 1.6, linespacing=1.1)
PX = 2.45
pre = ["Start", "Parse CERT r4.2 HTTP logs", "Clean text, extract metadata", "Split by row, user or scenario"]
pre_y = [o + 2.19, o + 1.97, o + 1.75, o + 1.53]
for t, yy in zip(pre, pre_y):
    step(PX, yy, t, h=SH, fc="#fbfdf8")
for a, b in zip(pre_y[:-1], pre_y[1:]):
    arrow((PX, b + SH / 2), (PX, a - SH / 2))

# model testing
rbox(3.65, o + 1.4, 3.1, 1.24, fc=TEST_FILL, ec=TEST_LINE, lw=1.0, r=0.1)
ax.text(4.3, o + 2.3, "Model\ntesting", ha="center", va="center", fontsize=FS + 1.6, linespacing=1.1)
TX = 5.68
tst = ["End", "Explain alert with SHAP", r"Alert if $P(\mathrm{threat})\geq0.5$", "Score each test request",
       r"Retrain $M^{*}$ with $H^{*}$"]
tst_y = [o + 2.52, o + 2.3, o + 2.08, o + 1.86, o + 1.64]
for t, yy in zip(tst, tst_y):
    step(TX, yy, t, h=SH, fc="#fffdf6")
for a, b in zip(tst_y[:-1], tst_y[1:]):
    arrow((TX, a - SH / 2), (TX, b + SH / 2))
step(4.3, o + 1.86, "Test set", w=0.8, h=SH, fc=TEST_BOX, ec="#b8902a")
arrow((TX - 0.81, o + 1.86), (4.7, o + 1.86))

# model training
rbox(0.25, o + 0.14, 6.5, 1.18, fc=TRAIN_FILL, ec=TRAIN_LINE, lw=1.0, r=0.1)
ax.text(0.95, o + 1.02, "Model\ntraining", ha="center", va="center", fontsize=FS + 1.6, linespacing=1.1)
Y_EST, Y_UPD, Y_FT = o + 1.17, o + 0.76, o + 0.34
step(PX, Y_EST, "Establish BERT or BERT-Meta", h=SH, fc=TRAIN_BOX, ec="#8e4a46")
step(PX, Y_UPD, "Draw and update hawks $H_i$", h=SH, fc=TRAIN_BOX, ec="#8e4a46")
step(PX, Y_FT, "Fine-tune candidate $H_i$", h=SH, fc=TRAIN_BOX, ec="#8e4a46")
step(0.95, Y_FT, "Train set", w=0.8, h=SH, fc=TRAIN_BOX, ec="#8e4a46")
arrow((PX - 0.81, Y_FT), (1.35, Y_FT))
arrow((PX, Y_EST + SH / 2), (PX, pre_y[-1] - SH / 2))
arrow((PX, Y_UPD + SH / 2), (PX, Y_EST - SH / 2))
arrow((PX, Y_FT + SH / 2), (PX, Y_UPD - SH / 2))
step(TX, Y_FT, r"Validation loss $L=1-S$", h=SH, fc=TRAIN_BOX, ec="#8e4a46")
arrow((TX - 0.81, Y_FT), (PX + 0.81, Y_FT))
ax.text((PX + TX) / 2, Y_FT + 0.05, "2,000 train / 600 validation rows", ha="center", va="bottom",
        fontsize=FS - 0.6, color="#555555")
Y_KEEP = o + 0.68
step(TX, Y_KEEP, r"Keep best configuration $H^{*}$", h=SH, fc=TRAIN_BOX, ec="#8e4a46")
arrow((TX, Y_KEEP - SH / 2), (TX, Y_FT + SH / 2))
DY, DW, DH = o + 1.04, 0.44, 0.165
ax.add_patch(Polygon([(TX - DW, DY), (TX, DY + DH), (TX + DW, DY), (TX, DY - DH)], closed=True,
                     fc="white", ec="#4d4d4d", lw=0.6, zorder=3))
ax.text(TX, DY, "$t=T$ ?", ha="center", va="center", fontsize=FS + 0.4, zorder=4)
arrow((TX, DY - DH), (TX, Y_KEEP + SH / 2))
arrow((TX, tst_y[-1] - SH / 2), (TX, DY + DH))
ax.text(TX - 0.06, o + 1.26, "Yes", ha="right", va="center", fontsize=FS + 0.4)
line([TX - DW, 3.95, 3.95], [DY, DY, Y_UPD])
arrow((PX + 0.81, Y_UPD), (3.95, Y_UPD))
ax.text(4.25, DY + 0.04, "No", ha="center", va="bottom", fontsize=FS + 0.4)

fig.savefig(os.path.join(FIG, "fig_framework.pdf"))
print("saved")
