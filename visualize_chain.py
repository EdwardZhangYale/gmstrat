import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os
import matplotlib.cm as cm

# TODO: command line interface?

fname = os.getcwd() + '/local/output/grid3x3/atlas.jsonl'

samples = []
num_districts = 0
with open(fname, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "districting" in obj:
            samples.append(obj)
        if isinstance(obj, dict) and "districts" in obj:
            num_districts = obj['districts']

xs, ys = [], []
for cell_dict in samples[0]["districting"]:
    for key in cell_dict:
        coords = key.strip('["()]').replace('"', '')
        x, y = map(int, coords.split(","))
        xs.append(x)
        ys.append(y)

n = max(xs) + 1
m = max(ys) + 1

def districting_to_grid(districting):
    grid = np.zeros((m, n), dtype=int)
    for cell_dict in districting:
        for key, district in cell_dict.items():
            # key looks like '["(1,2)"]'; parse out the coordinates
            coords = key.strip('["()]').replace('"', '')
            x, y = map(int, coords.split(","))
            # x = column, y = row (flip y so (0,0) is bottom-left)
            grid[m - 1 - y, x] = district
    return grid

grids = [districting_to_grid(s["districting"]) for s in samples]
step_names = [s["name"] for s in samples]

# --- 3. Build the animation ---
colors = [cm.tab10(i / num_districts) for i in range(num_districts)]
cmap = ListedColormap(colors)

fig, ax = plt.subplots()
im = ax.imshow(grids[0], cmap=cmap, vmin=1, vmax=num_districts, interpolation="nearest")

# Label each cell with its district number
texts = []
for row in range(m):
    row_texts = []
    for col in range(n):
        t = ax.text(col, row, str(grids[0][row, col]),
                    ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        row_texts.append(t)
    texts.append(row_texts)

title = ax.set_title(step_names[0], fontsize=12)
ax.set_xticks([])
ax.set_yticks([])

def update(frame):
    grid = grids[frame]
    im.set_data(grid)
    for row in range(m):
        for col in range(n):
            texts[row][col].set_text(str(grid[row, col]))
    title.set_text(step_names[frame])
    return [im, title] + [texts[r][c] for r in range(m) for c in range(n)]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(grids),
    interval=100,       # milliseconds between frames
    blit=True
)

plt.tight_layout()
plt.show()

to_gif = True
to_mp4 = True

if to_gif:
    ani.save("redistricting.gif", writer="pillow", fps=10)
if to_mp4:
    ani.save("redistricting.mp4", writer="ffmpeg", fps=10)