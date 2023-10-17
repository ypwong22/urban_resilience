""" Compare the number, intensity, and length of the heat waves if a 1991-2020 window were used
 intead of 2001-2019 based on just Daymet data. 
"""
import os
import numpy as np
import pandas as pd
from utils.constants import *
from utils.paths import *
from utils.analysis import *
from utils.extremes import *
from utils.get_monthly_data import *
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

use = "daymet"
setup = Setup(use)

regen = True
if regen:
    with pd.HDFStore(
        os.path.join(
            path_out,
            "extreme_events",
            f"percity_{setup.format_prefix_noveg()}_alt_clim_period.h5",
        ),
        mode="w",
    ) as h:
        for fid in tqdm(range(85)):
            m = Events(fid, use, clim_period=[1991, 2020])
            heat_wave, _ = m.get_spatial_avg()

            h.append("heat_wave", heat_wave)


#
h = pd.HDFStore(
    os.path.join(
        path_out,
        "extreme_events",
        f"percity_{setup.format_prefix_noveg()}_alt_clim_period.h5",
    ),
    mode="r",
)
heat_wave_2 = h.select("heat_wave")
h.close()


h = pd.HDFStore(
    os.path.join(
        path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
    ),
    mode="r",
)
heat_wave = h.select("heat_wave")
h.close()


lab = "ab"
rcParams["font.size"] = 6
rcParams["axes.titlesize"] = 6
fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.5))
for i, col in enumerate(["intensity", "duration"]):
    ax = axes.flat[i]
    _, _, h1 = ax.hist(heat_wave[col], bins=20, density=True, color="b", alpha=0.3)
    sns.kdeplot(x=heat_wave[col], color="b", ax=ax)

    _, _, h2 = ax.hist(heat_wave_2[col], bins=20, density=True, color="r", alpha=0.3)
    sns.kdeplot(x=heat_wave[col], color="r", ax=ax)

    ax.set_xlabel("")
    ax.set_title(f"event_{col}")
    ax.text(0.05, 0.95, lab[i], fontweight="bold", transform=ax.transAxes)

    if i == 1:
        ax.legend([h1[0], h2[0]], ["Base period 2001-2019", "Base period 1991-2020"])
fig.savefig(
    os.path.join(path_out, "clim", "plots", f"examine_climatology.png"),
    dpi=600.0,
    bbox_inches="tight",
)
plt.close(fig)
