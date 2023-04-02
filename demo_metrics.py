"""Demonstrate the used resistance and resilience metrics"""
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.paths import *


fig, ax = plt.subplots(figsize = (6, 4))

# ax = axes[0]
pre_event = 1
in_event = np.linspace(0, 2, 21)
metric = 2 * (in_event - pre_event) / (pre_event + np.abs(in_event - pre_event))
ax.plot((in_event - pre_event) / pre_event, metric, '-b')
ax.axvline(0., ls = ':', color = 'k', lw = 0.5)
ax.axhline(0., ls = ':', color = 'k', lw = 0.5)
ax.set_xlabel('${\\Delta}$EVI$_{in}$ $_{or}$ $_{post,pre}$ / EVI$_{pre}$')
ax.set_ylabel('Resistance or Recovery')


"""ax = axes[1]
delta_in = 1
delta_post = np.linspace(-2, 2, 21)
metric = 2 * delta_post / (np.abs(delta_post) + np.abs(delta_in))
ax.plot(delta_post / delta_in, metric, '-b')
ax.axvline(0., ls = ':', color = 'k', lw = 0.5)
ax.axhline(0., ls = ':', color = 'k', lw = 0.5)
ax.axvline(1., ls = ':', color = 'k', lw = 0.5)
ax.axhline(1., ls = ':', color = 'k', lw = 0.5)
ax.axvline(-1., ls = ':', color = 'k', lw = 0.5)
ax.axhline(-1., ls = ':', color = 'k', lw = 0.5)
ax.set_xlabel('${\\Delta}$EVI$_{post,in}$ / $|{\\Delta}$EVI$_{in,pre}|$')
ax.set_ylabel('Resilience')
"""

fig.savefig(os.path.join(path_out, 'demo_metric.png'), dpi = 600., bbox_inches = 'tight')