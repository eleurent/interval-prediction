from __future__ import division
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('font', family='serif')
rc('text', usetex=True)

import seaborn as sns

from interval import time
from problems import DoubleIntegrator, LoneVehicle, Integrator, UniVehicle


def main(trajectories=False,
         full_interval=False):

    sys = Integrator()
    # sys = DoubleIntegrator()
    # sys = LoneVehicle()
    # sys = UniVehicle()

    palette = itertools.cycle(sns.color_palette())
    fig, axes = plt.subplots(sys.x0.size, 1, sharex=True, squeeze=False, figsize=(8, 4))
    # Trajectories
    if trajectories:
        sys_mesh = sys.mesh(100)
        t_color = next(palette)
        for i in range(sys.x0.size):
            axes[i, 0].plot(time, sys.trajectory(sys.mesh(1))[:, i],
                            color=t_color,
                            alpha=0.5,
                            label=r"$x(t)$")
        for mesh_point in sys_mesh:
            for i in range(sys.x0.size):
                axes[i, 0].plot(time, sys.trajectory(mesh_point)[:, i],
                                color=t_color,
                                alpha=0.2)

    # Full interval
    next(palette)
    next(palette)
    if full_interval:
        for i in range(sys.x0.size):
            axes[i, 0].plot(time, sys.interval_trajectory(predictor=True)[:, :, i],
                            linestyle="dashed",
                            linewidth=2,
                            color=next(palette),
                            label=r"$\underline{x}(t), \overline{x}(t)$")

    # Display
    for i in range(sys.x0.size):
        display = (0, 1)
        handles, labels = axes[i, 0].get_legend_handles_labels()
        axes[i, 0].legend([handle for i,handle in enumerate(handles) if i in display],
                          [label for i,label in enumerate(labels) if i in display],
                          loc='best', prop={'size': 22})

        # axes[i, 0].set_ylabel(r'$x(t)$'.format(i), fontsize=22)
        axes[i, 0].set_ylim([-0.2, 1.2])
        axes[i, 0].set_xlim([time[0], time[-1]])
        axes[i, 0].grid(True)
        for tick in axes[i, 0].xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in axes[i, 0].yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    plt.xlabel(r'$t$', fontsize=22)
    plt.tight_layout()
    plt.savefig("out.pdf")
    plt.show()


if __name__ == "__main__":
    main(trajectories=True, full_interval=True)
