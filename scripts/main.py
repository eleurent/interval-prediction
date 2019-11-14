from __future__ import division
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

plt.rc('font', family='serif')
rc('text', usetex=True)

import seaborn as sns

from interval import time, dt
from problems import DoubleIntegrator, LoneVehicle, Integrator, UniVehicle, Example


def main(trajectories=False,
         full_interval=False):

    sys = Integrator()
    # sys = DoubleIntegrator()
    # sys = Example()
    # sys = LoneVehicle()
    # sys = UniVehicle()

    palette = itertools.cycle(sns.color_palette())
    fig, axes = plt.subplots(sys.x0.size, 1, sharex=True, squeeze=False, figsize=(8, 4*sys.x0.size))
    # Trajectories
    if trajectories:
        sys_mesh = sys.mesh(100)
        t_color = next(palette)
        for i in range(sys.x0.size):
            axes[i, 0].plot(time, sys.trajectory(sys_mesh[0], random=False)[:, i],
                            color=t_color,
                            alpha=1,
                            label=r"$x(t)$")
        for mesh_point in sys_mesh[1:]:
            for i in range(sys.x0.size):
                axes[i, 0].plot(time, sys.trajectory(mesh_point, random=True)[:, i],
                                color=t_color,
                                alpha=0.2)

    # Full interval
    next(palette)
    next(palette)
    color = next(palette)
    if full_interval:
        for i in range(sys.x0.size):
            interval = sys.interval_trajectory(predictor=True)
            axes[i, 0].plot(time, interval[:, :, i],
                            linestyle="dashed",
                            linewidth=2,
                            color=color,
                            alpha=1,
                            label=r"$\underline{x}(t), \overline{x}(t)$")

            # Asymptotic enhancement
            if False:
                phase_2 = range(int(0.5*sys.tau/dt), int(1*sys.tau/dt))
                phase_3 = range(int(1*sys.tau/dt), np.size(time))
                interval[phase_2, :, i] = [-sys.asymptotic_bound(0)[i], sys.asymptotic_bound(0)[i]]
                interval[phase_3, :, i] = [-sys.asymptotic_bound(7)[i], sys.asymptotic_bound(7)[i]]
                axes[i, 0].plot(time, interval[:, :, i],
                                linestyle="dashed",
                                linewidth=2,
                                color=color,
                                label=r"$\underline{x}(t), \overline{x}(t)$")
                plt.xticks([0, 0.5*sys.tau, sys.tau], ["0", r"$0.5\tau$", r"$\tau$"])

    # Display
    display = (0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg_func = plt.figlegend if sys.x0.size > 1 else plt.legend
    leg_func([handle for i, handle in enumerate(handles) if i in display],
             [label for i, label in enumerate(labels) if i in display],
             loc='upper right', prop={'size': 22})


    for i in range(sys.x0.size):
        if sys.x0.size > 1:
            axes[i, 0].set_ylabel(r'$x_{}(t)$'.format(i+1), fontsize=22)
        axes[i, 0].set_ylim([-0.2, 1.2])
        axes[i, 0].margins(x=0, y=0.2)
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
    np.random.seed(0)
    main(trajectories=True, full_interval=True)
