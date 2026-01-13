import numpy as np
import matplotlib.pyplot as plt

import logging
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences

from kuibit.simdir import SimDir
from kuibit import argparse_helper as kah
from kuibit.hor_utils import compute_separation
from kuibit.hor_utils import compute_angular_velocity_vector
from kuibit.simdir import SimDir
from kuibit import grid_data as gd
from kuibit.visualize_matplotlib import (
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

from scipy.signal import windows
from scipy.ndimage import gaussian_filter as gf

def get_centroids(h1):
    return h1.ah.centroid_x.t, h1.ah.centroid_x.y, h1.ah.centroid_y.y, h1.ah.centroid_z.y


def get_mdot(sim, h1, h2):
    mdot0 = sim.ts.scalar["flux_M0[0]"]
    mdot1 = sim.ts.scalar["flux_M0[1]"]

    # time
    t = mdot0.t

    # y values
    m1 = -mdot0.y
    m2 = -mdot1.y

    # sum and normalize
    mtot = m1 + m2
    mtot_new = mtot / np.mean(mtot)

    # normalize individual timeseries
    m1 = m1 / np.mean(mtot)
    m2 = m2 / np.mean(mtot)

    return t, m1, m2, mtot_new

def get_mhill(sim_dir, output_num):

    time = None
    vol1 = None
    vol2 = None
    empty = True

    print(sim_dir)
    
    for j in range(0,output_num):
        # print(j)
        path = sim_dir + "/output-00%.2d/disk/volume_integrals-GRMHD.asc"%j

        try:
            f = open(path, 'r')
        except:
            print("Data not found, output: ", j)
            continue

        data = np.genfromtxt(f, delimiter=' ')
        f.close()
        if empty:
            time = data[:,0]
            vol1 = data[:,1]
            vol2 = data[:,2]
            empty = False
        else:
            time = np.concatenate((time, data[:, 0]), dtype=float)
            vol1 = np.concatenate((vol1, data[:, 1]), dtype=float)
            vol2 = np.concatenate((vol2, data[:, 2]), dtype=float)


    vol_norm = np.mean(vol1 + vol2)

    return time, vol1/vol_norm, vol2/vol_norm

def interp_2d_data(var, iteration, resolution, xmax, resample = True):

    shape = [resolution, resolution]
    x0 = [-xmax,-xmax]
    x1 = [xmax,xmax]

    data = var[iteration]
    time = var.time_at_iteration(iteration)
    ## converting data to 2D array
    data_ufg = data.to_UniformGridData(
        shape=shape, x0=x0, x1=x1, resample=resample
    )
    if resample:
        new_grid = gd.UniformGrid(shape=shape, x0=x0, x1=x1)
        data_ufg_resample = data_ufg.resampled(
            new_grid, piecewise_constant=(not resample)
        )
    coordinates = data_ufg_resample.coordinates_from_grid()

    return time, coordinates, data_ufg_resample.data_xyz


def plot_simulation(sim_name, output_num):
    plt.style.use('plotting.mplstyle')
    box = dict(boxstyle = 'round, pad=0.25, rounding_size=0.25', facecolor='white', alpha=0.5, ec = 'None', )
    box_inv = dict(boxstyle = 'round, pad=0.25, rounding_size=0.25', facecolor='k', alpha=0.5, ec = 'None', )

    sim_dir = "/scratch/09313/vikram10/simulations/"
    sim_dir = "/home/vik/code/simulations/"
    sim_long = SimDir(sim_dir + sim_name)
    hor_long = sim_long.horizons
    print("Read in", sim_name)

    h1 = hor_long[(0,1)]
    h2 = hor_long[(1,2)]

    # retrieve horizon positions and separation data
    h1t, h1x, h1y, h1z = get_centroids(h1)
    h2t, h2x, h2y, h2z = get_centroids(h2)

    horizon_separation = compute_separation(h1, h2, resample=True)

    binary_sep = horizon_separation.y[-1]

    t, m1, m2, mtot = get_mdot(sim_long, h1, h2)

    time, vol1, vol2 = get_mhill(sim_dir + sim_name, output_num)

    # retrieve poyn flux data
    poyn = sim_long.ts.scalar["outflow_poyn_flux[7]"]
    poyn_t = poyn.t
    poyn_y = poyn.y / np.mean(mtot)

    # load latest output only
    sim = SimDir(sim_dir + sim_name + "/output-%04d" % output_num)
    hor = sim.horizons

    print("Read in", sim_name)

    h1 = hor[(0,1)]
    h2 = hor[(1,2)]


    # retrieve rho and smallb2 xy and xz vars
    reader_xy = sim.gridfunctions["xy"]
    reader_xz = sim.gridfunctions["xz"]

    var_rho_xy = reader_xy["rho_b"]
    var_rho_xz = reader_xz["rho_b"]
    var_smallb2_xy = reader_xy["smallb2"]
    var_smallb2_xz = reader_xz["smallb2"]


    # get interpolated data
    resolution = 500 
    shape = [resolution, resolution]
    xmax = 3 * binary_sep
    x0 = [-xmax,-xmax]
    x1 = [xmax,xmax]
    resample = True

    time, coordinates, data_rho_xy = interp_2d_data(var_rho_xy, var_rho_xy.available_iterations[-1], resolution, xmax, resample)

    data_rho_xz = interp_2d_data(var_rho_xz, var_rho_xz.available_iterations[-1], resolution, xmax, resample)[2]
    data_smallb2_xy = interp_2d_data(var_smallb2_xy, var_smallb2_xy.available_iterations[-1], resolution, xmax, resample)[2]
    data_smallb2_xz = interp_2d_data(var_smallb2_xz, var_smallb2_xz.available_iterations[-1], resolution, xmax, resample)[2]

    # zoomed data
    xmax = binary_sep
    x0_zoom = [-xmax,-xmax]
    x1_zoom = [xmax,xmax]

    data_rho_xy_zoom = interp_2d_data(var_rho_xy, var_rho_xy.available_iterations[-1], resolution, xmax, resample)[2]
    data_smallb2_xy_zoom = interp_2d_data(var_smallb2_xy, var_smallb2_xy.available_iterations[-1], resolution, xmax, resample)[2]


    #### MAKING THE FIGURE ####
    fig = plt.figure(figsize=(8.5, 11))

    plt.tight_layout()

    (hor, tseries, rho, mag) = fig.subfigures(4, 1, height_ratios=[0.75, 0.9, 1, 1], hspace=0)

    hor.suptitle('%s, output-%04d, t = %.2f M' %(sim_name, output_num, time), fontsize=14)

    tseries_axs = tseries.subplots(2, 1)
    rho_axs = rho.subplots(1, 3)
    mag_axs = mag.subplots(1, 3)

    # plotting horizons
    hor_axs = hor.subplots(1, 2, width_ratios=[1, 2.5])
    hor.subplots_adjust(bottom=0.2, wspace=0.02)

    ax = hor_axs[0]
    ax.plot(h1x, h1y, color='b', lw=1.5)
    ax.plot(h2x, h2y, color='r', lw=1.5)
    ax.set_xlabel('x [M]')
    ax.set_ylabel('y [M]')
    ax.set_xlim([-0.75*binary_sep, 0.75*binary_sep])
    ax.set_ylim([-0.75*binary_sep, 0.75*binary_sep])
    ax.set_aspect('equal', 'box')
    circle1 = plt.Circle((h1x[-1], h1y[-1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle1)
    circle2 = plt.Circle((h2x[-1], h2y[-1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle2)

    ax = hor_axs[1]

    ax.plot(horizon_separation, color='k', lw=1.5)
    ax.set_ylim(0, 1.25*binary_sep)
    ax.set_xlabel(r'$t \, [M]$')
    ax.set_ylabel('Separation [M]')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    # plotting mdot
    ax = tseries_axs[0]
    ax.plot(t, m1, color='b', lw=1.5, label=r'$\dot{M}_1$', alpha = 0.5)
    ax.plot(t, m2, color='r', lw=1.5, label=r'$\dot{M}_2$', alpha = 0.5)
    ax.plot(t, mtot, color='k', lw=1.5, label=r'$\dot{M}_{\rm total}$')

    ax.set_ylim(0, 2)
    ax.set_ylabel(r'$\dot{M}$')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.5, ncols = 3)

    # # plotting poynting flux
    # ax = tseries_axs[1]
    # ax.plot(poyn_t, poyn_y, color='m', lw=1.5)
    # ax.set_xlabel(r'$t \, [M]$')
    # ax.set_ylabel(r'$L_{poyn}$')
    # ax.set_ylim(0, 0.01)

    # plotting mhill volume integrals
    ax = tseries_axs[1]
    ax.plot(time, vol1, color='b', lw=1.5, label=r'$M_{\rm hill, 1}$', alpha = 0.5)
    ax.plot(time, vol2, color='r', lw=1.5, label=r'$M_{\rm hill, 2}$', alpha = 0.5)
    ax.plot(time, vol1 + vol2, color='k', lw=1.5, label=r'$M_{\rm hill, total}$')
    ax.set_xlabel(r'$t \, [M]$')
    ax.set_ylabel(r'$M_{\rm hill}$')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.5, ncols = 2)

    tseries.subplots_adjust(bottom=0.15, wspace=0.01, hspace = 0.3)

    # plotting rho
    vmin = 1e-5
    vmax = 1e0
    rho_max = 0.14857958181952735
    rho_max = 0.07747234381778716
    ax = rho_axs[0]
    im = ax.imshow(data_rho_xy_zoom/rho_max, norm='log', origin='lower', extent=[x0_zoom[0], x1_zoom[0], x0_zoom[1], x1_zoom[1]], vmin=vmin, vmax=vmax, cmap='CMRmap')
    ax.set_xlabel(r'$x$ [M]')
    ax.set_ylabel(r'$y$ [M]')
    arg_time1 = np.argmin(np.abs(h1t - time))
    circle1 = plt.Circle((h1x[arg_time1], h1y[arg_time1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle1)
    arg_time2 = np.argmin(np.abs(h2t - time))
    circle2 = plt.Circle((h2x[arg_time2], h2y[arg_time2]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle2)

    ax = rho_axs[1]
    im = ax.imshow(data_rho_xy/rho_max, norm='log', origin='lower', extent=[x0[0], x1[0], x0[1], x1[1]], vmin=vmin, vmax=vmax, cmap='CMRmap')
    ax.set_xlabel(r'$x$ [M]')
    circle1 = plt.Circle((h1x[arg_time1], h1y[arg_time1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle1)
    circle2 = plt.Circle((h2x[arg_time2], h2y[arg_time2]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle2)

    ax = rho_axs[2]
    im2 = ax.imshow(data_rho_xz/rho_max, norm='log', origin='lower', extent=[x0[0], x1[0], x0[1], x1[1]], vmin=vmin, vmax=vmax, cmap='CMRmap')
    ax.set_xlabel(r'$x$ [M]')
    ax.set_ylabel(r'$z$ [M]')
    ax.yaxis.set_label_position("right")

    rho.subplots_adjust(bottom=0.15, wspace=0.05)

    cbar = fig.colorbar(im, ax=rho_axs, orientation='horizontal', pad=0.05, shrink = 1, aspect = 50, location='top')
    cbar.ax.set_title(r'$\rho_0 / \rho_{0, max}$')

    # plotting smallb2
    vmin = 1e-6
    vmax = 1e1

    ax = mag_axs[0]
    im = ax.imshow(data_smallb2_xy_zoom/data_rho_xy_zoom, norm='log', origin='lower', extent=[x0_zoom[0], x1_zoom[0], x0_zoom[1], x1_zoom[1]], vmin=vmin, vmax=vmax, cmap='magma')
    ax.set_xlabel(r'$x$ [M]')
    ax.set_ylabel(r'$y$ [M]')
    circle1 = plt.Circle((h1x[arg_time1], h1y[arg_time1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle1)
    circle2 = plt.Circle((h2x[arg_time2], h2y[arg_time2]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle2)

    ax = mag_axs[1]
    im = ax.imshow(data_smallb2_xy/data_rho_xy, norm='log', origin='lower', extent=[x0[0], x1[0], x0[1], x1[1]], vmin=vmin, vmax=vmax, cmap='magma')
    ax.set_xlabel(r'$x$ [M]')
    circle1 = plt.Circle((h1x[arg_time1], h1y[arg_time1]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle1)
    circle2 = plt.Circle((h2x[arg_time2], h2y[arg_time2]), 0.5, color='k', zorder  = 2)
    ax.add_patch(circle2)

    ax = mag_axs[2]
    im2 = ax.imshow(data_smallb2_xz/data_rho_xz, norm='log', origin='lower', extent=[x0[0], x1[0], x0[1], x1[1]], vmin=vmin, vmax=vmax, cmap='magma')
    ax.set_xlabel(r'$x$ [M]')
    ax.set_ylabel(r'$z$ [M]')
    ax.yaxis.set_label_position("right")

    mag.subplots_adjust(bottom=0.15, wspace=0.05)

    cbar = fig.colorbar(im, ax=mag_axs, orientation='horizontal', pad=0.05, shrink = 1, aspect = 50, location='top')
    cbar.ax.set_title(r'$b^2/\rho_0$')

    plt.savefig("%s_%s.pdf"%(sim_name, str(output_num).zfill(4)), bbox_inches='tight')

    return
