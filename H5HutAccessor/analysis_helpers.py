"""
Dumping ground for a bunch of misc helper functions for OPAL analysis
"""

from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
from scipy.signal import argrelmin
from scipy.constants import (speed_of_light as clight, 
                            elementary_charge as jperev, 
                            elementary_charge as echarge, proton_mass)
from scipy.spatial import KDTree
from datetime import datetime
from itertools import cycle

# homebrew classes and functions
from CARBONCYCLFieldMap import CARBONCYCLFieldMap

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm

from collections import namedtuple, defaultdict
import os
import glob


POLE_ANGLE_D = 52.42     # deg
POLE_ANGLE = np.pi/180. * POLE_ANGLE_D # rad
POLE_OFFSET = 0.73   # m
GAP_WIDTH = 5e-2  # gap between quadrupole ends. per Dior, this is 5 cm space left for each quad's end windings
OUTER_CUTOFF = 4.7  # outer radial cutoff (m) for testing against arcs

RF_FREQ = 116.4e6  # in Hz
HARM = 25
N_SECTORS = 6
DPHI = -2*np.pi * HARM/N_SECTORS
T_OFFSET = -4.2728e-9
PHI0 = -25 * np.pi/180. + 2*np.pi*T_OFFSET*RF_FREQ
PHI1 = PHI0 + DPHI
PHI2 = PHI0 + 2*DPHI
PHI3 = PHI0 + 4*DPHI
PHI4 = PHI0 + 5*DPHI
PHI = np.array((PHI1, PHI2, PHI3, PHI4)) 

matplotlib.rcParams['font.size'] = 16
PMASSEV = proton_mass / jperev * clight**2

def rhat(theta):
    r""" 
    Returns the unit vector $\hat{r} = cos(\theta) \hat{x} + sin(\theta) \hat{y}$
    
    theta - angle in radians
    """
    theta = np.asarray(theta)
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1).squeeze()

def phihat(theta):
    r""" 
    Returns the unit vector $\hat{phi} = -sin(\theta) \hat{x} + cos(\theta) \hat{y}$
    
    theta - angle in radians
    """
    theta = np.asarray(theta)
    return np.stack((-np.sin(theta), np.cos(theta)), axis=-1).squeeze()

def plot_turns(ax, turns, ymin=None, ymax=None):
    colors = cycle(((0.0, 0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 0.3)))
    turnx_curr = 0
    ymin = ymin or ax.get_ylim()[0]
    ymax = ymax or ax.get_ylim()[1]
    xmax = ax.get_xlim()[1]
    for turnnum, turnx in enumerate(turns, 1):
        if turnx_curr > xmax:
            return
        ax.add_patch(plt.Rectangle((turnx, ymin), turnx_curr-turnx, ymax-ymin, color=colors.next()))
        ax.annotate(xy=(turnx, ymax*0.9), s=('Turn %d' % turnnum))
        turnx_curr = turnx

angles = (60, 120, 240, 300)
Vs = (2.407131677, 2.418053569, 2.412942259, 2.419481862)
rmins = (1.1513, 1.1702, 1.2335, 1.2389)
rmaxs = (4.2996, 4.3203, 4.3272, 4.2382)

def timeNow():
    return "{:%Y-%m-%d %H:%M}".format(datetime.now())

def plot_probes(fig, ax, angles=angles, rmins=rmins, rmaxs=rmaxs):
    plt.figure(fig.number)
    for angle, rmin, rmax in zip(angles,
                                 rmins,
                                 rmaxs):
        r = np.linspace(rmin, rmax, 100)
        x = r*np.cos(angle*np.pi/180.)
        y = r*np.sin(angle*np.pi/180.)
        for a in (angle-1, angle+1):
            xmin = 0.9*np.cos(a * np.pi/180.)
            xmax = rmax*np.cos(a * np.pi/180.)
            ymin = 0.9*np.sin(a * np.pi/180.)
            ymax = rmax*np.sin(a * np.pi/180.)
            ax.plot([xmin, xmax], [ymin, ymax], 'w-', linewidth=2)


def plot_ideal_xy(ax, plotMarkers=False):
    trajectory = ideal_trajectory('path-forJames.dat')
    pltopts = {'color': 'k', 'linestyle': 'solid', 'linewidth': 3}
    if plotMarkers:
        pltopts.update({'marker': '^', 'markeredgecolor': 'black', 
                        'markerfacecolor': 'black', 'markersize': 10})

    ax.plot(trajectory.xg, trajectory.yg, **pltopts)

def probedata(fn):
    r"""
    Retrieve OPAL probe data from the probe file `fn`
    
    Returns
    -------
    header, probedata
    """
    with open(fn) as f:
        header = f.readline()
        probestats = np.array([float(e) for line in f.readlines() for e in line.split()[1:]])
        probestats = probestats.reshape((probestats.size//9,9))

    return header, probestats

def getphases(mask):
    """ Given a mask compatible with glob.glob(), return a dictionary of phases from matching PROBE files """
    times = probetimes(mask)
    phases = {k: ((time*1e-9*(2*np.pi*RF_FREQ) + PHI) % (np.pi) - np.pi) * 180./np.pi for k,time in times.items()}
    if len(phases) == 1:
        return list(phases.values())[0]
    else:
        return phases

def probetimes(mask, wd='.'):
    """
    Parse all files matching PROBE*.loss (or a user-supplied mask for glob.glob()) 
    and return a probe timing dictionary, for ONLY the reference (pid=0) particle.

    Parameters
    ----------
    mask - file mask compatible with glob.glob() for probe files
    wd - (optional) working directory to search under (defaults to cwd)

    Returns
    -------
    data - dict with structure: `{sim_name: crossing_times}` where 
        `crossing_times` is an ndarray with shape (Ncrossings, Nprobes)
        giving the probe crossing time (in ns) for each probe event
    """
    files = glob.glob(os.path.join(wd, mask))
    data = defaultdict(dict) 
    for f in files:
        k = os.path.basename(os.path.dirname(f))
        fn = os.path.basename(f)
        data[k][fn] = probedata(f)[1]

    # now that all data is loaded, squash the data into single arrays
    for sim, d in data.items():
        Nprobes = len(d.values())
        maxrows = max(arr.shape[0] for arr in d.values())
        crossing_times = np.full(shape=(Nprobes, maxrows), fill_value=np.nan)
        for idx, arr in enumerate(d.values()):
            refidx = arr[:, -3] == 0  # select only rows where pid=0
            Ncrossing = refidx.size
            crossing_times[idx, :Ncrossing] = arr[refidx, -1]
        data[sim] = crossing_times.T

    return data

def probephase(tns, phi, freq=RF_FREQ):
    """ 
    Given time in ns `tns` and an initial phase and RF frequency (Hz), calculate the 
    RF phase (rad) at each time. 
    """
    return (2*np.pi*freq*tns*1e-9 + phi) % (np.pi) - np.pi

def bg_to_p(bgx, bgy=None, mass=PMASSEV):
    r"""
    Convert beta*gamma to momentum (defaults to eV/c)
    """
    if bgy is not None:
        return np.sqrt(bgx**2 + bgy**2) * mass
    else:
        return bgx * mass

IdealTrajectory = namedtuple('IdealTrajectory', ['xg', 'yg', 'rho', 'xc', 'yc', 'Bshape', 's'])
OPALTrajectory = namedtuple('OPALTrajectory', ['xg', 'yg', 's'])
def ideal_trajectory(fn):
    r"""
    Read output from Dior's path optimization and return namedtuple with the properties:
    xg, yg, rho, xc, yc, Bshape, s
    """
    with open(fn, 'r') as f:
        header = f.readline()
        f.readline()
        data = np.asarray([float(e) for line in f.readlines() for e in line.split()])
        data = data.reshape(data.size//7, 7)
        # cols = [col.flatten() for col in np.hsplit(data, 7)]
        cols = np.hsplit(data, 7)
        return IdealTrajectory(*cols)

class PathTree():
    r"""
    Helper class for looking up the nearest point on an ideal (Dior) trajectory
    """
    def __init__(self, fn):
        self.trajectory = ideal_trajectory(fn)
        idealpath = np.hstack((self.trajectory.xg, self.trajectory.yg))
        self.pathtree = KDTree(idealpath)

    def nearest_ideal_point(self, xpts, ypts, **kwargs):
        r""" Returns dist, idx of points on ideal path nearest to (xpts, ypts) """
        pts = np.stack((xpts.flatten(), ypts.flatten()), axis=1)
        return self.pathtree.query(pts, **kwargs)


def plot_trajectory_separation(ax, h, max_separation=0.05, sl=slice(None), indept_var=None, indept_var_label='time (ns)'):
    r"""
    Plot the separation between the ideal path (as dictated by the magic file) and the OPAL path
    ax - the matplotlib axis to plot on
    h - the "real" path from OPAL, to be compared against the ideal (H5HutAccessor object)
    max_separation=0.05 - maximum distance (m) to look for a nearest neighbor in the KDTree
    sl - slice object to apply over h
    indept_var - array to use as independent variable for plotting, must be same length as h.x[sl]
    indept_var_label - label for independent variable
    """
    idealpath_tree = PathTree('path-forJames.dat')
    # for each entry in the OPAL path, get the nearest ideal trajectory point
    d, idx = idealpath_tree.nearest_ideal_point(h.x[sl], h.y[sl], distance_upper_bound=max_separation)
    if indept_var is None:
        indept_var = h.t[sl] * 1e9
    ax.plot(indept_var, d*1e2, 'b-', marker='s', mfc='b')
    ax.set_xlabel(indept_var_label)
    ax.set_ylabel("Ideal-OPAL orbit separation (cm)")


def P_to_T(P, M=PMASSEV):
    r"""
    Convert a momentum to a kinetic energy
    """
    gamma = np.sqrt(1 + (P / M)**2)
    return (gamma-1)*M

def T_to_P(T, M=PMASSEV):
    r"""
    Convert a kinetic energy to a momentum
    """
    return np.sqrt(2*M*T)

def extract_ideal_energy(fn):
    r"""
    Returns (s, T) where s is the distance traveled by the beam and T is the kinetic energy in MeV
    """
    with open(fn, 'r') as f:
        f.readline()
        f.readline()
        linedata = np.asarray([float(value) for line in f.readlines() for value in line.split()])
        xg, yg, rho, xc, yc, B, distance = np.hsplit(linedata.reshape(linedata.size//7, 7), 7)
        sl = B > 1e-3
    P = B*rho * 1e9 / 3.3356 # in eV/c
    T = P_to_T(P) # in eV
    distance = np.concatenate(([0], distance[sl]))
    T = np.concatenate(([T[sl].min()], T[sl]))
    return distance, T/1e6

def turn_transitions(x, y, angle=0.0):
    r"""
    return array of indices of the arrays x, y indicating where the given particle (column) crossed
    the line defined by theta=angle (rad) (default = 0) in the last step.

    Parameters
    ----------
    x, y - 1d array_like (nturns,)
        of x and y coordinates
    angle - float
        defining angle (rad) for the transition line
    
    Returns
    -------
    indices - tuple (nparticles)
        indices of the crossing step for each particle

    Notes
    -----
    any `angle` can be provided, but the calculation will be done modulo 2*pi
    """
    assert np.ndim(x) == 1 and np.ndim(y) == 1, "input must be 1d"
    theta = np.arctan2(y, x)
    return argrelmin((theta + angle) % (2*np.pi))[0]

def create_probe(probenum, angle, rmin, rmax):
    r"""
    Creates the OPAL representation of a probe

    TODO: add azimuthal shift to align with cavities?
    """
    xmin = rmin*np.cos(angle * np.pi/180.)
    xmax = rmax*np.cos(angle * np.pi/180.)
    ymin = rmin*np.sin(angle * np.pi/180.)
    ymax = rmax*np.sin(angle * np.pi/180.)
    return "PROBE{}: PROBE, XSTART={:.2f}, XEND={:.2f}, YSTART={:.2f}, YEND={:.2f};".format(
                probenum, 1e3*xmin, 1e3*xmax, 1e3*ymin, 1e3*ymax)

def plot_cavities(fn, ax=None, vmin=0, vmax=10):
    """
    Helper function to plot cavities 1 thru 4 given a format-string that takes a cavity number 

    Parameters
    ----------
    fn - template path to HDF5 files, e.g. '/path/to/cav%d.h5', which will be 
    passed to str.format() with the cavity number as a lone argument
    """
    return [plot_cavity(fn.format(i), ax, vmin=vmin, vmax=vmax) for i in range(1, 5)]

def plot_cavity(fn, ax=None, norm_min=0.01, vmin=0, vmax=10):
    """ Given an H5Hut file `fn`, plot the fieldmap (defaults to current axis) """
    if ax is None:
        ax = plt.gca()
        
    with h5py.File(fn, 'r') as f:
        xi, yi, zi = f['Step#0/Block/Efield'].attrs['__Origin__']
        sx, sy, sz = f['Step#0/Block/Efield'].attrs['__Spacing__']

        Ex = f['Step#0/Block/Efield/0'][()]
        Ey = f['Step#0/Block/Efield/1'][()]
        Ez = f['Step#0/Block/Efield/2'][()]

        nz, ny, nx = Ex.shape
        xg = np.arange(xi, xi + nx * sx, sx)
        yg = np.arange(yi, yi + ny * sy, sy)
        zg = np.arange(zi, zi + nz * sz, sz)
        Z, Y, X = np.meshgrid(zg, yg, xg, indexing='ij')

        Enorm = np.linalg.norm((Ex, Ey, Ez), axis=0)
        Enorm = np.ma.masked_array(Enorm, mask=(Enorm < norm_min))

        s = nz//2, slice(None), slice(None)
        cax = ax.pcolormesh(X[s]/1e3, Y[s]/1e3, Enorm[s], cmap='inferno')
        return cax

def pole_origin(secnum):
    """ Return the properly-rotated origin of the polepiece for sector `secnum` """
    secang = 30 + 60*(secnum-1)
    # rotation origin for this sector
    c = np.cos(secang * np.pi/180.)
    s = np.sin(secang * np.pi/180.)
    ox = POLE_OFFSET*c 
    oy = POLE_OFFSET*s
    return ox, oy

def get_arcs(fn):
    """ 
    Get arc data from file `fn` 
    
    Parameters
    -----------
    fn - path or file to handle with columns:
        sector number, xcenter (m), ycenter (m), rho (m)
    
    Returns
    --------
    secnum, xc, yc, rho
    """
    data = np.loadtxt(fn, comments='#')
    secnum, xc, yc, rho = [c.squeeze() for c in np.split(data, 4, axis=1)]
    return (secnum, xc, yc, rho)

def arc_params(turnnum, secnum):
    """
    Given a turn number and sector number (starting from 1),
    return a dictionary giving the geometric information of the arc
    """
    _, xc, yc, rho = get_arcs('arcs.dat')

    secnum -= 1
    turnnum -= 1
    idx = secnum + 6*turnnum
    xci = xc[idx]
    yci = yc[idx]
    ri = rho[idx]

    THETA_MIN = secnum * np.pi/3
    
    c = np.cos(THETA_MIN + np.pi/6)
    s = np.sin(THETA_MIN + np.pi/6)
    # the arc centers are given in the sector's local frame, so we need to rotate into the global frame
    xci, yci = (c*xci - s*yci, s*xci + c*yci)
    
    return {'xc': xci, 'yc': yci, 'rho': ri}

def sector_lines(secnum):
    """
    Compute the boundary lines for sector `secnum`, i.e. (ox, oy, s1x, s1y, s2x, s2y) 
    where the lines between (ox, oy) and (sNx, sNy) define the two boundaries 
    of the sector (entrance and exit, respectively)
    """
    assert 1 <= secnum <= 6
    ox, oy = pole_origin(secnum)
    secang = (secnum-1) * np.pi/3. + np.pi/6.

    # from chat with Dior (Apr 2, 2018), can exclude fringe regions
    # by leaving the necessary 7.5cm space, and offsetting the pole origin
    # by this amount makes that exclusion the familiar angle comparison. no
    # need for lazily using `shapely` and doing (slow!) shape comparisons
    EDGE_OFFSET = 7.5e-2
    rh = rhat(secang)
    ox += EDGE_OFFSET/np.sin(POLE_ANGLE/2)*rh[0]
    oy += EDGE_OFFSET/np.sin(POLE_ANGLE/2)*rh[1]

    s1hat = rhat(secang - POLE_ANGLE/2.)
    s2hat = rhat(secang + POLE_ANGLE/2.)
    s1x = ox + 4.7 * s1hat[0]
    s1y = oy + 4.7 * s1hat[1]
    s2x = ox + 4.7 * s2hat[0]
    s2y = oy + 4.7 * s2hat[1]
    
    return ((ox, oy),
            (s1x, s1y), 
            (s2x, s2y))

def in_sector(x, y, secnum):
    """
    Given point(s) (x, y), determine which points are inside the specified sector
    
    With thanks to StackOverflow user Shard for the 
    https://math.stackexchange.com/a/274728
    """
    (ox, oy), (s1x, s1y), (s2x, s2y) = sector_lines(secnum)
    
    d1 = (x-ox)*(s1y - oy) - (y - oy)*(s1x - ox)
    d2 = (x-ox)*(s2y - oy) - (y - oy)*(s2x - ox)
    
    # the sign of the 'determinant' should be opposite if we're inside 
    # both lines, and the sector number tells us which half-plane we're in
    
    if 1 <= secnum <= 3:
        return (y > oy) & (np.sign(d1) == -np.sign(d2))
    else:
        return (y < oy) & (np.sign(d1) == -np.sign(d2))

assert all([
    in_sector(1.7, 1.2, 1),
    in_sector(0.2, 1.7, 2),
    in_sector(-1.7, 1.2, 3),
    in_sector(-1.7, -1.2, 4),
    in_sector(0.2, -1.7, 5),
    in_sector(1.7, -1.2, 6),
    not in_sector(0.2, 0.2, 1),
    not in_sector(0.2, 0.2, 2),
    not in_sector(0.2, 0.2, 3),
    not in_sector(0.2, 0.2, 4),
    not in_sector(0.2, 0.2, 5),
    not in_sector(0.2, 0.2, 6)
])
