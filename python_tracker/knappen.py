# if __name__ == "__main__":
#     raise Exception("This file is not meant to be run as a script. Please import it from another python file.")

from typing import Callable, Tuple, Union
import numpy as np
from numpy import ndarray
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.optimize import fsolve
from sympy import symbols
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.patches as patches
from scipy.interpolate import interp1d


EtaEta = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

invGeVinsec = 1/(1.52* (10**24))
invGeVincm = 1.9732697880000002 * (10 ** -14)
ATLASradiiPixel = np.array([3.10, 5.05, 8.85, 12.25, 29.9, 37.1, 44.3, 51.4])
ATLASzrange = [32, 40, 40, 40, 74.9, 74.9, 74.9, 74.9]
lengthinPhi = [0.005, 0.005, 0.005, 0.005] + [2*element for element in [0.0017, 0.0017, 0.0017, 0.0017]]

phisize = [0.0016129,0.000990099,0.000564972,0.000408163,0.000113712,0.0000916442,0.0000767494,0.0000661479]
zsize = [0.025, 0.04, 0.04, 0.04] + [2*element for element in [0.0580, 0.0580, 0.0580, 0.0580]]



NA = 6.022 * 10**23
me = .511 * 10**-3
re = (2.81794033 * 10**-15 * 100) # fm m / (10^15 fm) * (100 cm) / m
eV2GeV = 10**-9
hbarOmegaP = {"Si": 31.05}
SternheimerA = {"Si": 0.1492}
SternheimerK = {"Si": 3.2546}
SternheimerX0 = {"Si": 0.2015}
SternheimerX1 = {"Si": 2.8716}
SternheimerI = {"Si": 173}
SternheimerCbar = {"Si": 4.4355}
SternheimerDelta0 = {"Si": 0.14}
Z = {"Si": 14}
A = {"Si": 28.085}
Rho = {"Si": 2.33}
X0 = {"Si": 9.37}
Lambda = {"Si": 46.52}
Lambdac = {"Si": 30.16}
Eehole = {"Si": 3.6}

Thicknesses = [0.023, 0.025, 0.025, 0.025] + [0.0285, 0.0285, 0.0285, 0.0285]
delta_r = Thicknesses


def beta(gamma: Union[float, ndarray]) -> Union[float, ndarray]:
    """Calculate the relativistic beta parameter.

    Parameters:
    gamma (float or ndarray): Lorentz factor.

    Returns:
    float or ndarray: Relativistic beta value.
    """

    return np.sqrt(1 - 1 / np.array(gamma)**2)


def eta(gamma: Union[float, ndarray]) -> Union[float, ndarray]:
    """Calculate the relativistic eta parameter.

    Parameters:
    gamma (float or ndarray): Lorentz factor.

    Returns:
    float or ndarray: Relativistic eta value.
    """

    return np.sqrt(np.array(gamma)**2 - 1)


def Tmax(ee: Union[float, ndarray], mm: Union[float, ndarray]) -> Union[float, ndarray]:
    """Compute the maximum kinetic energy of a secondary particle produced in a collision.

    Parameters:
    ee (float or ndarray): Energy of the incident particle.
    mm (float or ndarray): Mass of the incident particle.

    Returns:
    float or ndarray: Maximum kinetic energy.
    """

    ee = np.array(ee)
    mm = np.array(mm)
    return 2 * me * eta(ee / mm)**2 / (1 + 2 * eta(ee / mm) / beta(ee / mm) * me / mm + (me / mm)**2)


def Delta(eta: Union[float, ndarray], Elename: str) -> Union[float, ndarray]:
    """Calculate the density effect correction for radiation energy loss.

    Parameters:
    eta (float or ndarray): Relativistic eta parameter.
    Elename (str): Name of the element.

    Returns:
    float or ndarray: Density effect correction value.
    """

    eta = np.array(eta)
    if np.log10(eta) >= SternheimerX1[Elename]:
        return 2 * np.log(10) * np.log10(eta) - SternheimerCbar[Elename]
    elif np.log10(eta) >= SternheimerX0[Elename]:
        return 2 * np.log(10) * np.log10(eta) - SternheimerCbar[Elename] + (SternheimerA[Elename] * (SternheimerX1[Elename] - np.log10(eta))**SternheimerK[Elename])
    else:
        return SternheimerDelta0[Elename] * 10**(2 * (np.log10(eta) - SternheimerX0[Elename]))


def delta_p_PDG(x: float, gamma: float, z: int, EleName: Union[str, list]) -> float:
    """Calculate the mean excitation energy using the PDG formula.

    Parameters:
    x (float): Thickness of the medium.
    gamma (float): Lorentz factor.
    z (int): Atomic number of the medium.
    EleName (str or list): Element name or list of elements.

    Returns:
    float: Mean excitation energy.
    """

    if isinstance(EleName, str):
        tmplist = [[1, EleName]]
    else:
        tmplist = EleName

    return sum([i[0] * Rho[i[1]] * 4 * np.pi * NA * re**2 * me * z**2 *0.5 * Z[i[1]] * x / (A[i[1]] * beta(gamma)**2) * 
                 (np.log(2 * me * eta(gamma)**2 / (eV2GeV * SternheimerI[i[1]])) + 
                  np.log((i[0] * Rho[i[1]] * 4 * np.pi * NA * re**2 * me * z**2) * 
                         0.5 * Z[i[1]] * x / (A[i[1]] * beta(gamma)**2) / (eV2GeV * SternheimerI[i[1]])) - 
                 beta(gamma)**2 - Delta(eta(gamma), i[1]) + 0.2) for i in tmplist])



def dE(gamma: float, delta_r: float) -> float:
    """Calculate the energy loss of a particle per unit length.

    Parameters:
    gamma (float): Lorentz factor.
    delta_r (float): Thickness of the medium.

    Returns:
    float: Energy loss per unit length.
    """

    return (10**3) * delta_p_PDG(delta_r, gamma, 1, "Si")


def s(x1: ndarray, x2: ndarray) -> ndarray:
    """Calculate the unit direction vector from point x1 to x2.

    Parameters:
    x1, x2 (ndarray): Coordinates of two points.

    Returns:
    ndarray: Unit direction vector.
    """

    return (x1 - x2) / norm(x1 - x2)


def vpar(v: ndarray, x1: ndarray, x2: ndarray) -> ndarray:
    """Compute the parallel component of velocity v with respect to the direction from x1 to x2.

    Parameters:
    v (ndarray): Velocity vector.
    x1, x2 (ndarray): Coordinates of two points.

    Returns:
    ndarray: Parallel component of velocity.
    """

    return np.dot(v, s(x1, x2)) * s(x1, x2)


def vper(v: ndarray, x1: ndarray, x2: ndarray) -> ndarray:
    """Compute the perpendicular component of velocity v with respect to the direction from x1 to x2.

    Parameters:
    v (ndarray): Velocity vector.
    x1, x2 (ndarray): Coordinates of two points.

    Returns:
    ndarray: Perpendicular component of velocity.
    """

    return v - vpar(v, x1, x2)


def gammasc(v: Union[float, ndarray]) -> Union[float, ndarray]:
    """Compute the Lorentz factor for a given velocity.

    Parameters:
    v (float or ndarray): Velocity.

    Returns:
    float or ndarray: Lorentz factor.
    """

    return 1/np.sqrt(1 - v**2)


def gamma(v: ndarray) -> float:
    """Calculate the Lorentz factor for a given velocity vector.

    Parameters:
    v (ndarray): Velocity vector.

    Returns:
    float: Lorentz factor.
    """

    return 1/(np.sqrt(1 - np.dot(v, v)))


def Lorentz(p4: ndarray) -> ndarray:
    """Calculate the Lorentz transformation matrix for a given 4-vector.

    Parameters:
    p4 (ndarray): 4-vector.

    Returns:
    ndarray: Lorentz transformation matrix.
    """

    v = norm(p4[1:4])/p4[0]
    n = p4[1:4]/norm(p4[1:4])
    gscv = gammasc(v)
    return np.array([
        [gscv, -gscv * v * n[0], -gscv * v * n[1], -gscv * v * n[2]],
        [-gscv * v * n[0], 1 + (gscv - 1) * n[0]**2, (gscv - 1) * n[0] * n[1], (gscv - 1) * n[0] * n[2]],
        [-gscv * v * n[1], (gscv - 1) * n[1] * n[0], 1 + (gscv - 1) * n[1]**2, (gscv - 1) * n[1] * n[2]],
        [-gscv * v * n[2], (gscv - 1) * n[2] * n[0], (gscv - 1) * n[2] * n[1], 1 + (gscv - 1) * n[2]**2]
    ])


def DpDt(v: ndarray, x1: ndarray, x2: ndarray, q: float, ECM: ndarray, BCM: ndarray, T: float, quirkflag: bool) -> ndarray:
    """Compute the rate of change of momentum for a particle in an electromagnetic field.

    Parameters:
    v (ndarray): Velocity vector.
    x1, x2 (ndarray): Coordinates of two points.
    q (float): Particle charge.
    ECM (ndarray): Electric field in center-of-mass frame.
    BCM (ndarray): Magnetic field in center-of-mass frame.
    T (float): Parameter related to string tension for quirk pairs.
    quirkflag (bool): Flag to indicate if the particle is a quirk.

    Returns:
    ndarray: Rate of change of momentum.
    """

    if quirkflag:
        vp = vper(v, x1, x2)
        vp_dot_vp = np.dot(vp, vp)
        s_x1_x2 = s(x1, x2)
        return (-T * ((np.sqrt(1 - vp_dot_vp) * s_x1_x2) + (np.dot(vpar(v, x1, x2), s_x1_x2) * vp / np.sqrt(1 - vp_dot_vp)))) + q * (ECM + np.cross(v, BCM))
    else:
        return q * (ECM + np.cross(v, BCM))


def FindTracks(vec1: ndarray, vec2: ndarray, rootsigma: float, quirkflag: bool, B: ndarray = np.array([0, 0, 1.18314 * 10**-16])) -> Tuple[Callable[[ndarray], ndarray], Callable[[ndarray], ndarray], float]:
    """Find the trajectories of particles based on initial 4-vectors.

    Parameters:
    vec1, vec2 (ndarray): Initial 4-vectors of the particles.
    rootsigma (float): Root of the sigma value.
    quirkflag (bool): Flag to indicate if the particle is a quirk.
    Optional: B (ndarray): Magnetic field in center-of-mass frame.

    Returns:
    tuple: Interpolated solutions for the trajectories and the maximum time value.
    """

    m = np.sqrt(np.dot(vec1, EtaEta.dot(vec1)))
    T = (rootsigma * (10**-9))**2
    boost = Lorentz((vec1 + vec2))
    boostback = Lorentz(np.dot(EtaEta, vec1 + vec2))
    vCM = (vec1 + vec2)[1:4] / (vec1 + vec2)[0]
    ECM = gamma(vCM) * np.cross(vCM, B)
    BCM = gamma(vCM) * B - ((gamma(vCM) - 1) * (np.dot(B, vCM / norm(vCM))) * (vCM / norm(vCM)))
    tmax = False
    e = np.sqrt(4 * math.pi * 1/137)
    
    def ode_system(t, y):
        gammabeta1 = y[:3]
        gammabeta2 = y[3:6]
        x1 = y[6:9]
        x2 = y[9:12]
        Ef = y[12:15]
        dgammabeta1dt = (1/m) * DpDt((gammabeta1/np.sqrt(1+(np.dot(gammabeta1,gammabeta1)))), x1, x2, 1, Ef, BCM, T, quirkflag)
        dgammabeta2dt = (1/m) * DpDt((gammabeta2/np.sqrt(1+(np.dot(gammabeta2,gammabeta2)))), x2, x1, -1, Ef, BCM, T, quirkflag)
        dx1dt = gammabeta1/np.sqrt(1+(np.dot(gammabeta1,gammabeta1)))
        dx2dt = gammabeta2/np.sqrt(1+(np.dot(gammabeta2,gammabeta2)))
        dEfdt = np.array([0,0,0])
        dydt = np.concatenate((dgammabeta1dt, dgammabeta2dt, dx1dt, dx2dt, dEfdt))
        return dydt

    # Find tmax
    def event_function(t, y): 
        vec = np.array([invGeVinsec, invGeVincm, invGeVincm, invGeVincm])
        sumx12 = np.array([y[6],y[7], y[8]]) + np.array([y[9],y[10], y[11]])
        return np.linalg.norm((vec * np.dot(boostback, np.array([t, sumx12[0],sumx12[1], sumx12[2]])))[1:3]) - 60

    # Initial conditions
    gammabeta1_0 = (1/m) * (np.dot(boost,vec1)[1:4])
    gammabeta2_0 = (1/m) * (np.dot(boost,vec2)[1:4])
    x1_0 = np.array([0,10**(-5),0])
    x2_0 = np.array([0,-10**(-5),0])
    Ef_0 = ECM
    y0 = np.concatenate((gammabeta1_0, gammabeta2_0, x1_0, x2_0, Ef_0))
    initial_state = y0
    
    event_function.terminal = True

    # Integration time span
    t_span = (0.1, 10**19) 
    solution = solve_ivp(ode_system, t_span, initial_state,method = 'RK23', t_eval = np.linspace(0.1,10**19,10000000), events = event_function)
    
    tmax_lst = solution.t_events[0]

    if len(tmax_lst) > 0:
        tmax = tmax_lst[0]
        solution.t = np.append(solution.t, tmax)
        solution.y = np.concatenate((solution.y, solution.y_events[0][0][:, np.newaxis]), axis=1)

        
    sol1_keys, sol1_values, sol2_keys, sol2_values = [], [], [], []
    for i in range(0,len(solution.t)):
        sol1_keys.append(solution.t[i])
        sol1_values.append(np.array([invGeVinsec, invGeVincm, invGeVincm, invGeVincm]* np.dot(boostback, np.array([solution.t[i], solution.y[6,i], solution.y[7,i], solution.y[8,i]]))))
        sol2_keys.append(solution.t[i])
        sol2_values.append(np.array([invGeVinsec, invGeVincm, invGeVincm, invGeVincm]* np.dot(boostback, np.array([solution.t[i], solution.y[9,i], solution.y[10,i], solution.y[11,i]]))))

    sol1_keys = np.array(sol1_keys)
    sol1_values = np.array(sol1_values)
    sol2_keys = np.array(sol2_keys)
    sol2_values = np.array(sol2_values)

    
    sol1 = interp1d(sol1_keys, sol1_values, axis=0, bounds_error=False, fill_value=None)
    sol2 = interp1d(sol2_keys, sol2_values, axis=0, bounds_error=False, fill_value=None)

    return sol1, sol2, tmax


def FindEdges(vec1: ndarray, vec2: ndarray, root_sigma: float, layernr: int) -> list:
    """Find the boundary edges where the particles might intersect with a detector layer.

    Parameters:
    vec1, vec2 (ndarray): Initial 4-vectors of the particles.
    root_sigma (float): Root of the sigma value.
    layernr (int): Layer number of the detector.

    Returns:
    list: List of intersection points or None if no intersections.
    """

    vcm = (vec1 + vec2) / 2
    vcmInv = np.array([vcm[0], -vcm[1], -vcm[2], -vcm[3]])
    lvcm = Lorentz(vcm)
    vecscm = np.array([lvcm.dot(v) for v in [vec1, vec2]])
        
    m = np.sqrt(vecscm[0].dot(EtaEta).dot(vecscm[0]))
    sigma = (root_sigma*10**-9)**2
    v0 = np.linalg.norm(vecscm[0][1:]) / vecscm[0][0]

    tmax = (m * v0) / (np.sqrt(1 - v0**2) * sigma)
    rmax = (m * (-1 + 1/np.sqrt(1 - v0**2))) / sigma


    endpoints = np.array([[tmax,((v[1:4]/norm(v[1:4])) * rmax)[0],((v[1:4]/norm(v[1:4])) * rmax)[1],((v[1:4]/norm(v[1:4])) * rmax)[2]] if norm(v[1:4]) != 0 else [tmax,(np.array([0,0,0]) * rmax)[0],(np.array([0,0,0]) * rmax)[1],(np.array([0,0,0]) * rmax)[2]] for v in vecscm])
    endpointslab = np.array([Lorentz(vcmInv).dot(e) for e in endpoints])

    t = symbols('t')

    line1 = (vcm[1:4] * t + endpointslab[0, 1:4]) * np.array([invGeVincm, invGeVincm, invGeVincm])
    line2 = (vcm[1:4] * t + endpointslab[1, 1:4]) * np.array([invGeVincm, invGeVincm, invGeVincm])
    eq1 = sp.Eq(line1[:2].dot(line1[:2]), ATLASradiiPixel[layernr-1]**2)
    eq2 = sp.Eq(line2[:2].dot(line2[:2]), ATLASradiiPixel[layernr-1]**2)

    eq1tsol, eq2tsol = sp.solve(eq1, t), sp.solve(eq2, t)
    
    line1sol = np.array([line1[2].subs(t,eq1tsol[i]) for i in range(len(eq1tsol))])
    line2sol = np.array([line2[2].subs(t,eq2tsol[i]) for i in range(len(eq2tsol))])
    solutions = sp.flatten([line1sol,line2sol])

    r = np.array([float(sol.evalf()) for sol in solutions if isinstance(sol.evalf(), sp.core.numbers.Float)])
    
    return r if len(r) > 0 else [0]
   

epsilon = 10**6
inGeVtoNanoSec = 10**9 / (1.52 * 10**24)


def beta_beta(trajMap: interp1d, t0: float) -> float:
    """Calculate the beta parameter at a given time for a trajectory.

    Parameters:
    trajMap (function): Function representing the trajectory.
    t0 (float): Time at which beta is to be calculated.

    Returns:
    float: Beta value at the given time.
    """

    return ((np.linalg.norm(trajMap([t0 + epsilon])[0][1:4]) - np.linalg.norm(trajMap([t0])[0][1:4])) / (trajMap([t0 + epsilon])[0][0] - trajMap([t0])[0][0])) / (3 * 10**10)


def gamma_gamma(traj: interp1d, t0: float) -> float:
    """Calculate the Lorentz factor at a given time for a trajectory.

    Parameters:
    traj (function): Function representing the trajectory.
    t0 (float): Time at which the Lorentz factor is to be calculated.

    Returns:
    float: Lorentz factor at the given time.
    """

    return 1 / np.sqrt(1 - (beta_beta(traj, t0)**2))


def FindIntersections(traj: interp1d) -> list:
    """Find the intersection points of a particle trajectory with various detector layers.

    Parameters:
    traj (interp1d): Function representing the trajectory.

    Returns:
    list: List of intersection points with various layers in the detector.
    """

    t_values = traj.x
    r_values = np.linalg.norm([traj(t)[1:3] for t in t_values], axis=1)
    
    coarselist = np.column_stack((t_values[:-1], r_values[:-1], t_values[1:], r_values[1:]))

    finallist = []

    for layer in range(8):
        layerlist = coarselist[(coarselist[:, 1] - ATLASradiiPixel[layer]) * (coarselist[:, 3] - ATLASradiiPixel[layer]) < 0]
        sols = [fsolve(lambda t: np.linalg.norm(traj(t)[1:3]) - ATLASradiiPixel[layer], (t_low+t_high)/2) for t_low, _, t_high, _ in layerlist]

        for sol in sols:
            t = sol[0]
            finallist.append([
                (layer + 1) ,
                round(traj(t)[3] / zsize[1]) * zsize[1] ,
                round(np.arctan2(traj(t)[2], traj(t)[1])/ phisize[layer]) * phisize[layer] ,
                dE(gamma_gamma(traj, t), delta_r[layer]),
                abs(beta_beta(traj, t) * gamma_gamma(traj, t)),
                inGeVtoNanoSec * gamma_gamma(traj, t) * t
            ])

    return finallist


def RunPoint(vec1, vec2, root_sigma, plotflag, quirkflag) -> list:
    """Run the simulation for a given initial condition and return the intersections with detector layers.

    Parameters:
    vec1, vec2 (ndarray): Initial 4-vectors of the particles.
    root_sigma (float): Root of the sigma value.
    plotflag (bool): Flag to indicate if plots should be generated.
    quirkflag (bool): Flag to indicate if the particle is a quirk.

    Returns:
    list: Sorted list of intersections with the detector layers.
    """

    findedges1,findedges2 = FindEdges(vec1, vec2, root_sigma, 1),FindEdges(vec1, vec2, root_sigma, 2)

    if (findedges1 is not None and min(map(abs,findedges1)) < ATLASzrange[0]) or (findedges2 is not None and min(map(abs,findedges2)) < ATLASzrange[1]):
        sol1, sol2, tmax = FindTracks(vec1, vec2, root_sigma, quirkflag)
        sol1_array = sol1.y[0]

        if plotflag:
            sol1_array = sol1.y
            sol2_array = sol2.y

            fig, ax = plt.subplots()

            # Plotting the trajectories
            ax.plot(sol1_array[:, 1], sol1_array[:, 2], 'b-')
            ax.plot(sol2_array[:, 1], sol2_array[:, 2], 'r-')

            for i in ATLASradiiPixel:
                circle = patches.Circle((0,0), i, fill=False)
                ax.add_patch(circle)

            rad = ATLASradiiPixel[-1]
            ax.set_xlim(-rad, rad)
            ax.set_ylim(-rad, rad)

            ax.axhline(0, color='black',linewidth=0.5)
            ax.axvline(0, color='black',linewidth=0.5)

            # Making the aspect ratio of the plot 1:1 so circles look like circles
            ax.set_aspect('equal', adjustable='box')
            plt.show()


        intersections1 = FindIntersections(sol1)
        intersections2 = FindIntersections(sol2)

        return sorted([[*i, 1] for i in intersections1] + [[*i, 2] for i in intersections2], key=lambda x: tuple(x))

    else:
        return [[0]]
    

vecs1 = np.array([[421.69956147, 258.12146064, 
    154.10248991, -254.86516886], [202.65928421, -123.22431566, 
    12.442253162, 56.848428683]])

AA = RunPoint(np.array(vecs1[0]), np.array(vecs1[1]), 500, False, True)
print(AA)