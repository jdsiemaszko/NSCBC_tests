"""
USAGE: python post_processing.py <TEMPORAL folder path> <mode:{'inlet' or 'outlet'}> <forcing file path> <relaxation value>
"""


import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy
from planar_wave_extraction import main as PWE
from scipy.optimize import curve_fit
from scipy.ndimage import shift
from scipy.signal import welch

c = 340.29
dt = 1e-6

fmin = 150
fmax = 15000
fmax_interp = 10000

probe_index=50


mode = sys.argv[2] if len(sys.argv) > 2 else 'outlet'
print(f"using mode: {mode}")

if len(sys.argv) > 3:
    forcing_file = sys.argv[3]
else:
    forcing_file = 'BC/signal_alt.U' if mode=="outlet" else 'BC/signal.U'

if len(sys.argv) > 4:
    sigma = float(sys.argv[4])
    Kref = 2 * c * sigma

dir = sys.argv[1] if len(sys.argv) > 1 else 'TEMPORAL'
print('using dir:', dir)


ua, pa, fa, ga, f, g, p, u, time, x, y, z = PWE(plots=False, directory=dir)
deltat = time[1]-time[0]

fa = f[probe_index, :]
ga = g[probe_index, :]

factor = 1.0

Lref = x[-1] - x[0]

deltat_shift_right = x[probe_index] / c
deltat_shift_left = (Lref-x[probe_index]) / c

# --- Welch’s method instead of FFT ---
fs = 1.0 / (time[1] - time[0])

f, Pxx_fa = welch(fa, fs=fs, nperseg=1024)
_, Pxx_ga = welch(ga, fs=fs, nperseg=1024)

forcingdata = np.loadtxt(forcing_file, delimiter =',')
forcingtime = forcingdata[0, :]
forcingval = forcingdata[1, :]

forcing = np.interp(time, forcingtime, forcingval)

f_forcing, Pxx_forcing = welch(forcing, fs=fs, nperseg=1024)

# --- time-domain plot ---
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(time - deltat_shift_right, fa, label=r'$f_{tot}$', color='r', alpha=0.5)
ax.plot(time - deltat_shift_left, ga, label=r'$g_{tot}$', color='b', alpha=0.5)
ax.plot(time, forcing, label='outlet forcing' if mode=="outlet" else "inlet forcing", color='k', linestyle='dashed')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
plt.close(fig)

# --- frequency-domain plot (Welch spectra) ---
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.sqrt(Pxx_fa), label=r'$f_{tot}$', color='r', alpha=0.5)
ax.plot(f, np.sqrt(Pxx_ga), label=r'$g_{tot}$', color='b', alpha=0.5)
ax.plot(f_forcing, np.sqrt(Pxx_forcing), label='outlet forcing' if mode=="outlet" else "inlet forcing", 
        color='k', linestyle='dashed', alpha=0.5)

ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(1.0, 2.0*fmax)

plt.tight_layout()
plt.show()
plt.close(fig)


def model(f, tau):
    return np.abs(-1 / (1 + 1j * 2 * np.pi * f * tau))

def fit_tauIO(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(-1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov

def fit_tauFI(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(1.0 - 1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov

def fit_tauInletv2(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(1.0 - 1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov


# leftovers

# Use Welch-based ratio as transfer function estimate
reflection_coeff_total = np.sqrt(Pxx_fa) / np.sqrt(Pxx_ga) if mode=="outlet" else np.sqrt(Pxx_ga)/np.sqrt(Pxx_fa)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(reflection_coeff_total), label=r"$|r_{inlet}|$" if mode=="outlet" else r"$|r_{outlet}|$", color='r', alpha=0.5)
args = np.where((f > fmin) & (f < fmax_interp))[0]
tau_IN, vfitIN, popt, pcov = fit_tauIO(f[args], np.abs(reflection_coeff_total)[args])
if mode=="outlet":
    print(f'sigma INLET: {2 / c / tau_IN:.8f}')
else:
    print(f'sigma OUTLET: {2 / c / tau_IN:.8f}')

ax.plot(f[args], vfitIN, color='b', linestyle='dashed', linewidth=2, 
        label=rf'best fit' + f', $\\sigma = {2 / c / tau_IN:.8f}$')

ax.plot(f[args], model(f[args], tau = 2 / c / sigma), color='k', linestyle='dashed', linewidth=3, 
        label=rf'theoretical' + f', $\\sigma = {2 * sigma}$')

ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(1.0, 2.0*fmax)
# ax.set_aspect('equal')

plt.tight_layout()
plt.show()
plt.close(fig)
