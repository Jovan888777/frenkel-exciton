 # -*- coding: utf-8 -*-
"""
file: sample_spectral_density.py
author: Ardavan Farahvash (MIT)

description: 
    
Samples a spectral density function at a discrete set of frequencies.

------
Input: A spectral density function
K(w) = \sum_i c_i^2/w_i^2 \delta(w - w_i) = c^2(w)/w^2 \rho(w), where \rho(w) is the density of states
Functions are available for standard forms such as the Debye-Lorentz spectra.

-----
Output: a csv file containing the frequency and system-environment coupling of each mode. 
"""

import numpy as np
import matplotlib.pyplot as plt

##########  NICE PLOTS  ###########
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer-Modern']
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['xtick.labelsize'] =  18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['lines.linewidth'] = 1.5
###################################

# For this script we use a consistent unit system in order to make unit conversions easy
# The unit system is often called molecular dynamics units, 
# https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html
# Time - picoseconds
# Distance - nanometers
# Mass - g/mol
# Energy - kJ/mol
# Force - kJ/mol/nm

global hbar, meV_to_kJmol # declares global variables that can be used within any function
hbar = 0.0635077993       # reduced Planck's constant in units of kJ mol−1 ps
meV_to_kJmol = 0.09648530749925795 # conversion factor, number of kJ mol−1 in one meV


########## Functions

def debye_lorentz_spectral_density(lamda, w_c, w_arr):
    """
    Construct a Debye-Lorentz spectral density
    
    K(w) = 4 lambda / pi *  w_c/(w^2 + w_c^2)

    Parameters
    ----------
    lamda : Float
        Reorganization energy.
    w_c : Float
        Frequency cutoff (width of distribution).
    w_arr : Numpy array
        Array of frequencies values to evaluate K(w) on.

    Returns
    -------
    Kw : Numpy array
        Spectral Density.
    """
    Kw = (4 * lamda / np.pi) * w_c/(w_arr**2 + w_c**2)
    return Kw

def calculate_reorgE(Kw,w_arr):
    """
    Calculates total reorganization energy
    
    lambda = 0.5 * \int_0^\infty K(w) dw

    Parameters
    ----------
    Kw : Numpy array
        Spectral density.
    w_arr : Numpy array
        Frequencies.

    Returns
    -------
    lambda : Float
        Reorganization energy.
    """
    lamda = 0.5 * np.trapz(Kw,w_arr)
    return lamda
    
    
def calculate_system_bath_coupling(omegas,lamda):
    """
    Calculates system bath coupling for a given set of frequencies
    c^2(w) = w^2 K(w) / rho(w) = 2 * lambda * w^2/N

    Parameters
    ----------
    omegas : Numpy array
        Frequencies.
    lamda : Float
        Reorganization energy.

    Returns
    -------
    c_sq : Numpy array
        System-environment couplings.
    """
    c_sq = 2 * lamda * omegas**2 / len(omegas)
    return c_sq

    
def sample_distribution(x,pdf_x,N):
    """
    Generate a random sample for an arbitrary probability distribution of 1 variable.
    Uses a numerical inverse transform method

    Parameters
    ----------
    pdf_x : Numpy array.
        Probability density function evaluated at discrete points.
    x : Numpy array.
        Support of probability density function.
    N : Int.
        Number of random samples.

    Returns
    -------
    random_samples : Numpy Arrray
        Sampled points.

    """

    # calculate cumulative distribution function via trapezoid method
    cdf_x = np.zeros(len(x))
    cdf_x[1:] = np.cumsum( (pdf_x[0:-1] + pdf_x[1:])/2.0 * (x[1:]-x[0:-1]) )
    
    # normalize CDF
    cdf_x = cdf_x/cdf_x[-1]
    
    # sample using inverse CDF method
    values_cdf = np.random.rand(N)
    indices_x = np.searchsorted(cdf_x, values_cdf)
    random_samples = x[indices_x]
    
    return random_samples
    
def sample_cauchy_distribution(x_c,N):
    """
    Sample from a Cauchy distribution centered at origin

    Parameters
    ----------
    x_c : Float
        Parameter (width) of cauchy distribution.
    N : Int
        Number of samples.

    Returns
    -------
    random_samples : Numpy Arrray
        Sampled points.

    """
    values = np.random.rand(N)
    random_samples = x_c * np.tan(np.pi/2.0 * values)
    return random_samples
    
    
def tests():
    """
    Runs a series of tests to make sure program works correctly
    """
    
    ### Test 1: Check that the re-organization energy of Debye-Lorentz spectral density 
    ### matches theoretical value
    print("Test 1\n")
    lamda = 50 / meV_to_kJmol
    w_c = 170/ meV_to_kJmol / hbar
    w_arr = np.linspace(0,1000*w_c,50000)
    N = 500

    Kw = debye_lorentz_spectral_density(lamda,w_c,w_arr)
    lamda_calc = calculate_reorgE(Kw,w_arr)
    
    plt.figure()
    plt.plot(w_arr,Kw)
    plt.xlim(0,10*w_c)
    print("Input reorganization energy = %0.2f, Calculated reorganization energy = %0.2f"%(lamda,lamda_calc))
    print("-------------\n")
    
    ### Test 2: Check that integral of density of states gives N modes
    print("Test 2\n")
    
    rho = N * Kw / (2.0 * lamda_calc)
    N_calc = np.trapz(rho,w_arr)
    
    print("N = %0.2f, Calculated N = %0.2f"%(N,N_calc))
    print("-------------\n")
    
    ### Test 3: Check relationship between rho and c^2
    print("Test 3\n")
    print("Creates plot")
    
    c_sq_1 = 2 * lamda * w_arr**2 / N
    c_sq_2 = w_arr**2 * Kw / rho
    
    plt.figure()
    plt.plot(w_arr,c_sq_1, linewidth=2.0, alpha=0.5, label=r"$c^2(\omega)$ method 1")
    plt.plot(w_arr,c_sq_2, linewidth=2.0, alpha=0.5, label=r"$c^2(\omega)$ method 2")
    plt.legend()
    print("-------------\n")

    ### Test 4: Check distribution sampling
    samples_1 = sample_distribution(w_arr,rho/N,N)
    samples_2 = sample_cauchy_distribution(w_c,N)
    
    plt.figure()
    n1,bins1,_ = plt.hist(samples_1 ,bins=100  , density=True,alpha=0.5,label="Numerical sampler")
    n2,bins2,_    = plt.hist(samples_2, bins=bins1, density=True,alpha=0.5,label="Analytical sampler")
    plt.legend()
    plt.xlim(0,100*w_c)

if __name__ == "__main__":
    ########## Script Parameters
    
    N     = 300      # Number of modes to sample 
    out_name = "freqs.csv"
    
    # Parameters for specral density
    
    lamda = 50 * meV_to_kJmol       # Reorganization energy in kJ/mol
    w_c = 17 * meV_to_kJmol / hbar  # Cutoff frequency in ps-1
    w_arr = np.linspace(0,100*w_c,5000) # Discrete frequency points to construct Kw
    
    ########## Run Script
    
    Kw = debye_lorentz_spectral_density(lamda,w_c,w_arr)
    
    lamda_calc = calculate_reorgE(Kw,w_arr)            # calculate reorganization energy
    
    rho =  N * Kw / (2.0 * lamda_calc)                 # calculate density of modes
    
    sampled_freqs = sample_distribution(w_arr,rho/N,N) # sample frequencies
    
    sampled_c_sq = calculate_system_bath_coupling(sampled_freqs,lamda) # sampled c^2
    
    sampled_x_e = 2 * lamda_calc / np.sqrt(sampled_c_sq) # sampled excited state displacements
    

    ########## Plot and output data
    plt.figure()
    plt.plot(w_arr,rho/N, label="original distribution")
    plt.hist(sampled_freqs,bins=100,density=True, label="sampled distribution")
    plt.xlim(0,20*w_c)
    plt.xlabel(r"$\omega$ ps$^{-1}$")
    plt.ylabel(r"$\rho(\omega)$")
    plt.legend()
    
    plt.figure()
    plt.plot( np.sort(sampled_freqs), np.sqrt(sampled_c_sq[np.argsort(sampled_freqs)] ) )
    plt.xlim(-5,20*w_c)
    plt.xlabel(r"$\omega$ ps$^{-1}$")
    plt.ylabel(r"$c(\omega)$")

    plt.figure()
    plt.plot( np.sort(sampled_freqs), sampled_x_e[np.argsort(sampled_freqs)] )
    plt.xlim(-5,20*w_c)
    plt.xlabel(r"$\omega$ ps$^{-1}$")
    plt.ylabel(r"$x_e$")
    
    file = open(out_name,'w')
    file.write("Sampled frequencies for harmonic oscillator bath of a Debye-Lorentz Spectral Density\n")
    file.write("Parameters, lambda=%0.1f kJ/mol, omega_c=%0.1f kJ/mol, mass=1 g/mol\n"%(lamda,w_c))
    file.write("n, omega_n - (1/ps), c(omega_n), x_e(omega_n) - (nm)\n")
    for i in range(N):
        file.write("%d,%0.1f,%0.1f,%0.3f\n"%(
            i+1,sampled_freqs[i],np.sqrt(sampled_c_sq[i]),sampled_x_e[i]))
    file.close()
