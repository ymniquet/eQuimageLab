# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Histogram stretch functions."""

import numpy as np

def shadow_stretch_function(x, shadow):
  """Return the linear shadow stretch function f(x).

  The input data x is clipped below shadow and linearly stretched
  to map [shadow, 1] onto [0, 1].
  The output, stretched data therefore fit in the [0, infty[ range.

  Args:
    x (numpy.ndarray): The input data.
    shadow (float): The shadow level (expected < 1).

  Returns:
    numpy.ndarray: The stretched data.
  """
  x = np.clip(x, shadow, None)
  return (x-shadow)/(1.-shadow)

def shadow_highlight_stretch_function(x, shadow, highlight):
  """Return the linear shadow/highlight stretch function f(x).

  The input data x is clipped below shadow and above highlight and linearly stretched
  to map [shadow, highlight] onto [0, 1].
  The output, stretched data therefore fit in the [0, 1] range.

  Args:
    x (numpy.ndarray): The input data.
    shadow (float): The shadow level (expected < 1).
    highlight (float): The highlight level (expected > shadow).

  Returns:
    numpy.ndarray: The stretched data.
  """
  x = np.clip(x, shadow, highlight)
  return (x-shadow)/(highlight-shadow)

def dynamic_range_stretch_function(x, fr, to):
  """Return the linear dynamic range stretch function f(x).

  The input data x is linearly stretched to map [fr[0], fr[1]] onto [to[0], to[1]],
  then clipped in the [to[0], to[1]] range.

  Args:
    x (numpy.ndarray): The input data.
    fr (a tuple or list of two floats): The input range.
    to (a tuple or list of two floats): The output range.

  Returns:
    numpy.ndarray: The stretched data.
  """
  return np.interp(x, fr, to)

def asinh_stretch_function(x, stretch):
  """Return the arcsinh stretch function f(x).

  The arcsinh stretch function is defined as:

    f(x) = arcsinh(stretch*x)/arcsinh(stretch)

  Args:
    x (numpy.ndarray): The input data.
    stretch (float): The stretch factor (expected >= 0).

  Returns:
    numpy.ndarray: The stretched data.
  """
  return np.arcsinh(stretch*x)/np.arcsinh(stretch) if abs(stretch) > 1.e-6 else x

def ghyperbolic_stretch_function(x, logD1, b, SYP, SPP, HPP, inverse):
  """Return the generalized hyperbolic stretch function f(x).

  For details about generalized hyperbolic stretches, see: https://ghsastro.co.uk/.
  This function clips the input data x in the [0, 1] range before stretching.

  Note:
  Code adapted from https://github.com/mikec1485/GHS/blob/main/src/scripts/GeneralisedHyperbolicStretch/lib/GHSStretch.js.

  Todo:
  Do not clip the input data and extend the transformation outside [0, 1] ?

  Args:
    x (numpy.ndarray): The input data.
    logD1 (float): The global stretch factor ln(D+1) (expected >= 0).
    b (float): The local stretch factor.
    SYP (float): The symmetry point (expected in [0, 1]).
    SPP (float): The shadow protection point (expected in [0, SYP]).
    HPP (float): The highlight protection point (expected in [SYP, 1]).
    inverse (bool): Return the inverse stretch function if True.

  Returns:
    numpy.ndarray: The stretched data.
  """
  D = np.exp(logD1)-1.
  x = np.clip(x, 0., 1.)
  if abs(D) < 1.e-6: # Identity.
    return x
  else:
    y = np.empty_like(x)
    if abs(b) < 1.e-6:
      qs = np.exp(-D*(SYP-SPP))
      q0 = qs-D*SPP*np.exp(-D*(SYP-SPP))
      qh = 2.-np.exp(-D*(HPP-SYP))
      q1 = qh+D*(1.-HPP)*np.exp(-D*(HPP-SYP))
      q  = 1./(q1-q0)
      # Coefficient for x < SPP.
      b1 = D*np.exp(-D*(SYP-SPP))*q
      # Coefficients for SPP <= x < SYP.
      a2 = -q0*q
      b2 = q
      c2 = -D*SYP
      d2 = D
      # Coefficients for SYP <= x < HPP.
      a3 = (2.-q0)*q
      b3 = -q
      c3 = D*SYP
      d3 = -D
      # Coefficients for x >= HPP.
      a4 = (qh-q0-D*HPP*np.exp(-D*(HPP-SYP)))*q
      b4 = D*np.exp(-D*(HPP-SYP))*q
      # GHS transformation.
      if not inverse:
        mask = (x <  SPP)
        y[mask] =    b1*x[mask]
        mask = (x >= SPP) & (x < SYP)
        y[mask] = a2+b2*np.exp(c2+d2*x[mask])
        mask = (x >= SYP) & (x < HPP)
        y[mask] = a3+b3*np.exp(c3+d3*x[mask])
        mask = (x >= HPP)
        y[mask] = a4+b4*x[mask]
      else:
        SPT =    b1*SPP
        SYT = a2+b2*np.exp(c2+d2*SYP)
        HPT = a4+b4*HPP
        mask = (x <  SPT)
        y[mask] = x[mask]/b1
        mask = (x >= SPT) & (x < SYT)
        y[mask] = (np.log((x[mask]-a2)/b2)-c2)/d2
        mask = (x >= SYT) & (x < HPT)
        y[mask] = (np.log((x[mask]-a3)/b3)-c3)/d3
        mask = (x >= HPT)
        y[mask] = (x[mask]-a4)/b4
    elif abs(b+1.) < 1.e-6:
      qs = -np.log(1.+D*(SYP-SPP))
      q0 = qs-D*SPP/(1.+D*(SYP-SPP))
      qh = np.log(1.+D*(HPP-SYP))
      q1 = qh+D*(1.-HPP)/(1.+D*(HPP-SYP))
      q  = 1./(q1-q0)
      # Coefficient for x < SPP.
      b1 = D/(1.+D*(SYP-SPP))*q
      # Coefficients for SPP <= x < SYP.
      a2 = -q0*q
      b2 = -q
      c2 = 1.+D*SYP
      d2 = -D
      # Coefficients for SYP <= x < HPP.
      a3 = -q0*q
      b3 = q
      c3 = 1.-D*SYP
      d3 = D
      # Coefficients for x >= HPP.
      a4 = (qh-q0-D*HPP/(1.+D*(HPP-SYP)))*q
      b4 = q*D/(1.+D*(HPP-SYP))
      # GHS transformation.
      if not inverse:
        mask = (x <  SPP)
        y[mask] =    b1*x[mask]
        mask = (x >= SPP) & (x < SYP)
        y[mask] = a2+b2*np.log(c2+d2*x[mask])
        mask = (x >= SYP) & (x < HPP)
        y[mask] = a3+b3*np.log(c3+d3*x[mask])
        mask = (x >= HPP)
        y[mask] = a4+b4*x[mask]
      else:
        SPT =    b1*SPP
        SYT = a2+b2*np.log(c2+d2*SYP)
        HPT = a4+b4*HPP
        mask = (x <  SPT)
        y[mask] = x[mask]/b1
        mask = (x >= SPT) & (x < SYT)
        y[mask] = (np.exp((x[mask]-a2)/b2)-c2)/d2
        mask = (x >= SYT) & (x < HPT)
        y[mask] = (np.exp((x[mask]-a3)/b3)-c3)/d3
        mask = (x >= HPT)
        y[mask] = (x[mask]-a4)/b4
    else:
      if b < 0.:
        b  = -b
        qs = (1.-(1.+D*b*(SYP-SPP))**((b-1.)/b))/(b-1.)
        q0 = qs-D*SPP*(1.+D*b*(SYP-SPP))**(-1./b)
        qh = ((1.+D*b*(HPP-SYP))**((b-1.)/b)-1.)/(b-1.)
        q1 = qh+D*(1.-HPP)*(1.+D*b*(HPP-SYP))**(-1./b)
        q  = 1./(q1-q0)
        # Coefficient for x < SPP.
        b1 = D*(1.+D*b*(SYP-SPP))**(-1./b)*q
        # Coefficients for SPP <= x < SYP.
        a2 = (1./(b-1.)-q0)*q
        b2 = -q/(b-1.)
        c2 = 1.+D*b*SYP
        d2 = -D*b
        e2 = (b-1.)/b
        # Coefficients for SYP <= x < HPP.
        a3 = (-1./(b-1.)-q0)*q
        b3 = q/(b-1.)
        c3 = 1.-D*b*SYP
        d3 = D*b
        e3 = (b-1.)/b
        # Coefficients for x >= HPP.
        a4 = (qh-q0-D*HPP*(1.+D*b*(HPP-SYP))**(-1./b))*q
        b4 = D*(1.+D*b*(HPP-SYP))**(-1./b)*q
      else:
        qs = (1.+D*b*(SYP-SPP))**(-1./b)
        q0 = qs-D*SPP*(1.+D*b*(SYP-SPP))**(-(1.+b)/b)
        qh = 2.-(1.+D*b*(HPP-SYP))**(-1./b)
        q1 = qh+D*(1.-HPP)*(1.+D*b*(HPP-SYP))**(-(1.+b)/b)
        q  = 1./(q1-q0)
        # Coefficient for x < SPP.
        b1 = D*(1.+D*b*(SYP-SPP))**(-(1.+b)/b)*q
        # Coefficients for SPP <= x < SYP.
        a2 = -q0*q
        b2 = q
        c2 = 1.+D*b*SYP
        d2 = -D*b
        e2 = -1./b
        # Coefficients for SYP <= x < HPP.
        a3 = (2.-q0)*q
        b3 = -q
        c3 = 1.-D*b*SYP
        d3 = D*b
        e3 = -1./b
        # Coefficients for x >= HPP.
        a4 = (qh-q0-D*HPP*(1.+D*b*(HPP-SYP))**(-(b+1.)/b))*q
        b4 = D*(1.+D*b*(HPP-SYP))**(-(b+1.)/b)*q
      # GHS transformation.
      if not inverse:
        mask = (x <  SPP)
        y[mask] =    b1*x[mask]
        mask = (x >= SPP) & (x < SYP)
        y[mask] = a2+b2*(c2+d2*x[mask])**e2
        mask = (x >= SYP) & (x < HPP)
        y[mask] = a3+b3*(c3+d3*x[mask])**e3
        mask = (x >= HPP)
        y[mask] = a4+b4*x[mask]
      else:
        SPT =    b1*SPP
        SYT = a2+b2*(c2+d2*SYP)**e2
        HPT = a4+b4*HPP
        mask = (x <  SPT)
        y[mask] = x[mask]/b1
        mask = (x >= SPT) & (x < SYT)
        y[mask] = (((x[mask]-a2)/b2)**(1./e2)-c2)/d2
        mask = (x >= SYT) & (x < HPT)
        y[mask] = (((x[mask]-a3)/b3)**(1./e3)-c3)/d3
        mask = (x >= HPT)
        y[mask] = (x[mask]-a4)/b4
    return y

def midtone_stretch_function(x, midtone, inverse):
  """Return the midtone stretch function f(x).

  The midtone stretch function is defined as:

    f(x) = (midtone-1)*x/((2*midtone-1)*x-midtone)

  In particular, f(0) = 0, f(midtone) = 0.5 and f(1) = 1.

  Args:
    x (numpy.ndarray): The input data.
    midtone (float): The midtone level (expected in ]0, 1[]).
    inverse (bool): Return the inverse stretch function if True.

  Returns:
    numpy.ndarray: The stretched data.
  """
  return midtone*x/((2.*midtone-1.)*x-midtone+1.) if inverse else (midtone-1.)*x/((2.*midtone-1.)*x-midtone)

def gamma_stretch_function(x, gamma):
  """Return the gamma stretch function f(x).

  The gamma stretch function is defined as:

    f(x) = x**gamma

  This function clips the input data x below 0 before stretching.

  Args:
    x (numpy.ndarray): The input data.
    gamma (float): The stretch exponent (expected > 0).

  Returns:
    numpy.ndarray: The stretched data.
  """
  x = np.clip(x, 0., None)
  return x**gamma

def midtone_levels_stretch_function(x, shadow, midtone, highlight, low, high):
  """Return the shadow/midtone/highlight/low/high levels adjustment function f(x).

  This function:

    1) Clips the input data in the [shadow, highlight] range and maps [shadow, highlight] to [0, 1].
    2) Applies the midtone stretch function f(x) = (m-1)*x/((2*m-1)*x-m),
       with m = (midtone-shadow)/(highlight-shadow) the remapped midtone.
    3) Maps [low, high] to [0, 1] and clips the output data in the [0, 1] range.

  Args:
    x (numpy.ndarray): The input data.
    midtone (float): The input midtone level (expected in ]0, 1[).
    shadow (float): The input shadow level (expected < midtone).
    highlight (float): The input highlight level (expected > midtone).
    low (float): The "low" output level (expected <= 0).
    high (float): The "high" output level (expected >= 1).

  Returns:
    numpy.ndarray: The stretched data.
  """
  midtone = (midtone-shadow)/(highlight-shadow)
  x = np.clip(x, shadow, highlight)
  x = (x-shadow)/(highlight-shadow)
  y = (midtone-1.)*x/((2.*midtone-1.)*x-midtone)
  return np.interp(y, (low, high), (0., 1.))
