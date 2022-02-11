==============================
Kalman Filter
==============================

:Aughor: Xuemei Wang
:Date: 2021-10-15

.. Contents::
   :local:

Introduction
---------------

Kalman Filtering is an algorithm that provides estimates of some unknown variables given the measurements observed over time.

In 1960, R.E.Kalman published his famous paper describing a recursive solution to the discrete-data linear filtering problem.

The Kalman filter is a set of mathematical equations that provides an efficient computational (recursive) means to estimate the state of a process,
in a way that imnimizes the mean of the squared error.

The Kalman filter has long been regarded as the optimal solution to many tracking and data prediction tasks.
Its use in the analysis of visual motion has been documented frequently.
The filger is constructed as a mean squared error minimiser, and an alternative derivation of the filter can be shown how the filter relates to maximum likelihood statistics.
The purpose of filtering is to extract the required information from a signal, ignoring everything else.

Abstractly, Kalman filtering can be seen as a particular approach to combining approximations of an unknown value to produce a better approximation.

Kalman filtering is a classic state estimation technique used in application areas such as signal processing and autonomous control of vehicles.
It is now being used to solve problems in computer systems such as controlling the voltage and frequency of processors.

Kalman  filtering is a state estimation technique inverted in 1960 by Rudolf E. Kalman. Because of its ability to extract useful information from noisy data and its small computational and memory requirements, it is used in many application areas including spacecraft navigation, motion planning in robotics, signal processing, and wireless sensor networks.
Recernt work has used Kalman filtering in controllers for computer systems.


Descriptions
---------------

The confidence in a device is modeled formally by the variance of the distribution associated with that device;
The smaller the variance, the higher our confidence in the measurements made by the device.



Assume that we want to know the value of a variable within a process of the form


Kalman filters are used to estimate states based on linear dynamical systems in state space format.
The process model defines the evolution of the state from time :math:`k -1` to time :math:`k` as

The system that is considered is composed of two essential ingredients.
First, the state is assumed to be described by



.. math::
   :label: eq:pfx

   x_{k} = F x_{k -1} + B u_{k -1} + w_{k -1}

and the measurement data are related to the state by

.. math::
   z_{k} = H x_{k} + v_{k}

Prediction:

.. math::

        &\hat{x}_k^- = F \hat{x}_{k-1}^+ + B u_{k -1} \\
        &P_k^- = F P_{k-1}^+ F^T + Q

Update:

.. math::

        &\tilde{y}_k = z_k - H \hat{x}_k^- \\
        &K_k = P_k^- H^T (R + H P_k^- H^T)^{-1} \\
        &\hat{x}_k^+ = \hat{x}_k^- + K_y \tilde{y} \\
        &P_k^+ = (I - K_k H)P_k^- \\
