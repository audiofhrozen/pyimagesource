# Copyright 2018 Waseda University (Nelson Yalta)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from scipy.optimize import fminbound
from scipy.special import expn


# ======== Sabine's formula
def sabine(a, rt, room, weight):
    alpha = a * weight
    V = np.prod(room)               # Room volume
    Sx = room[0, 1] * room[0, 2]    # Wall surface X
    Sy = room[0, 0] * room[0, 2]    # Wall surface Y
    Sz = room[0, 0] * room[0, 1]    # Wall surface Z
    A = Sx * (alpha[0, 0] + alpha[0, 1]) + Sy * (
        alpha[0, 2] + alpha[0, 3]) + Sz * (alpha[0, 4] + alpha[0, 5])
    err = np.abs(rt - 0.161 * V / A)
    return err


# ======== Millington-Sette's formula
def millington_sette(a, rt, room, weight):
    alpha = a * weight
    V = np.prod(room)              # Room volume
    Sx = room[0, 1] * room[0, 2]   # Wall surface X
    Sy = room[0, 0] * room[0, 2]   # Wall surface Y
    Sz = room[0, 0] * room[0, 1]   # Wall surface Z
    A = -(Sx * (np.log(1 - alpha[0, 0]) + np.log(1 - alpha[0, 1])) + Sy * (
        np.log(1 - alpha[0, 2]) + np.log(1 - alpha[0, 3])) + Sz * (
        np.log(1 - alpha[0, 4]) + np.log(1 - alpha[0, 5])))
    err = np.abs(rt - 0.161 * V / A)
    return err


# %======== Norris and Eyring's formula
def norris_eyring(a, rt, room, weight):
    alpha = a * weight
    V = np.prod(room)              # Room volume
    Sx = room[0, 1] * room[0, 2]   # Wall surface X
    Sy = room[0, 0] * room[0, 2]   # Wall surface Y
    Sz = room[0, 0] * room[0, 1]   # Wall surface Z
    St = 2 * Sx + 2 * Sy + 2 * Sz  # Total wall surface
    A = Sx * (alpha[0, 0] + alpha[0, 1]) + Sy * (
        alpha[0, 2] + alpha[0, 3]) + Sz * (alpha[0, 4] + alpha[0, 5])
    am = 1 / St * A
    err = np.abs(rt + 0.161 * V / (St * np.log(1 - am)))
    return err


# ======== Fitzroy's approximation
def fitzroy(a, rt, room, weight):
    raise Exception('WIP, not working')
    alpha = a * weight
    V = np.prod(room)              # Room volume
    Sx = room[0, 1] * room[0, 2]   # Wall surface X
    Sy = room[0, 0] * room[0, 2]   # Wall surface Y
    Sz = room[0, 0] * room[0, 1]   # Wall surface Z
    St = 2 * Sx + 2 * Sy + 2 * Sz  # Total wall surface
    tx = -2 * Sx / np.log(1 - np.mean(alpha[0:2]))
    ty = -2 * Sy / np.log(1 - np.mean(alpha[2:4]))
    tz = -2 * Sz / np.log(1 - np.mean(alpha[4:6]))
    err = abs(rt - 0.161 * V / (St ** 2) * (tx + ty + tz))
    return err


# ======== Arau's formula
def arau(a, rt, room, weight):
    raise Exception('WIP, not working')
    alpha = a * weight
    V = np.prod(room)              # Room volume
    Sx = room[0, 1] * room[0, 2]   # Wall surface X
    Sy = room[0, 0] * room[0, 2]   # Wall surface Y
    Sz = room[0, 0] * room[0, 1]   # Wall surface Z
    St = 2 * Sx + 2 * Sy + 2 * Sz            # Total wall surface
    Tx = (0.161 * V / (-St * np.log(1 - np.mean(alpha[0:2])))) ** (2 * Sx / St)
    Ty = (0.161 * V / (-St * np.log(1 - np.mean(alpha[2:4])))) ** (2 * Sy / St)
    Tz = (0.161 * V / (-St * np.log(1 - np.mean(alpha[4:6])))) ** (2 * Sz / St)
    err = abs(rt - (Tx * Ty * Tz))
    return err


# ======== Neubauer and Kostek's formula
def neubauer_kostek(a, rt, room, weight):
    raise Exception('WIP, not working')
    V = np.prod(room)              # Room volume
    Sx = room[0, 1] * room[0, 2]   # Wall surface X
    Sy = room[0, 0] * room[0, 2]   # Wall surface Y
    Sz = room[0, 0] * room[0, 1]   # Wall surface Z
    St = 2 * Sx + 2 * Sy + 2 * Sz   # Total wall surface
    r = 1 - a * weight
    rww = np.mean(r[0:4])
    rcf = np.mean(r[4:6])
    rb = np.mean(r)
    aww = np.log(1 / rb) + (r(1) * (r(1) - rww) * (Sx ** 2) + r(2) * (r(2) - rww) * Sx ** 2 + r(3) * (
        r(3) - rww) * Sy ^ 2 + r(4) * (r(4) - rww) * Sy ^ 2) / ((rww * (2 * Sx + 2 * Sy)) ** 2)
    acf = np.log(1 / rb) + (r(5) * (r(5) - rcf) * Sz ^ 2 + r(6) * (r(6) - rcf) * Sz ^ 2) / ((rcf * 2 * Sz) ^ 2)
    err = abs(rt - 0.32 * V / (St ^ 2) * (room(3) * (room(1) + room(2)) / aww + room(1) * room(2) / acf))
    return err


def ISM_RIRpow_approx(aa, room, cc, timepts, rt_type=None, rt_val=None):
    """ISM_RIRpow_approx  Approximation of ISM RIR power (Lehmann & Johansson's method)

     [P_VEC,T_VEC,OK_FLAG] = ISM_RIRpow_approx(ALPHA,ROOM,C,T_VEC,RT_TYPE,RT_VAL)

     This function returns the predicted values of RIR power in P_VEC (as
     would result from ISM simulations) estimated by means of the EDC
     approximation method described in: "Prediction of energy decay in room
     impulse responses simulated with an image-source model", J. Acoust. Soc.
     Am., vol. 124(1), pp. 269-277, July 2008. The values of P_VEC are
     computed for the time points given as input in T_VEC (in sec), which is
     assumed to contain increasing values of time. The vector T_VEC (and
     corresponding vector P_VEC) will be cropped if the numerical computation
     limits are reached for the higher time values in T_VEC (for which NaNs
     are generated in P_VEC), in which case the output parameter OK_FLAG will
     be set to 0 (1 otherwise).

     The environmental setting is defined via the following input parameters:

        ALPHA: 1-by-6 vector, corresponding to each wall's absorption
               coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall closest
               to the origin. E.g.: [0.5 0.5 0.45 0.87 0.84 0.32].
      RT_TYPE: character string, measure of reverberation time used for the
               definition of the coefficients in ALPHA. Set to either 'T60' or
               'T20'.
       RT_VAL: scalar, value of the reverberation time (in seconds) defined by
               RT_TYPE. E.g.: 0.25.
         ROOM: 1-by-3 vector, indicating the rectangular room dimensions
               (in m): [x_length y_length z_length]. E.g.: [4 4 3].
            C: scalar (in m/s), propagation speed of sound waves. E.g.: 343.
    """

    eps = np.finfo(float).eps
    numradpts = len(timepts)
    radpts = cc * timepts              # radius values corresponding to time points

    bxx = (np.sqrt(1. - aa[0, 0]) * np.sqrt(1. - aa[0, 1])) ** (1. / room[0, 0])
    byy = (np.sqrt(1. - aa[0, 2]) * np.sqrt(1. - aa[0, 3])) ** (1. / room[0, 1])
    bzz = (np.sqrt(1. - aa[0, 4]) * np.sqrt(1. - aa[0, 5])) ** (1. / room[0, 2])

    if bxx == byy and byy == bzz:
        intcase = 1
    elif bxx == byy and bxx != bzz:
        intcase = 2
    elif byy == bzz and bzz != bxx:
        if bzz < bxx:     # coordinate swap x<->z
            foo = bxx
            bxx = bzz
            bzz = foo
            intcase = 2
        else:
            intcase = 3
    elif bxx == bzz and bzz != byy:
        if bzz < byy:     # coordinate swap y<->z
            foo = byy
            byy = bzz
            bzz = foo
            intcase = 2
        else:
            intcase = 4
    else:
        intcase = 5
        if bxx > bzz and bxx > byy:     # coordinate swap x<->z
            foo = bxx
            bxx = bzz
            bzz = foo
        elif byy > bzz and byy > bxx:  # coordinate swap y<->z
            foo = byy
            byy = bzz
            bzz = foo

    amppts1 = np.zeros((numradpts))
    for ss in range(numradpts):    # compute amplitude/energy estimates
        Bx = bxx ** radpts[ss]
        Bx = eps if Bx == 0 else Bx
        By = byy ** radpts[ss]
        By = eps if By == 0 else By
        Bz = bzz ** radpts[ss]
        Bz = eps if Bz == 0 else Bz
        if intcase == 1:
            int2 = Bx
        elif intcase == 2:
            int2 = (Bx - Bz) / np.log(Bx / Bz)
        elif intcase == 3:
            n1 = np.log(Bz / Bx)
            int2 = Bz * (expn(1, n1) + np.log(n1) + 0.5772156649) / n1
        elif intcase == 4:
            n1 = np.log(Bz / By)
            int2 = Bz * (expn(1, n1) + np.log(n1) + 0.5772156649) / n1
        else:
            n1 = np.log(Bz / By)
            n2 = np.log(Bz / Bx)
            int2 = Bz * (np.log(n1 / n2) + expn(1, n1) - expn(1, n2)) / np.log(Bx / By)
        amppts1[ss] = int2 / radpts[ss]      # 'propto' really...

    okflag = 1
    foo = np.where(np.isnan(amppts1))[0]
    if len(foo) > 0:
        amppts1 = amppts1[0:foo[0] - 1]
        timepts = timepts[0:foo[0] - 1]
        okflag = 0

    if rt_type is not None:
        if rt_type == 60:
            sl = np.exp(3.05 * np.exp(-1.85 * rt_val))
        elif rt_type == 20:
            sl = np.exp(3.52 * np.exp(-7.49 * rt_val))
        else:
            raise ValueError('Incorrect type of rt_type')
        amppts1 = amppts1 / np.exp(sl * (timepts - timepts[0]))
    return amppts1, timepts, okflag


# ======== Lehmann & Johannson's EDC approximation method
def lehmann_johansson_60(a, t60, room, weight, cc):
    starttime = 1.4 * np.mean(room) / cc        # start time t0
    DPtime = np.mean(room) / cc                 # direct path "estimate"
    aa = a * weight

    numradpts = 60
    stoptime = 2 * t60
    while True:  # loop to determine appropriate stop time
        timepts = np.linspace(starttime, stoptime, numradpts)  # time points where to compute data
        amppts1, timepts, okflag = ISM_RIRpow_approx(aa, room, cc, timepts)
        for ii in range(amppts1.shape[0]):
            amppts1[ii] = np.sum(amppts1[ii:])
        amppts1 = 10 * np.log10(amppts1 / amppts1[0])

        if amppts1[-1] >= -60:
            if okflag == 0:
                raise ValueError('Problem computing EDC approximation!')
            numradpts = numradpts + 30     # more points are required for accurate T60 estimate
            stoptime = stoptime + t60
            continue
        sind = np.where(amppts1 >= -60)[0][-1]
        deltaX = timepts[1] - timepts[0]
        deltaY = amppts1[sind + 1] - amppts1[sind]
        deltaA = -60 - amppts1[sind]
        t2 = timepts[sind] + deltaA * deltaX / deltaY
        if t2 > (stoptime * 2 / 3):
            numradpts = numradpts + 30     # more points are required for accurate T60 estimate
            stoptime = stoptime + t60
            if okflag == 0:
                break   # use current time point if numerical limit is reached
            continue
        else:
            break

    t60est = t2 - DPtime
    err = np.abs(t60 - t60est)
    return err


# %======== Lehmann & Johannson's EDC approximation method
def lehmann_johansson_20(a, t20, room, weight, cc):
    starttime = 1.4 * np.mean(room) / cc        # start time t0
    aa = a * weight
    numradpts = 40
    stoptime = 5 * t20
    while True:  # loop to determine appropriate stop time
        timepts = np.linspace(starttime, stoptime, numradpts)  # time points where to compute data

        amppts1, timepts, okflag = ISM_RIRpow_approx(aa, room, cc, timepts)

        for ii in range(len(amppts1)):
            amppts1[ii] = np.sum(amppts1[ii:])
        amppts1 = 10 * np.log10(amppts1 / amppts1[0])

        if amppts1[-1] >= -25:
            if okflag == 0:
                raise ValueError('Problem computing EDC approximation!')
            numradpts = numradpts + 30     # more points are required for accurate T20 estimate
            stoptime = stoptime + 3 * t20
            continue
        sind = np.where(amppts1 >= -5)[0][-1]
        deltaX = timepts[1] - timepts[0]
        deltaY = amppts1[sind + 1] - amppts1[sind]
        deltaA = -5 - amppts1[sind]
        t1 = timepts[sind] + deltaA * deltaX / deltaY
        sind = np.where(amppts1 >= -25)[0][-1]
        deltaY = amppts1[sind + 1] - amppts1[sind]
        deltaA = -25 - amppts1[sind]
        t2 = timepts[sind] + deltaA * deltaX / deltaY

        if t2 > stoptime * 2 / 3:
            numradpts = numradpts + 30     # more points are required for accurate T20 estimate
            stoptime = stoptime + 3 * t20
            if okflag == 0:
                break   # use current time point if numerical limit is reached
            continue
        else:
            break

    t20est = t2 - t1
    err = np.abs(t20 - t20est)
    return err


def AbsCoeff(rttype, rt, room, weight, method, c=None, xtol=1e-05):
    """function [out,OKflag] = ISM_AbsCoeff(rttype,rt,room,weight,method,varargin)

    ISM_AbsCoeff  Calculates absorption coefficients for a given reverberation time

     [ALPHA,OKFLAG] = ISM_AbsCoeff(RT_TYPE,RT_VAL,ROOM,ABS_WEIGHT,METHOD)
     [ALPHA,OKFLAG] = ISM_AbsCoeff( ... ,'c',SOUND_SPEED_VAL)

     Returns the six absorption coefficients in the vector ALPHA for a given
     vector of room dimensions ROOM and a given value RT_VAL of reverberation
     time, with RT_TYPE corresponding to the desired measure of reverberation
     time, i.e., either 'T60' or 'T20'. Calling this function with RT_VAL=0
     simply returns ALPHA=[1 1 1 1 1 1] (anechoic case), regardless of the
     settings of the other input parameters.

     The parameter ABS_WEIGHTS is a 6 element vector of absorption
     coefficients weights which adjust the relative amplitude ratios between
     the six absorption coefficients in the resulting ALPHA vector. This
     allows the simulation of materials with different absorption levels on
     the room boundaries. Leave empty or set ABS_WEIGHTS=ones(1,6) to obtain
     uniform absorption coefficients for all room boundaries.

     If the desired reverberation time could not be reached with the desired
     environmental setup (i.e., practically impossible reverberation time
     value given ROOM and ABS_WEIGHTS), the function will issue a warning on
     screen accordingly. If the function is used with two output arguments,
     the on-screen warnings are disabled and the function sets the flag OKFLAG
     to 0 instead (OKFLAG is set to 1 if the computations are successful).

     The returned coefficients are calculated using one of the following
     methods, defined by the METHOD parameter:

        * Lehmann and Johansson  (METHOD='LehmannJohansson')
        * Sabine                 (METHOD='Sabine')
        * Norris and Eyring      (METHOD='NorrisEyring')
        * Millington-Sette       (METHOD='MillingtonSette')
        * Fitzroy                (METHOD='Fitzroy')
        * Arau                   (METHOD='Arau')
        * Neubauer and Kostek    (METHOD='NeubauerKostek')

     In case the first computation method is selected (i.e., if METHOD is set
     to 'LehmannJohansson'), this function also accepts an additional
     (optional) argument 'c', which will set the value of the sound wave
     propagation speed to SOUND_SPEED_VAL. If omitted, 'c' will default to 343
     m/s. This parameter has no influence on the other six computation
     methods.

     Lehmann & Johansson's method relies on a numerical estimation of the
     energy decay in the considered environment, which leads to accurate RT
     prediction results. For more detail, see: "Prediction of energy decay in
     room impulse responses simulated with an image-source model", J. Acoust.
     Soc. Am., vol. 124(1), pp. 269-277, July 2008. The definition of T20 used
     with the 'LehmannJohansson' method corresponds to the time required by
     the energy--time curve to decay from -5 to -25dB, whereas the definition
     of T60 corresponds to the time required by the energy--time curve to
     decay by 60dB from the time lag of the direct path in the transfer
     function.

     On the other hand, the last six calculation methods are based on various
     established equations that attempt to predict the physical reverberation
     time T60 resulting from given environmental factors. These methods are
     known to provide relatively inaccurate results. If RT_TYPE='T20', the
     value of T20 for these methods then simply corresponds to T60/3 (linear
     energy decay assumption). For more information, see: "Measurement of
     Absorption Coefficients: Sabine and Random Incidence Absorption
     Coefficients" in the online room acoustics teaching material "AEOF3/AEOF4
     Acoustics of Enclosed Spaces" by Y.W. Lam, The University of Salford,
     1995, as well as the paper: "Prediction of the Reverberation Time in
     Rectangular Rooms with Non-Uniformly Distributed Sound Absorption" by R.
     Neubauer and B. Kostek, Archives of Acoustics, vol. 26(3), pp. 183-202,
     2001.

    """
    if c is None:
        c = 343.

    if rttype != 't60' and rttype != 't20':
        raise ValueError('Unrecognised ''RT_TYPE'' parameter (must be either ''T60'' or ''T20'').')

    if weight is None:
        weight = np.ones((1, 6))
    else:
        weight = weight / np.amax(weight)

    if rt == 0:
        out = np.ones(weight.shape)
        return out
    logging.info('Type of method selected: {}'.format(method))
    if method == 'Sabine':
        if rttype == 't20':
            rt = 3 * rt  # linear energy decay assumption
        out = fminbound(sabine, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'NorrisEyring':
        if rttype == 't20':
            rt = 3 * rt  # linear energy decay assumption
        out = fminbound(norris_eyring, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'MillingtonSette':
        if rttype == 't20':
            rt = 3 * rt  # linear energy decay assumption
        out = fminbound(millington_sette, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'Fitzroy':
        if rttype == 't20':
            rt = 3 * rt  # linear energy decay assumption
        out = fminbound(fitzroy, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'Arau':
        if rttype == 't20':
            rt = 3 * rt        # linear energy decay assumption
        out = fminbound(arau, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'NeubauerKostek':
        if rttype == 't20':
            rt = 3 * rt   # linear energy decay assumption
        out = fminbound(neubauer_kostek, 0.0001, 0.9999, [rt, room, weight], xtol=xtol)
    elif method == 'LehmannJohansson':
        if rttype == 't20':
            out = fminbound(lehmann_johansson_20, 0.0001, 0.9999, [rt, room, weight, c], xtol=xtol)
        else:
            out = fminbound(lehmann_johansson_60, 0.0001, 0.9999, [rt, room, weight, c], xtol=xtol)
    else:
        raise ValueError('Unrecognised ''METHOD'' parameter (see help for a list of accepted methods).')

    if out < 0.0001 + 3 * xtol:
        logging.warning("""Some absorption coefficients are close to the allowable limits (alpha->0). The \n
                        resulting reverberation time might end up lower than desired for the given environmental \n
                        setup. Try to relax some environmental constraints so that the desired reverberation time \n'
                        is physically achievable (e.g., by increasing the room volume, increasing the maximum gap \n'
                        between the absorption weights, or decreasing the desired RT value).""")
        raise ValueError('out of tolerance')
    elif out > 0.9999 - 3 * xtol:
        logging.warning("""Some absorption coefficients are close to the allowable limits (alpha->1). The \n
                        resulting reverberation time might end up higher than desired for the given environmental \n
                        setup. Try to relax some environmental constraints so that the desired reverberation time \n
                        is physically achievable (e.g., by reducing the room volume, reducing the maximum gap \n
                        between the absorption weights, or increasing the desired RT value).'""")
        raise ValueError('out of tolerance')
    out = weight * out
    return out
