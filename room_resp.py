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

from ism_rir.abscoeff import ISM_RIRpow_approx
import logging  # NOQA
import numpy as np


RIRvec = None
TimePoints = None


def ISM_RIR_DecayTime(delta_dB_vec, rt_type, rt_val, aa, room, X_src, X_rcv, Fs, cc=343.0):
    """ISM_RIR_DecayTime  RIR decay time using Lehmann & Johansson's EDC approximation method

     DT = ISM_RIR_DecayTime(DELTA_dB,RT_TYPE,RT_VAL,ALPHA,ROOM,SOURCE,SENSOR,Fs)
     DT = ISM_RIR_DecayTime(DELTA_dB,RT_TYPE,RT_VAL,ALPHA,ROOM,SOURCE,SENSOR,Fs,C)

     This function determines the time DT taken by the energy in a RIR to
     decay by DELTA_dB when using Lehmann & Johansson's image-source method
     implementation (see: "Prediction of energy decay in room impulse
     responses simulated with an image-source model", J. Acoust. Soc. Am.,
     vol. 124(1), pp. 269-277, July 2008). The parameter DELTA_dB can be
     defined as a vector of dB values, in which case this function returns DT
     as a vector containing the corresponding decay times. Note that DT does
     not necessarily correspond to the usual definition of quantities such as
     T20, T30, or T60. The resulting DT values are computed according to
     Lehmann and Johannson's EDC (energy decay curve) approximation method
     (see above reference) used in conjunction with a RIR reconstruction
     method based on diffuse reverberation modeling (see "Diffuse
     reverberation model for efficient image-source simulation of room
     impulse responses", IEEE Trans. Audio, Speech, Lang. Process., 2010).

     The environmental room setting is given via the following parameters:

           Fs: scalar, sampling frequency (in Hz). Eg: 8000.
        ALPHA: 1-by-6 vector, corresponding to each wall's absorption
               coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall
               closest to the origin. E.g.: [0.5 0.5 0.45 0.87 0.84 0.32].
      RT_TYPE: character string, measure of reverberation time used for the
               definition of the coefficients in ALPHA. Set to either 'T60' or
               'T20'.
       RT_VAL: scalar, value of the reverberation time (in seconds) defined by
               RT_TYPE. E.g.: 0.25.
       SOURCE: 1-by-3 vector, indicating the location of the source in space
               (in m): [x y z]. E.g.: [1 1 1.5].
       SENSOR: 1-by-3 vector, indicating the location of the microphone in
               space (in m): [x y z]. E.g.: [2 2 1.5].
         ROOM: 1-by-3 vector, indicating the rectangular room dimensions
               (in m): [x_length y_length z_length]. E.g.: [4 4 3].
            C: (optional) scalar (in m/s), propagation speed of sound waves.
               If omitted, C will default to 343m/s.
    """
    if isinstance(delta_dB_vec, float):
        delta_dB_vec = np.asarray([delta_dB_vec])
    delta_dB_vec = np.abs(delta_dB_vec)

    # -=:=- Check user input:
    if np.any(aa > 1) or np.any(aa < 0):
        raise ValueError('Parameter ''ALPHA'' must be in range (0...1].')
    elif np.any(delta_dB_vec == 0):
        raise ValueError('Parameter ''DELTA_dB'' must contain non-zero scalar values.')
    elif len(delta_dB_vec.shape) > 1:
        raise ValueError('Parameter ''DELTA_dB'' must be a 1-D vector (float).')

    if np.all(aa == 1) or rt_val == 0:
        raise ValueError('ISM_RIR_DecayTime cannot be used for anechoic environments.')

    if rt_type == 60:
            t60_appval = rt_val
    elif rt_type == 20:
            t60_appval = rt_val * 3       # coarse t60 estimate to determnine end time in EDCapprox computations
    else:
        raise ValueError('Unknown ''RT_TYPE'' argument.')

    # -=:=- Pre-processing -=:=-
    dp_del = np.linalg.norm(X_src - X_rcv) / cc          # direct path
    delta_dB_max = np.max(delta_dB_vec)
    n_ddbv = 1 if len(delta_dB_vec.shape) < 1 else len(delta_dB_vec.shape)
    starttime = np.amax([1.4 * np.mean(room) / cc, dp_del])    # start time t0, ensure >= dp_delay
    starttime_sind = np.floor(starttime * Fs)   # sample index
    RIR_start_DTime = 2 * starttime

    # -=:=- select window size -=:=-
    n_win_meas = 6                          # approximate nr of (useful) measurements
    TT = (RIR_start_DTime - starttime) / n_win_meas
    w_len = np.floor(TT * Fs)                   # current window length (samples)
    w_len = w_len + (w_len % 2) - 1       # make w_len odd
    w_len_half = int(np.floor(w_len / 2))

    # -=:=- pre-compute start of RIR for lambda correction -=:=-
    RIR = ISM_RoomResp(Fs, np.sqrt(1 - aa), rt_type, rt_val, X_src, X_rcv, room, MaxDelay=RIR_start_DTime, c=cc)
    RIRlen = len(RIR)

    # -=:=- Measure average energy -=:=-
    fit_time_perc = 0.35
    we_sind_vec = np.arange(starttime_sind + w_len_half, RIRlen, w_len).astype(np.int)     # window end indices
    wc_sind_vec = we_sind_vec - w_len_half                  # window centre indices
    wb_sind_vec = wc_sind_vec - w_len_half - 1              # window beginning indices
    if wb_sind_vec[0] <= 0:
        wb_sind_vec[0] = 1                                  # case where t0 is less than a half window
    n_win_meas = len(wc_sind_vec)
    en_vec_meas = np.zeros((n_win_meas,))
    for ww in range(n_win_meas):
        en_vec_meas[ww] = np.mean(RIR[wb_sind_vec[ww]:we_sind_vec[ww]] ** 2)
    t_vec_meas = wc_sind_vec / Fs
    fit_starttime = RIRlen * fit_time_perc / Fs
    fit_start_wind = np.where(t_vec_meas >= fit_starttime)[0][0]   # window index of start of fit

    # -=:=- Decay time estimate -=:=-
    DTime_vec = np.nan * np.ones(delta_dB_vec.shape)
    stind = 3
    while stind > 0:
        # compute + lambda-adjust EDC approximation
        # IMapprox computed up to several times what linear decay predicts
        stoptime = stind * delta_dB_max / 60 * t60_appval
        timepts = np.arange(starttime_sind, stoptime * Fs, w_len) / Fs
        stoptime = timepts[-1]
        # compute EDC approximation
        amppts1, timepts, okflag = ISM_RIRpow_approx(aa, room, cc, timepts, rt_type, rt_val)
        foo = en_vec_meas[fit_start_wind:] / amppts1[fit_start_wind:n_win_meas]      # offset compensation (lambda)
        amppts1 = amppts1 * np.mean(foo)

        # reconstruct approx. full RIR for proper EDC estimation (logistic noise approx.)
        amppts1_rec = np.interp(np.arange(RIRlen + 1, stoptime * Fs) / Fs, timepts, amppts1)
        RIR_rec = np.concatenate([RIR.T, np.sqrt(amppts1_rec)])
        RIR_rec_len = len(RIR_rec)

        # approx. full RIR EDC
        edc_rec = np.zeros((RIR_rec_len,))
        for nn in range(RIR_rec_len):
            edc_rec[nn] = np.sum(RIR_rec[nn:] ** 2)     # Energy decay using Schroeder's integration method
        edc_rec = 10 * np.log10(edc_rec / edc_rec[0])          # Decay curve in dB.
        tvec_rec = np.arange(RIR_rec_len) / Fs

        # Determine time of EDC reaching delta_dB decay:
        if edc_rec[-1] > -delta_dB_max:
            stind += 1
            if okflag == 0:
                raise ValueError('Problem computing decay time (parameter ''DELTA_dB'' may be too large)')
            if stind >= 25:
                raise ValueError('Problem computing decay time (parameter ''DELTA_dB'' may be too large)')
            continue

        for nn in range(n_ddbv):
            foo = np.where(edc_rec <= -delta_dB_vec[nn])[0][0]
            DTime_vec[nn] = 1.15 * tvec_rec[foo]            # statistical offset correction...

        # make sure IM approx was computed for more than 3/2 the resulting decay time
        if np.max(DTime_vec) > stoptime * 2 / 3:
            stind += 1                              # increase time if necessary
            if okflag == 0:
                raise ValueError('Problem computing decay time (parameter ''DELTA_dB'' may be too large)')
            if stind >= 25:
                raise ValueError('Problem computing decay time (parameter ''DELTA_dB'' may be too large)')
            continue
        else:
            stind = 0          # stop the computations
    return DTime_vec


def Check_nDim(a, b, d, l, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs):
    global RIRvec
    global TimePoints
    FoundNValBelowLim = 0
    n = 1                       # Check delay values for n=1 and above
    dist = np.linalg.norm([2 * a - 1, 2 * b - 1, 2 * d - 1] * X_src +
                          X_rcv - Rr * [n, l, m])
    foo_time = dist / c
    while foo_time <= MaxDelay:       # if delay is below TF length limit for n=1, check n=2,3,4...
        foo_amplitude = np.prod(beta ** np.abs([n - a, n, l - b, l, m - d, m])) / (4 * np.pi * dist)
        RIRvec = RIRvec + foo_amplitude * np.sinc((TimePoints - foo_time) * Fs)
        n += 1
        dist = np.linalg.norm([2 * a - 1, 2 * b - 1, 2 * d - 1] * X_src + X_rcv - Rr * [n, l, m])
        foo_time = dist / c
    if n != 1:
        FoundNValBelowLim = 1

    n = 0          # Check delay values for n=0 and below
    dist = np.linalg.norm([2 * a - 1, 2 * b - 1, 2 * d - 1] * X_src + X_rcv - Rr * [n, l, m])
    foo_time = dist / c
    while foo_time <= MaxDelay:    # if delay is below TF length for n=0, check n=-1,-2,-3...
        foo_amplitude = np.prod(beta ** np.abs([n - a, n, l - b, l, m - d, m])) / (4 * np.pi * dist)
        RIRvec = RIRvec + foo_amplitude * np.sinc((TimePoints - foo_time) * Fs)
        n -= 1
        dist = np.linalg.norm([2 * a - 1, 2 * b - 1, 2 * d - 1] * X_src + X_rcv - Rr * [n, l, m])
        foo_time = dist / c
    if n != 0:
        FoundNValBelowLim = 1
    return FoundNValBelowLim


def Check_lDim(a, b, d, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs):
    FoundLValBelowLim = 0
    l = 1                       # Check delay values for l=1 and above
    FoundNValBelowLim = Check_nDim(a, b, d, l, m, X_rcv, X_src,
                                   Rr, c, MaxDelay, beta, Fs)

    while FoundNValBelowLim == 1:
        l += 1
        FoundNValBelowLim = Check_nDim(a, b, d, l, m, X_rcv, X_src,
                                       Rr, c, MaxDelay, beta, Fs)

    if l != 1:
        FoundLValBelowLim = 1

    l = 0                       # Check delay values for l=0 and below
    FoundNValBelowLim = Check_nDim(a, b, d, l,
                                   m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)
    while FoundNValBelowLim == 1:
        l -= 1
        FoundNValBelowLim = Check_nDim(a, b, d, l,
                                       m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)

    if l != 0:
        FoundLValBelowLim = 1
    return FoundLValBelowLim


def ISM_RoomResp(Fs, beta, rt_type, rt_val, X_src, X_rcv, room, c=343., Delta_dB=50., MaxDelay=None):
    """ISM_RoomResp  RIR based on Lehmann & Johansson's image-source method

     RIR = ISM_RoomResp(Fs,BETA,RT_TYPE,RT_VAL,SOURCE,SENSOR,ROOM)
     RIR = ISM_RoomResp( ... ,'arg1',val1,'arg2',val2,...)

     This function generates the room impulse response (RIR) between a sound
     source and an acoustic sensor, based on various environmental parameters
     such as source and sensor positions, enclosure's dimensions and
     reflection coefficients, etc., according to Lehmann and Johansson's
     implementation of the image-source method (see below). The input
     parameters are defined as follows:

           Fs: scalar, sampling frequency (in Hz). Eg: 8000.
         BETA: 1-by-6 vector, corresponding to each wall's reflection
               coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall closest
               to the origin. This function assumes strictly non-negative BETA
               coefficients. Set to [0 0 0 0 0 0] to obtain anechoic response
               (direct path only), in which case the value of RT_VAL is
              discarded. E.g.: [0.75 0.75 0.8 0.25 0.3 0.9].
      RT_TYPE: character string, measure of reverberation time used for the
               definition of the coefficients in BETA. Set to either 'T60' or
               'T20'.
       RT_VAL: scalar, value of the reverberation time (in seconds) defined by
               RT_TYPE. Set to 0 to obtain anechoic response (same effect as
               setting BETA to [0 0 0 0 0 0]), in which case the BETA
               coefficients are discarded. E.g.: 0.25.
       SOURCE: 1-by-3 vector, indicating the location of the source in space
               (in m): [x y z]. E.g.: [1 1 1.5].
       SENSOR: 1-by-3 vector, indicating the location of the microphone in
               space (in m): [x y z]. E.g.: [2 2 1.5].
         ROOM: 1-by-3 vector, indicating the rectangular room dimensions
               (in m): [x_length y_length z_length]. E.g.: [4 4 3].

     In addition, a number of other (optional) parameters can be set using a
     series of 'argument'--value pairs. The following parameters (arguments)
     can be used:

              'c': scalar, speed of acoustic waves (in m/s). Defaults to 343.
       'Delta_dB': scalar (in dB), parameter determining how much the resulting
                   impulse response is cropped (i.e. RIR length): the impulse
                   response is computed until the time index where its overall
                   energy content has decreased by 'Delta_dB' decibels, after
                   which the computations stop. Not relevant if BETA=zeros(1,6).
                   Defaults to 50.
       'MaxDelay': scalar (in seconds), defines the desired length for the
                   computed RIR. If defined as non-empty, this parameter
                   overrides the setting of 'Delta_dB'. Use 'MaxDelay' if the
                   RIR length is known exactly prior to its computation. Not
                   relevant if BETA=zeros(1,6). Defaults to [].
     'SilentFlag': set to 1 to disable all on-screen messages from this
                   function. Defaults to 0.

     This function returns the time coefficients of the filter (transfer
     function) in the parameter RIR. The filter coefficients are real and
     non-normalised. The first value in the vector RIR, i.e., RIR(1),
     corresponds to time t=0. The number of coefficients returned is variable
     and results from the value of 'Delta_dB' defined by the user: the filter
     length will be as large as necessary to capture all the relevant
     highest-order reflections.

     This implementation uses Lehmann and Johansson's variant (see "Prediction
     of energy decay in room impulse responses simulated with an image-source
     model", J. Acoust. Soc. Am., vol. 124(1), pp. 269-277, July 2008) of
     Allen & Berkley's "Image Method for Efficiently Simulating Small-room
     Acoustics" (J. Acoust. Soc. Am., vol. 65(4), April 1979). This function
     implements a phase inversion for each sound reflection off the room's
     boundaries, which leads to more accurate room impulse responses (when
     compared to RIRs recorded in real acoustic environments). Also, the
     computations make use of fractional delay filters, which allow the
     representation of non-integer delays for all acoustic reflections.

     Explanations for the following code -------------------------------------
     This implementation of the image method principle has been speficically
     optimised for execution speed. The following code is based on the
     observation that for a particular dimension, the delays from the image
     sources to the receiver increases monotonically as the absolute value of
     the image index (m, n, or l) increases. Hence, all image sources whose
     indices are above or equal to a specific limit index (for which the
     received delay is above the relevant cut-off value) can be discarded. The
     following code checks, for each dimension, the delay of each received
     path and automatically determines when to stop, thus avoiding unnecessary
     computations (the amount of TF cropped depends on the 'Delta_dB'
     parameter).
     The resulting number of considered image sources hence automatically
     results from environmental factors, such as the room dimensions, the
     source and sensor positions, and the walls' reflection coefficients. As a
     result, the length of the computed transfer function has an optimally
     minimum length (no extra padding with negligibly small values).
    --------------------------------------------------------------------------
    """
    global RIRvec
    global TimePoints
    # -=:=- Check user input:
    if X_rcv[0] >= room[0, 0] or \
            X_rcv[1] >= room[0, 1] or \
            X_rcv[2] >= room[0, 2] or \
            X_rcv[0] <= 0 or \
            X_rcv[1] <= 0 or \
            X_rcv[2] <= 0:
        raise ValueError('Receiver must be within the room boundaries!')
    elif X_src[0] >= room[0, 0] or \
            X_src[1] >= room[0, 1] or \
            X_src[2] >= room[0, 2] or \
            X_src[0] <= 0 or \
            X_src[1] <= 0 or \
            X_src[2] <= 0:
        raise ValueError('Source must be within the room boundaries!')
    elif np.any(beta > 1) or np.any(beta < 0):
        raise ValueError('Parameter ''BETA'' must be in the range [0...1).')

    beta = - np.abs(beta)       # implement phase inversion in Lehmann & Johansson's ISM implementation

    X_src = X_src[:]            # Source location
    X_rcv = X_rcv[:]            # Receiver location
    beta = beta[:]              # Reflection coefficients
    Rr = 2 * room[:]            # Room dimensions

    # -=:=- Calculate maximum time lag to consider in RIR -=:=-
    if np.any(beta != 0) and rt_val != 0:      # non-anechoic case: compute RIR's decay time necessary to reach
        if MaxDelay is None:          # Delta_dB (using Lehmann & Johansson's EDC approximation method)
            MaxDelay = ISM_RIR_DecayTime(Delta_dB, rt_type, rt_val, 1 - beta ** 2, room, X_src, X_rcv, Fs, c)
    else:                               # Anechoic case: allow for 5 times direct path in TF
        DPdel = np.norm(X_rcv - X_src) / c  # direct path delay in [s]
        MaxDelay = 5 * DPdel
        beta = np.zeros((6,))		# in case rt_val=0 only
    TForder = int(np.ceil(MaxDelay * Fs))       # total length of RIR [samp] to reach Delta_dB

    TimePoints = np.arange(TForder) / Fs  # NOQA
    RIRvec = np.zeros((TForder,))

    # -=:=- Summation over room dimensions:
    for a in range(2):
        for b in range(2):
            for d in range(2):
                m = 1              # Check delay values for m=1 and above
                FoundLValBelowLim = Check_lDim(a, b, d, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)
                while FoundLValBelowLim == 1:
                    m += 1
                    FoundLValBelowLim = Check_lDim(a, b, d, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)
                m = 0              # Check delay values for m=0 and below
                FoundLValBelowLim = Check_lDim(a, b, d, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)
                while FoundLValBelowLim == 1:
                    m -= 1
                    FoundLValBelowLim = Check_lDim(a, b, d, m, X_rcv, X_src, Rr, c, MaxDelay, beta, Fs)
    return RIRvec
