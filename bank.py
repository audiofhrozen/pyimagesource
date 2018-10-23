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

from ism_rir.abscoeff import AbsCoeff
from ism_rir.room_resp import ISM_RoomResp
import logging
import numpy as np


def audiodata(rirs, data):
    # Data should be 1-D
    # Reverberation for 1 source only
    dims = rirs.shape[0]
    length = data.shape[0]
    outdata = np.zeros((dims, length))
    for i in range(dims):
        outdata[i] = np.convolve(data, rirs[i, 0], mode='full')[:length]
    return outdata


class Room_Impulse_Response(object):
    """Environmental parameters for image-source method simulation

     Room_Impulse_Response(args)

     This function can be used as a template for the definition of the
     different parameters for an image-source method simulation, typically
     providing inputs to the functions 'ISM_RIR_bank.m' as well as
     'fast_ISM_RIR_bank.m' (Lehmann & Johansson's ISM implementations). This
     function returns the structure 'SetupStruc' with the following fields:

        sampling_freq: sampling frequency in Hz.
                 room: 1-by-3 array of enclosure dimensions (in m),
                       [x_length y_length z_length].
              mic_pos: N-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of N
                       microphones in the environment (in m).
     source_trajectory: M-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of M
                       source trajectory points in the environment (in m).
      reverberation: list of two scalar values [a, b]
                     where a is the reverberation type (60 or 20) and b
                     is the desired reverberation time (in s).
               c: (optional) sound velocity (in m/s).
     abs_weights: (optional) 1-by-6 vector of absorption coefficients weights,
                  [w_x1 w_x2 w_y1 w_y2 w_z1 w_z2].
     method: Type of Algorithm for reverberation.
     verbose: Verbosity control
     processes: number of subprocess to calculate the impulses.

     The structure field 'c' is optional in the sense that the various
     functions developed in relation to Lehmann & Johansson's ISM
     implementation assume a sound velocity of 343 m/s by default. If defined
     in the function below, the field 'SetupStruc.c' will take precedence and
     override the default value with another setting.

     The field 'abs_weight' corresponds to the relative weights of each of the
     six absorption coefficients resulting from the desired reverberation time
     T60. For instance, defining 'abs_weights' as [0.8 0.8 1 1 0.6 0.6] will
     result in the absorption coefficients (alpha) for the walls in the
     x-dimension being 20% smaller compared to the y-dimension walls, whereas
     the floor and ceiling will end up with absorption coefficients 40%
     smaller (e.g., to simulate the effects of a concrete floor and ceiling).
     Note that setting some of the 'abs_weight' parameters to 1 does NOT mean
     that the corresponding walls will end up with a total absorption! If the
     field 'abs_weight' is omitted, the various functions developed in
     relation to Lehmann & Johansson's ISM implementation will set the
     'abs_weight' parameter to [1 1 1 1 1 1], which will lead to uniform
     absorption coefficients for all room boundaries.

     The reverberation list may contain one of the two fields '60' or
     '20'. 60 corresponds to the time required by the impulse response to
     decay by 60dB, whereas 20 is defined as the time required for the
     impulse response to decay from -5 to -25dB. Simply define either one of
     these fields in the file below. Set this value to 0 for anechoic
     environments (direct path only).
    """

    def __init__(self, sampling_freq,
                 room, mic_pos, source_trajectory, reverberation, abs_weights=None,
                 c=None, method='LehmannJohansson', verbose=False, processes=1):
        super(Room_Impulse_Response, self).__init__()
        self.sampling_freq = sampling_freq
        self.room = room
        self.mic_pos = mic_pos
        self.source_trajectory = source_trajectory
        self.reverberation = reverberation
        self.abs_weights = abs_weights
        self.c = c
        self.method = method
        self.verbose = verbose
        if processes < 1:
            raise ValueError('The number of processes should be equal or greater than 1')
        self.processes = processes
        if verbose:
            logging.basicConfig(
                level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        else:
            logging.basicConfig(
                level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
            logging.warning('Skip DEBUG/INFO messages')

    def bank(self, Delta_dB=50.0):

        """Function [RIR_cell] = ISM_RIR_bank(setupstruc,RIRFileName,varargin)

          ISM_RIR_bank  Bank of RIRs using Lehmann & Johansson's image-source method

          [RIR_CELL] = ISM_RIR_bank(SETUP_STRUC,RIR_FILE_NAME)

          This function generates a bank of room impulse responses (RIRs) for a
          particular user-defined room setup, using Lehmann and Johansson's
          implementation of the image-source method (see: "Prediction of energy
          decay in room impulse responses simulated with an image-source model", J.
          Acoust. Soc. Am., vol. 124(1), pp. 269-277, July 2008). The input
          SETUP_STRUC is a structure of enviromental parameters containing the
          following fields:

                  Fs: sampling frequency (in Hz).
                room: 1-by-3 vector of enclosure dimensions (in m),
                      [x_length y_length z_length].
             mic_pos: N-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of N
                      microphones (in m).
            src_traj: M-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of M
                      source trajectory points (in m).
          reverberation (T20 or T60): scalar value (in s), desired reverberation time.
                   c: (optional) sound velocity (in m/s).
          abs_weights: (optional) 1-by-6 vector of absorption coefficients weights,
                      [w_x1 w_x2 w_y1 w_y2 w_z1 w_z2].

          If the field SETUP_STRUC.c is undefined, the function assumes a default
          value of sound velocity of 343 m/s.

          The field 'abs_weight' corresponds to the relative weights of each of the
          six absorption coefficients resulting from the desired reverberation time.
          For instance, defining 'abs_weights' as [1 1 0.8 0.8 0.6 0.6] will result
          in the absorption coefficients (alpha) for the walls in the y-dimension
          being 20% smaller compared to the x-dimension walls, whereas the floor
          and ceiling will end up with absorption coefficients 40% smaller (e.g.,
          to simulate the effects of a concrete floor and ceiling). If this field
          is omitted, the parameter 'abs_weight' will default to [1 1 1 1 1 1],
          which leads to uniform absorption coefficients for all room boundaries.

          The structure SETUP_STRUC may contain one of the two fields 'T60' or
          'T20'. This function will automatically determine which reverberation
          type is used and compute the desired room absorption coefficients
          accordingly. T20 is defined as the time required for the impulse response
          energy to decay from -5 to -25dB, whereas T60 corresponds to the time
          required by the impulse response energy to decay by 60dB. Setting the
          corresponding field value to 0 achieves anechoic impulse responses
          (direct path only).

          In addition, a number of other (optional) parameters can be set using a
          series of 'argument'--value pairs. The following parameters (arguments)
          can be used:

           'Delta_dB': scalar (in dB), parameter determining how much the resulting
                       impulse response is cropped: the impulse response is
                       computed until the time index where its overall energy
                       content has decreased by 'Delta_dB' decibels, after which
                       the computations stop. Not relevant if the reverberation
                       time is set to 0 (anechoic case). Defaults to 50.

          This function returns a 2-dimensional cell array RIR_CELL containing the
          RIRs for each source trajectory point and each microphone, organised as
          follows: RIR_CELL{mic_index,traj_index}. The resulting filter length
          may differ slightly in each computed RIR.

          This function also saves the computation results on file. The argument
          RIR_FILE_NAME determines the name of the .mat file where the variable
          RIR_CELL is to be saved. If a file already exists with the same name as
          the input argument, the user will be prompted to determine whether the
          file is to be overwritten or not. The given parameter RIR_FILE_NAME can
          be a full access path to the desired file. If no access path is given,
          the file is saved in the current working directory.
        """

        if self.abs_weights is None:
            self.abs_weights = np.ones((1, 6))
        elif self.abs_weights.shape[1] != 6:
            logging.warning('The given weights is not an array of 6, the values will be set to 1')
            self.abs_weights = np.ones((1, 6))

        if self.c is None:
            self.c = 343.0
        if self.reverberation[0] == 60:
            alpha = AbsCoeff('t60', self.reverberation[1], self.room, self.abs_weights, self.method, self.c)
        elif self.reverberation[0] == 20:
            alpha = AbsCoeff('t20', self.reverberation[1], self.room, self.abs_weights, self.method, self.c)
        else:
            raise ValueError('Missing T60 or T20 field.')
        rttype = self.reverberation[0]
        rtval = self.reverberation[1]
        beta = np.sqrt(1 - alpha)

        nMics = self.mic_pos.shape[0]  # number of microphones
        nSPts = self.source_trajectory.shape[0]  # number of source trajectory points

        # -=:=- Compute RIR bank -=:=-
        RIR_cell = np.empty((nMics, nSPts), dtype=object)  # pre-allocate cell array
        logging.info('Computing room impulse responses. ')
        mics_range = range(nMics)
        if self.processes > 1:
            pass
        else:
            if self.verbose:
                from tqdm import tqdm
                mics_range = tqdm(mics_range)
            for mm in mics_range:
                X_rcv = self.mic_pos[mm, :]
                # compute ISM room impulse response for each source-receiver combinations
                for tt in range(nSPts):
                    X_src = self.source_trajectory[tt, :]
                    RIR_cell[mm, tt] = ISM_RoomResp(self.sampling_freq,
                                                    beta, rttype, rtval,
                                                    X_src, X_rcv, self.room,
                                                    self.c, Delta_dB)
        self.RIR_cell = RIR_cell
        return RIR_cell
