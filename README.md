# ism_rir
[![Build Status](https://travis-ci.com/Fhrozen/ism_rir.svg?branch=master)](https://travis-ci.com/Fhrozen/ism_rir)

Image-source method for room acoustics

This code is a transcription for python. The original work for matlab is found at [Lehmann's main page](http://www.eric-lehmann.com/).

## Example
Execute `test_rir.py` found in the example folder
```sh
$ python example/test_rir.py
```

## ToDo:
- Enable additional reveberation algorithm. 

## Requirements

- numpy
- scipy
- pathos (Optional): Parallel processing.

## Benchmark:
Time evaluated for a 3-microphones array (in seconds):

| Language |  T60<br>(Reveb. 200ms) | T60<br>(Reveb. 800ms) | T20<br>(Reveb. 200ms) |
| --- | --- | --- | ---- |
| Matlab | 5 | 900 | 774 |
| Python (no pool) | 7 | 600 | 276 | 
| Python (3 cpus) | 6 | 360 | 166 | 

## References

[1]	E. Lehmann and A. Johansson, Prediction of energy decay in room impulse responses simulated with an image-source model, Journal of the Acoustical Society of America, vol. 124(1), pp. 269-277, July 2008.

[2]	E. Lehmann, A. Johansson, and S. Nordholm, Reverberation-time prediction method for room impulse responses simulated with the image-source model, Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA'07), pp. 159-162, New Paltz, NY, USA, October 2007.