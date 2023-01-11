import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

from scipy.signal import find_peaks, find_peaks_cwt


def quality_color(quality):
    if quality < 1.5:
        return "red"
    if quality < 3.5:
        return "orange"
    return "green"


def raw_data_quality(data: np.ndarray, plot: bool = False) -> float:
    """
    Computes a quality index for a raw ultrasound reading

    Parameters
    ----------
    data : np.ndarray
        Raw ultrasound data
    plot : bool
        If true, plot the input ultrasound data to visualise the quality

    Returns
    -------
    float quality:
        Quality index on a 1-5 scale. 5 means good quality, 1 means poor quality.
    """
    min_peak_height = 40

    # Remove main bang
    data = data[3000:]

    maxs, _ = find_peaks(data, height=min_peak_height, width=20)

    if len(maxs) == 0:
        return 1

    max_peak_i = np.argmax(data[maxs])
    maxs_without_biggest = maxs[maxs != maxs[max_peak_i]]

    hist, _ = np.histogram(data[maxs], bins=[0, 80, 120, 160, 200, 256], density=True)
    hist = hist / np.max(hist)

    if plot:
        plt.plot(data, color="blue")
        plt.plot(maxs, data[maxs], "o")
        plt.plot(maxs[max_peak_i], data[maxs][max_peak_i], "x")
        plt.ylim([0, 255])
        print(hist)

    if hist[4] > 0:
        quality = 5
        if hist[4] > 3:
            quality -= 1
    elif hist[3] > 0:
        quality = 4
        if hist[3] > 4:
            quality -= 1
    elif hist[2] > 0:
        quality = 3
        if hist[2] > 5:
            quality -= 1
    elif hist[1] > 0:
        quality = 2
        if hist[1] > 6:
            quality -= 1
    else:
        quality = 1

    return quality


def detect_main_bang(data) -> Tuple(int, int):
    # TODO
    pass


def wavefront(data, temperature, threshold, window_in_meters, freq=500000):
    """Finds the wavefront index in a raw ultrasound data

    Parameters
    ----------
    data : List[int]
        Raw ultrasound data
    temperature : float
        Temperature of the device in Celsius
    threshold : float ([0 - 1])
        Threshold as a fraction of the max value to find the wavefront, between 0 and 1
    window_in_meters : float
        Window around the max value in which to find the wavefront, in meters
    freq : int, optional
        Frequency of the signal in Hz, by default 500000

    Returns
    -------
    int
        Index of the wavefront
    """
    def argmax(x):
        return max(range(len(x)), key=lambda i: x[i])

    # Error handling
    if math.isnan(temperature) or len(data) < 10000:
        return -1

    # Index at which to cut the main bang
    bang_index = 2500

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_index:])
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)

    # Compute the window to check for the threshold around the max index,
    # which depends on the frequency of the signal
    window_in_samples = int(window_in_meters / sound_speed * freq) * 2

    # Find index of max value
    max_index = argmax(data[bang_index:]) + bang_index

    # Start and end index of the window in which to check for the threshold
    start = int(max_index - window_in_samples / 2)
    end = int(max_index + window_in_samples / 2)

    # Find the index at which the data is at the threshold inside the window
    wf_index = -1
    for i, d in enumerate(data[start:end], start=start):
        if d > threshold:
            wf_index = i
            break
    if False:
        print(sound_speed)
        print(max_index)
        print(window_in_samples)
        print(start, end)
        print(wf_index)
        plt.plot(data)
        plt.axhline(threshold)
        plt.axvline(start, color="black", linestyle="--")
        plt.axvline(end, color="black", linestyle="--")
        plt.axvline(wf_index, color="green")
        plt.axvline(max_index, color="yellow", linestyle="--")
        plt.show()
    return wf_index


def denoise(data: np.ndarray, threshold: float) -> np.ndarray:
    data[data < np.max(data) * threshold] = 0
    return data


def denoise_abs(data: np.ndarray, threshold: float) -> np.ndarray:
    data[data < threshold] = 0
    return data


def algo_v1(data: np.ndarray, plot: bool = False) -> float:

    data_untouched = data.copy()

    time_delay = 4000
    #data = data / np.max(data[time_delay:])
    data = data[time_delay:]
    min_peak_height = 0.5 * np.max(data)
    denoised = denoise(data.copy(), 0.1)

    """
    if plot:
        pd_data = pd.Series(data)
        noise_threshold = NoiseThresholdv1().process(data)
        main_bang = MainBangDetectorv1().process(pd_data, noise_threshold)

        plt.plot(denoised, color="blue")
        plt.axvline(main_bang["main_bang_start"], color="black", label="Main bang start")
        plt.axvline(main_bang["main_bang_end"], color="black", label="Main bang end")
        plt.axhline(y=min_peak_height, color="black", linestyle="--")
    """

    maxs, _ = find_peaks(denoised, height=min_peak_height, width=20)
    max_peak_i = np.argmax(denoised[maxs])

    if plot:
        plt.plot(denoised, color="blue")
        max_peak = denoised[maxs][max_peak_i]
        plt.plot(maxs[max_peak_i], max_peak, "o")

        plt.plot(maxs, denoised[maxs], "x")

    tof = np.average(maxs, weights=range(len(maxs), 0, -1))

    #tof = denoised.dot(range(len(denoised))) / np.sum(denoised)

    tof_interp = np.interp(tof, [5000, 30000], [-1, 1])
    tof += 2000 * tof_interp

    tof = (tof + time_delay) * 2
    return tof


@dataclass
class Window:
    start_index: int
    width: int


def find_best_window(data: np.ndarray, width: int) -> Window:
    start_index = 2500
    resolution = 50

    highest_sum = 0
    best_window = Window(start_index=start_index, width=width)

    for current_index in range(start_index, len(data) - width, resolution):
        data_window = data[current_index:current_index + width]
        current_sum = data_window.sum()
        if current_sum > highest_sum:
            highest_sum = current_sum
            best_window = Window(start_index=current_index, width=width)

    return best_window


def compute_cdm(data: np.ndarray, start_index: int) -> int:
    nt = NoiseThresholdv1()
    threshold = nt.process(data)
    data = denoise_abs(data.copy(), threshold)
    indices = np.arange(start_index, len(data) + start_index)
    data = data*data
    cdm = 0 if np.sum(data) == 0 else np.average(indices, weights=data)
    return int(cdm)


def compute_cdm_in_window(data: np.ndarray, start_index: int, window_size: int) -> Tuple[int, Window]:
    win = find_best_window(data[start_index:], window_size)
    cdm = compute_cdm(data[win.start_index:win.start_index + win.width], start_index=win.start_index)
    return cdm, win


class NoiseThresholdv1(object):
    def __init__(self):
        pass

    @classmethod
    def name(cls):
        return 'NoiseThresholdv1'

    def process(self, data):
        average_post_main_bang = data[(len(data) // 5):].mean()
        nearest_gate = (average_post_main_bang // 15) * 15.0

        if nearest_gate < 15.0:
            return 15.0
        elif nearest_gate < 60.0:
            return nearest_gate
        else:
            return 60.0


class MainBangDetectorv1(object):
    def __init__(self):
        pass

    @classmethod
    def name(cls):
        return 'MainBangDetectorv1'

    def process(self, data, noise_threshold) -> dict:

        output = {'main_bang_start': None, 'main_bang_end': None}

        output['main_bang_start'] = self._find_start(data, noise_threshold)
        output['main_bang_end'] = self._find_end(data, noise_threshold, output['main_bang_start'])

        return output

    def _find_start(self, data, noise_threshold):
        for index, sample in data.items():
            if sample > noise_threshold:
                return index
        return None

    def _find_end(self, data, noise_threshold, start_idx):
        inflection_index = self._find_neg_inflection(data[start_idx:])
        return self._find_main_bang_minimum(data[inflection_index:], noise_threshold)

    def _find_neg_inflection(self, data):
        max = 0
        half_max = 0
        for index, sample in data.items():
            if sample > max:
                max = sample
                half_max = sample / 2
            elif sample < half_max:
                return index
        return None

    def _find_main_bang_minimum(self, data, noise_threshold):
        minimum_index = data.index[0]
        for index, sample in data.items():
            minimum_sample = data[minimum_index]
            if sample < minimum_sample:
                minimum_index = index
                minimum_sample = sample

            if (minimum_sample < noise_threshold) or (sample > minimum_sample + 4*15):
                return minimum_index
        return None


class MainBangDetectorMinimalDelay(object):
    def __init__(self, minimal_delay):
        self.minimal_delay = minimal_delay

    def name(self):
        return 'MainBangDetectorMinimalDelay_' + str(self.minimal_delay)

    def process(self, data, noise_threshold):

        output = {'main_bang_start': None, 'main_bang_end': None}

        output['main_bang_start'] = self._find_start(data, noise_threshold)
        output['main_bang_end'] = self._find_end(data, noise_threshold, output['main_bang_start'] + self.minimal_delay)

        return output

    def _find_start(self, data, noise_threshold):
        for index, sample in data.items():
            if sample > noise_threshold:
                return index
        return None

    def _find_end(self, data, noise_threshold, start_idx):
        for index, sample in data[start_idx:].items():
            if sample < noise_threshold:
                return index
        return None

        #inflection_index = self._find_neg_inflection(data[start_idx:])
        # return self._find_main_bang_minimum(data[inflection_index:], noise_threshold)

    def _find_neg_inflection(self, data):
        max = 0
        half_max = 0
        for index, sample in data.items():
            if sample > max:
                max = sample
                half_max = sample / 2
            elif sample < half_max:
                return index
        return None

    def _find_main_bang_minimum(self, data, noise_threshold):
        minimum_index = data.index[0]
        for index, sample in data.items():
            minimum_sample = data[minimum_index]
            if sample < minimum_sample:
                minimum_index = index
                minimum_sample = sample

            if (minimum_sample < noise_threshold) or (sample > minimum_sample + 4*15):
                return minimum_index
        return None


class CenterOfMassv1(object):
    def __init__(self):
        pass

    @classmethod
    def name(cls):
        return 'CenterOfMassv1'

    def process(self, data, noise_threshold):

        data[data < noise_threshold] = 0

        if data.sum() == 0:
            return 0
        return data.dot(data.index) // data.sum()


class CenterOfMassLin(object):
    def __init__(self):
        pass

    @classmethod
    def name(cls):
        return 'CenterOfMassLin'

    def process(self, data, noise_threshold):

        data -= noise_threshold
        data[data < 0] = 0

        if data.sum() == 0:
            return 0

        return data.dot(data.index) // data.sum()


class CenterOfMassQuad(object):
    def __init__(self):
        pass

    @classmethod
    def name(cls):
        return 'CenterOfMassQuad'

    def process(self, data, noise_threshold):

        data -= noise_threshold
        data[data < 0] = 0

        data2 = data.pow(2)

        if data2.sum() == 0:
            return 0

        return data2.dot(data2.index) // data2.sum()


class CenterOfMassQuadwGain(object):
    def __init__(self, map_contiguous, gain_map, map_name):
        self.map_contiguous = map_contiguous
        self.gain_map = gain_map
        self.map_name = map_name

    def name(self):
        return 'CenterOfMassQuadwGain_' + self.map_name

    def process(self, data, noise_threshold):

        data -= noise_threshold
        data[data < 0] = 0

        data2 = data.pow(2)

        sum_weighted_val = 0.0
        sum_val = 0.0

        if self.map_contiguous:
            map = np.empty(data2.size)
            map.fill(self.gain_map[-1])
            map[:self.gain_map.size] = self.gain_map

            data2 *= map

            sum_weighted_val = data2.dot(data2.index)
            sum_val = data2.sum()
        else:
            gain_it = np.nditer(self.gain_map)

            for idx, val in data2.iteritems():
                if val or self.map_contiguous:
                    try:
                        gain_val = next(gain_it)
                    except StopIteration:
                        gain_val = self.gain_map[-1]  # keep last value

                    sum_weighted_val += idx*val*gain_val
                    sum_val += val*gain_val

        if sum_val == 0:
            return 0
        return sum_weighted_val // sum_val


class EdgeDetectorwGain(object):
    def __init__(self, map_contiguous, gain_map, map_name):
        self.map_contiguous = map_contiguous
        self.gain_map = gain_map
        self.map_name = map_name

    def name(self):
        return 'EdgeDetectorwGain_' + self.map_name

    def process(self, data, noise_threshold):

        data -= noise_threshold
        data[data < 0] = 0

        if self.map_contiguous:
            map = np.empty(data.size)
            map.fill(self.gain_map[-1])
            map[:self.gain_map.size] = self.gain_map

            data *= map

            edge_idx = data[data.gt(data.max() / 2)].index[0]
        else:
            edge_idx = data.size

        return edge_idx
