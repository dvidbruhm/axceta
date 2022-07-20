import numpy as np

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

        output = {'main_bang_start':None, 'main_bang_end':None}

        output['main_bang_start'] = self._find_start(data, noise_threshold)
        output['main_bang_end'] = self._find_end(data, noise_threshold, output['main_bang_start'])

        return output

    def _find_start(self, data, noise_threshold):
        for index,sample in data.items():
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

        output = {'main_bang_start':None, 'main_bang_end':None}

        output['main_bang_start'] = self._find_start(data, noise_threshold)
        output['main_bang_end'] = self._find_end(data, noise_threshold, output['main_bang_start'] + self.minimal_delay)

        return output

    def _find_start(self, data, noise_threshold):
        for index,sample in data.items():
            if sample > noise_threshold:
                return index
        return None

    def _find_end(self, data, noise_threshold, start_idx):
        for index, sample in data[start_idx:].items():
            if sample < noise_threshold:
                return index
        return None

        #inflection_index = self._find_neg_inflection(data[start_idx:])
        #return self._find_main_bang_minimum(data[inflection_index:], noise_threshold)

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
                        gain_val = self.gain_map[-1] # keep last value

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
