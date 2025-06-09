from essentia_engine import EssentiaEngine
import numpy as np
from essentia.standard import Extractor, MonoLoader, FrameGenerator, PoolAggregator
import essentia
import math

debug = False 

class Extractor:
    """
    The extractor uses the Essentia engine to extract features from an audio file.
    The input is a path to an audio file and the output is segment information. 
    """

    # initialize
    def __init__(self, sr, fs, hs):
        self.sample_rate = sr  # sample rate
        self.frame_size = fs  # samples in each frame
        self.hop_size = hs
  
        # the essentia engine make sure that the features were extracted
        self.engine = EssentiaEngine(
            self.sample_rate, self.frame_size, self.hop_size)

    # run the segmentation
    def extract(self, afile):
        # extract the regions
        segments = self.extract_regions(afile)
        return segments 

    # use the bf classifier to extract background, foreground, bafoground regions
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def extract_regions(self, afile):

        # instantiate the loading algorithm
        loader = MonoLoader(filename=afile, sampleRate=self.sample_rate)
        
        # perform the loading
        audio = loader()

        # create pool for storage and aggregation
        pool = essentia.Pool()
        accumulator = {}
 
        processed = []  # storage for the classified segments
        # extract all features

        pool = self.engine.extractor(audio)

        aggrigated_pool = PoolAggregator(defaultStats=['mean', 'stdev'])(pool)

        # narrow everything down to select features
        features_dict = {}
        descriptor_names = aggrigated_pool.descriptorNames()

        # unpack features in lists
        for descriptor in descriptor_names:
            # little to no values in these features, ignore
            if('tonal' in descriptor or 'rhythm' in descriptor):
                continue
            value = aggrigated_pool[descriptor]
            # unpack arrays
            if (str(type(value)) == "<class 'numpy.ndarray'>"):
                for idx, subVal in enumerate(value):
                    features_dict[descriptor + '.' + str(idx)] = subVal
                continue
            # ignore strings
            elif(isinstance(value, str)):
                pass
            # add singular values
            else:
                features_dict[descriptor] = value

        # reset counter and clear pool
        pool.clear()
        aggrigated_pool.clear()
        del pool
        del aggrigated_pool

        # prepare dictionary for filtering
        vector = np.array(list(features_dict.values()))
        fnames = np.array(list(features_dict.keys()))

        # remove NAN values, this can happen on segments of short length
        vector = np.nan_to_num(vector)

        # create clean dictionary for the database
        features_filtered = {}
        for idx, val in enumerate(vector):
            features_filtered[fnames[idx]] = val
        for key in features_filtered.keys():
            if key not in accumulator.keys():
                accumulator[key] = features_filtered[key]
            else:
                accumulator[key] += features_filtered[key]

        return accumulator