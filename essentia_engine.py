import essentia.standard as engine

class EssentiaEngine:
	"""
	Essentia Engine to extract audio features.
	https://essentia.upf.edu/
	"""

	def __init__(self, sampleRate, frameSize, hopSize):
		self.sampleRate = sampleRate
		self.frameSize = frameSize
		self.hopSize = hopSize

		# algorithms
		# https://essentia.upf.edu/reference/std_Extractor.html
		self.extractor = engine.Extractor(lowLevel=True, dynamics=False, rhythm=False, 
										highLevel=False, midLevel=False, tuning=False, 
										lowLevelFrameSize = frameSize, lowLevelHopSize = hopSize, 
										dynamicsFrameSize = frameSize, dynamicsHopSize = hopSize)
		print('engine start')