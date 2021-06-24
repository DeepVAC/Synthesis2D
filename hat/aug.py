from deepvac.aug import Composer, AugFactory

class SynthesisHatComposer(Composer):
    def __init__(self, deepvac_config):
        super(SynthesisHatComposer, self).__init__(deepvac_config)
        ac = AugFactory("ColorJitterAug@0.5 => BrightnessJitterAug@0.5 => ContrastJitterAug@0.5 => Pil2CvAug => GaussianAug@0.5", deepvac_config)
        self.addAugFactory("ac", ac)