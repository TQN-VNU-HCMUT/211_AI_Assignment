from .vietocr.tool.predictor import Predictor
from .vietocr.tool.config import Cfg

class TextRecognition():
    def __init__(self):
        config = Cfg.load_config_from_name('vgg_transformer')

        # config['weights'] = './weights/transformerocr.pth'
        config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False

        self.detector = Predictor(config)

    def predict(self, img):
        return self.detector.predict_batch(img)



