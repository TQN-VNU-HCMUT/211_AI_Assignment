from cv2 import imshow, waitKey, resize
import numpy as np
from PIL import Image

from .corner_detection.corner_detection import CornerDetection
from .text_detection.text_detection import TextDetection
from .text_recognition.text_recognition import TextRecognition
from .config import text_detection, text_recognition

from .text_detection.utils.image_utils import sort_text

class CompleteModel():
    def __init__(self):
        self.corner_detection_model = CornerDetection()
        self.text_detection_model = TextDetection(path_model=text_detection['path_to_model'],
                                                path_labels=text_detection['path_to_labels'],
                                                thres_nms=text_detection['nms_ths'], 
                                                thres_score=text_detection['score_ths'])
        self.text_recognition_model = TextRecognition()

        # init boxes
        self.id_boxes = None
        self.name_boxes = None
        self.birth_boxes = None
        self.add_boxes = None
        self.home_boxes = None

    def detect_corner(self, image):
        pass

    def detect_text(self, img):
        # detect text boxes
        detection_boxes, detection_classes, category_index = self.text_detection_model.predict(img)
        
        # sort text boxes according to coordinate
        self.id_boxes, self.name_boxes, self.birth_boxes, self.home_boxes, self.add_boxes = sort_text(detection_boxes, detection_classes)

    def recog_text(self, img):
        img = np.array(resize(img, (900, 600)))
        field_dict = dict()

        # crop boxes according to coordinate
        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0]
                crop.append(Image.fromarray(img[ymin:ymax, xmin:xmax]))
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box
                    crop.append(Image.fromarray(img[ymin:ymax, xmin:xmax]))

            return crop

        list_ans = list(crop_and_recog(self.id_boxes))
        list_ans.extend(crop_and_recog(self.name_boxes))
        list_ans.extend(crop_and_recog(self.birth_boxes))
        list_ans.extend(crop_and_recog(self.add_boxes))
        list_ans.extend(crop_and_recog(self.home_boxes))

        result = self.text_recognition_model.predict(list_ans)
        field_dict['id'] = result[0]
        field_dict['name'] = ' '.join(result[1:len(self.name_boxes) + 1])
        field_dict['birth'] = result[len(self.name_boxes) + 1]
        field_dict['home'] = ' '.join(result[-len(self.home_boxes):])
        field_dict['add'] = ' '.join(result[len(self.name_boxes) + 2: -len(self.home_boxes)])

        return field_dict

    def predict(self, img):
        cropped_img = self.corner_detection_model.scan(img)
        self.detect_text(cropped_img)
        detected_img = self.text_detection_model.text_detection_model.draw(cropped_img)
        return detected_img, self.recog_text(cropped_img)
