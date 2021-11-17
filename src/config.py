corner_detection = {
    'path_to_model': './src/detector/config_corner_detection/model.tflite',
    'path_to_labels': './src/detector/config_corner_detection/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.3
}

text_detection = {
    'path_to_model': './src/text_detection/config/model.tflite',
    'path_to_labels': './src/text_detection/config/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.2
}

text_recognition = {
    'base_config': './src/text_recognition/config/base.yml',
    'vgg_config': './src/text_recognition/config/vgg-transformer.yml',
    'model_weight': './src/text_recognition/config/transformerocr.pth'
}
