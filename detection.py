import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import time
import numpy as np

configs = config_util.get_configs_from_pipeline_file(r"mobilenet/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(r"mobilenet/checkpoint", 'ckpt-0')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(r"mscoco_label_map.pbtxt",
                                                                    use_display_name=True)

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

cap = cv2.VideoCapture(2) #ganti dengan port kamera (0/1/2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev_frame_time = 0
new_frame_time = 0
while True:
    ret, image_np = cap.read()
    image_np_expanded = np.expand_dims(image_np, axis=0)
    jumlahOrang = 0
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    boxes = np.squeeze(detections['detection_boxes'][0].numpy())
    scores = np.squeeze(detections['detection_scores'][0].numpy())
    classes = np.squeeze((detections['detection_classes'][0].numpy() + label_id_offset).astype(int))
    indices = np.argwhere(classes == 1)
    boxes = np.squeeze(boxes[indices])
    # print(boxes)
    scores = np.squeeze(scores[indices])
    scoreThres = scores > 0.50
    scoreCheck = scores[scoreThres]
    jumlahOrang = len(scoreCheck)
    classes = np.squeeze(classes[indices])
    # print(classes)
    if (jumlahOrang != 0):
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.50,
            agnostic_mode=False)
    cv2.rectangle(image_np_with_detections, (10,420), (320,460), (255,255,255), -1)
    cv2.putText(image_np_with_detections,"Jumlah Orang: {}".format(jumlahOrang), (20,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 3)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(image_np_with_detections,"FPS = {}".format(fps), (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 
    
    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()