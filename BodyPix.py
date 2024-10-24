import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
from matplotlib import pyplot as plt
import numpy as np

def bodyPix():
    load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
    bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

    # get vid cap device
    cap = cv2.VideoCapture(0)

    # loop through frame
    while cap.isOpened():
        ret, frame = cap.read()

        # BodyPix Detections
        result = bodypix_model.predict_single(frame)
        mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
        masked_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Show result to user on desktop
        cv2.imshow('BodyPix', masked_image)

        # Break loop outcome
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()  # Releases webcam or capture device
    cv2.destroyAllWindows()  # Closes imshow frames


