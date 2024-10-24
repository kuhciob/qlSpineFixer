import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mediapipe.python import solution_base
from ultralytics import YOLO
from PIL import Image

root = 'C:/!Lvasuk/Programing/qlSpineFixer/spinefix/'

def apply_perspective_transform(src_image, res_img, src_points, dst_points):
    # Обчислюємо матрицю гомографічного перетворення на основі ключових точок
    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    # Застосовуємо перспективне перетворення до зображення м'язів
    transformed_image = cv2.warpPerspective(src_image, M, (res_img.shape[1], res_img.shape[0]))
    return transformed_image

def retractKeypoints(inkeypoints, image):
    h, w, _ = image.shape

    arr = inkeypoints.xyn.numpy()[0]

    keypoints = [
        (int(arr[5][0] * w),
         int(arr[5][1] * h)),
        (int(arr[6][0] * w),
         int(arr[6][1] * h)),
        (int(arr[11][0] * w),
         int(arr[11][1] * h)),
        (int(arr[12][0] * w),
         int(arr[12][1] * h))
    ]

    return keypoints

def crop_back_section(image, keypoints, padding=100):
    left_shoulder, right_shoulder, left_hip, right_hip = keypoints
    x_min = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]) - padding
    x_max = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]) + padding
    y_min = min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]) - padding
    y_max = max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]) + padding

    cropped_section = image[y_min:y_max, x_min:x_max]
    return cropped_section, (x_min, y_min)

def yoloDetection():
    model = YOLO('yolov8m-pose.pt')

    path = root + '/back03.png' #[(216, 221), (394, 216), (249, 488), (356, 488)]

    #muscle_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    muscle_image = cv2.imread(path)

    muscle_keypoints = [(216, 221), (394, 216), (249, 488), (356, 488)]

    if not muscle_keypoints:
        muscle_results = model(muscle_image, show_boxes=False)
        if len(muscle_results) > 0:
            muscle_keypoints = retractKeypoints(muscle_results[0].keypoints, muscle_image)

        else:
            print("Ключові точки на зображенні м'язів не знайдено")
            exit()

    print(muscle_keypoints)

    cap = cv2.VideoCapture(0)
    resolutions = [[640, 480], [1280, 720], [1920, 1080]]
    resolution = resolutions[1]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cropped_back, top_left_corner = crop_back_section(muscle_image, muscle_keypoints)

    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame
        image.flags.writeable = False

        try:
            results = model(image, show=False, conf=0.3, show_boxes=False)
            result = results[0]
            video_keypoints  = retractKeypoints(result.keypoints, image)

            # Plot results image
            pts1 = np.float32([
                (muscle_keypoints[0][0] - top_left_corner[0], muscle_keypoints[0][1] - top_left_corner[1]),
                (muscle_keypoints[1][0] - top_left_corner[0], muscle_keypoints[1][1] - top_left_corner[1]),
                (muscle_keypoints[2][0] - top_left_corner[0], muscle_keypoints[2][1] - top_left_corner[1]),
                (muscle_keypoints[3][0] - top_left_corner[0], muscle_keypoints[3][1] - top_left_corner[1])
            ])
            # pts1 = np.float32([
            #     (muscle_keypoints[0][0], muscle_keypoints[0][1] ),
            #     (muscle_keypoints[1][0], muscle_keypoints[1][1] ),
            #     (muscle_keypoints[2][0], muscle_keypoints[2][1] )
            # ])
            pts2 = np.float32([video_keypoints[0], video_keypoints[1], video_keypoints[2], video_keypoints[3]])

            # Вибираємо чотири ключові точки для перспективного перетворення (наприклад, плечі та стегна)
            src_points = pts1
            dst_points = pts2

            # Застосовуємо перспективне перетворення до зображення м'язів
            transformed_muscle_image = apply_perspective_transform(cropped_back, image, src_points, dst_points)

            #matrix = cv2.getAffineTransform(pts1, pts2)
            #transformed_back = cv2.warpAffine(cropped_back, matrix, (image.shape[1], image.shape[0]))

            image = result.plot(boxes = False)  # BGR-order numpy array
            image = cv2.addWeighted(image, 0.7, transformed_muscle_image, 0.5, 0)
        except:
            pass

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break