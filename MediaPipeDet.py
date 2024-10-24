import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
root = 'C:/!Lvasuk/Programing/pythonProject/spinefix/'

def mediapipeGetMuscleAnatomy():
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        path = 'D:/anatomy-of-male-muscular-system-back-view.jpg'
        #path = 'D:/Schwarzenegger.jpg'

        image = cv2.cvtColor( cv2.imread(path), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            keypoints = [
                (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
            ]

            for point in keypoints:
                cv2.circle(image, point, 10, (0, 0, 255), -1)

        except:
            pass


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('Mediapipe Feed', image)

        res = cv2.waitKey(0)
        print('You pressed %d (0x%x), LSB: %d (%s)' % (res, res, res % 256,
                                                       repr(chr(res % 256)) if res % 256 < 128 else '?'))
        return landmarks

def get_keypoints(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        image.flags.writeable = False
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Ключові точки: плечі та стегна
            keypoints = [
                (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
            ]


            print(keypoints)
            return keypoints
        return None

# Обрізання секції спини з запасом
def crop_back_section(image, keypoints, padding=100):
    left_shoulder, right_shoulder, left_hip, right_hip = keypoints
    x_min = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]) - padding
    x_max = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]) + padding
    y_min = min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]) - padding
    y_max = max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]) + padding

    cropped_section = image[y_min:y_max, x_min:x_max]
    return cropped_section, (x_min, y_min)

# Обчислення відстані між точками
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def scale_image_by_keypoints(muscle_image, video_keypoints, muscle_keypoints):
    # Відстані між ключовими точками на зображенні м'язів
    muscle_shoulder_distance = calculate_distance(muscle_keypoints[0], muscle_keypoints[1])
    muscle_hip_distance = calculate_distance(muscle_keypoints[2], muscle_keypoints[3])
    muscle_shoulder_hip_distance = calculate_distance(muscle_keypoints[0], muscle_keypoints[2]) +\
                                   calculate_distance(muscle_keypoints[1], muscle_keypoints[3]) / 2

    # Відстані між ключовими точками на відео
    video_shoulder_distance = calculate_distance(video_keypoints[0], video_keypoints[1])
    video_hip_distance = calculate_distance(video_keypoints[2], video_keypoints[3])
    video_shoulder_hip_distance = calculate_distance(video_keypoints[0], video_keypoints[2]) + \
                         calculate_distance(video_keypoints[1], video_keypoints[3]) / 2


    # Обчислення коефіцієнтів масштабування для точнішого накладання
    if(muscle_shoulder_distance < video_shoulder_distance):
        scale_x = video_shoulder_distance / muscle_shoulder_distance
        scale_y = video_shoulder_hip_distance / muscle_shoulder_hip_distance
    else :
        scale_x = muscle_shoulder_distance / video_shoulder_distance
        scale_y = muscle_shoulder_hip_distance / video_shoulder_hip_distance

    # Масштабування зображення м'язів
    resized_muscle_image = cv2.resize(muscle_image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    return resized_muscle_image

def mediapipeDetection():
    cap = cv2.VideoCapture(0)

    resolutions = [[640, 480], [1280, 720], [1920, 1080]]
    resolution = resolutions[1]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    path = root + '/back01.png'
    path = root + '/back03.png'

    #path = root + '/back02.jpg'
    #path = 'D:/Schwarzenegger.jpg'

    muscle_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    muscle_keypoints = get_keypoints(muscle_image)
    print(muscle_keypoints)

    for point in muscle_keypoints:
        cv2.circle(muscle_image, point, 10, (0, 0, 255), -1)

    muscle_image.flags.writeable = True
    muscle_image = cv2.cvtColor(muscle_image, cv2.COLOR_RGB2BGR)

    if muscle_keypoints:
        cropped_back, top_left_corner = crop_back_section(muscle_image, muscle_keypoints)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image.shape

                    # Ключові точки: плечі та стегна
                    video_keypoints = [
                        (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)),
                        (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)),
                        (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
                        (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
                    ]

                    if video_keypoints:
                        # Масштабування зображення з урахуванням відстаней між точками
                        #scaled_back = scale_image_by_keypoints(cropped_back, video_keypoints, muscle_keypoints)

                        # Афінне перетворення для точного розміщення
                        pts1 = np.float32([
                            (muscle_keypoints[0][0] - top_left_corner[0], muscle_keypoints[0][1] - top_left_corner[1]),
                            (muscle_keypoints[1][0] - top_left_corner[0], muscle_keypoints[1][1] - top_left_corner[1]),
                            (muscle_keypoints[2][0] - top_left_corner[0], muscle_keypoints[2][1] - top_left_corner[1])
                        ])
                        pts2 = np.float32([video_keypoints[0], video_keypoints[1], video_keypoints[2]])

                        matrix = cv2.getAffineTransform(pts1, pts2)
                        transformed_back = cv2.warpAffine(cropped_back, matrix, (frame.shape[1], frame.shape[0]))
                        # Накладання на відео
                        image = cv2.addWeighted(image, 0.7, transformed_back, 0.5, 0)

                except:
                    pass

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()