import ast
import os.path
from xml.dom import minidom
import cv2
from ultralytics import YOLO
root = 'C:/!Lvasuk/Programing/qlSpineFixer/\DataSet'
from os import listdir
from os.path import isfile, join

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


#### ONLY ONE OBJECT PER IMAGE IS ASSUMED ####

def CVATtoCOCO():
    out_dir = './out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = [minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations1.xml'),
             minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations2.xml'),
             minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations3.xml')]

    for file in files:
        #file = minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations1.xml')
        #file = minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations2.xml')
        #file = minidom.parse('C:/!Lvasuk/Programing/qlSpineFixer/annotations3.xml')

        images = file.getElementsByTagName('image')

        for image in images:

            width = int(image.getAttribute('width'))
            height = int(image.getAttribute('height'))
            name = image.getAttribute('name')
            elem = image.getElementsByTagName('points')
            bbox = image.getElementsByTagName('box')[0]
            xtl = int(float(bbox.getAttribute('xtl')))
            ytl = int(float(bbox.getAttribute('ytl')))
            xbr = int(float(bbox.getAttribute('xbr')))
            ybr = int(float(bbox.getAttribute('ybr')))
            w = xbr - xtl
            h = ybr - ytl
            label_file = open(os.path.join(out_dir, name[:-4] + '.txt'), 'w')
            print(elem)

            for e in elem:

                label_file.write('0 {} {} {} {} '.format(str((xtl + (w / 2)) / width), str((ytl + (h / 2)) / height),
                                                         str(w / width), str(h / height)))

                points = e.attributes['points']
                points = points.value.split(';')
                points_ = []
                print(points)
                for p in points:
                    p = p.split(',')
                    p1, p2 = p
                    points_.append([int(float(p1)), int(float(p2))])
                for p_, p in enumerate(points_):
                    label_file.write('{} {}'.format(p[0] / width, p[1] / height))
                    if p_ < len(points_) - 1:
                        label_file.write(' ')
                    else:
                        label_file.write('\n')