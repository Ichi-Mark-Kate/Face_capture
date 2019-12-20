from scipy.spatial import distance
import os
import numpy as np
import tensorflow as tf
import cv2
import pafy
import collections
import dlib
import PIL.Image as Image
from glob import glob
import shutil
from skimage import io

sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


class FaceCapture:

    def face_descr(self, img):
        dets = detector(img, 1)
        if len(dets) != 0:
            for _, d in enumerate(dets):
                shape = sp(img, d)
            return facerec.compute_face_descriptor(img, shape)
        return 0

    def youtube_face_detection(self, folder_name, url, index_img=1, similarity_threshold=0.4, preftype="mp4"):
        vpafy = pafy.new(url)
        play = vpafy.getbest(preftype=preftype)
        cap = cv2.VideoCapture(play.url)
        while True:
            ret1, img1 = cap.read()
            face_descriptor1 = self.face_descr(img1)
            if face_descriptor1 != 0:
                cv2.imwrite('data/detected_faces/' + folder_name + '/image_{}.jpg'.format(index_img), np.array(img1))
                index_img += 1
                while True:
                    ret2, img2 = cap.read()
                    if img2 is None: break
                    face_descriptor2 = self.face_descr(img2)
                    if face_descriptor2 != 0:
                        dist = distance.euclidean(face_descriptor1, face_descriptor2)
                        if dist > similarity_threshold:
                            cv2.imwrite('data/detected_faces/' + folder_name + '/image_{}.jpg'.format(index_img),
                                        np.array(img2))
                            print('image_{}.jpg'.format(index_img))
                            index_img += 1
                            face_descriptor1 = face_descriptor2
                break
        cap.release()

    def crop_faces(self, folder_name='video', height=90, score=.8):

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile('model/frozen_inference_graph_face.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
                index_img = 0
                for path_index in range(1, len(os.listdir('data/detected_faces/' + folder_name + '/'))):
                    try:
                        image = cv2.imread('data/detected_faces/' + folder_name + '/image_' + str(path_index) + ".jpg")
                        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        print(path_index)
                        boxes = np.squeeze(boxes)
                        scores = np.squeeze(scores)
                        box_to_color_map = collections.defaultdict(str)
                        for i in range(boxes.shape[0]):
                            if scores[i] > score:
                                box = tuple(boxes[i].tolist())
                                box_to_color_map[box] = 'Chartreuse'
                        for box, _ in box_to_color_map.items():
                            ymin, xmin, ymax, xmax = box
                            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                            im_width, im_height = image_pil.size
                            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                            if bottom - top > height:
                                image_crop = image_pil.copy()
                                image_cut = image_crop.crop((left, top, right, bottom))
                                cv2.imwrite('data/crop_faces/' + folder_name + '/image_{}.jpg'.format(index_img), np.array(image_cut))
                                index_img += 1
                    except: continue

    def face_detect(self, path_img):
        img = io.imread(path_img)
        image_pillow = Image.fromarray(np.uint8(img)).convert('RGB')
        height, width = image_pillow.size
        rectangle = dlib.rectangle(0, 0, width, height)
        shape = sp(img, rectangle)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return os.path.basename(path_img), face_descriptor

    def face_filter(self, folder_name='video', diff_save=False, similarity_threshold=0.4):

        def save_in_diff_size(path_img, weight_im=178, height_im=218):
            img_for_resize = cv2.imread('data/original_images/' + folder_name + '/' + path_img)
            if img_for_resize is not None:
                img_for_resize = Image.fromarray(np.uint8(img_for_resize)).convert('RGB')
                resized_img = img_for_resize.resize((weight_im, height_im), Image.ANTIALIAS)
                cv2.imwrite('data/images_' + str(weight_im) + 'x' + str(height_im) + '/' + folder_name + '/'
                            + path_img, np.array(resized_img))

        path_img2_list = ['start']
        for path_img1 in glob('data/crop_faces/' + folder_name + '/*.jpg'):
            try:
                path_img1, face_descriptor1 = self.face_detect(path_img1)
                for path_img2 in glob('data/crop_faces/' + folder_name + '/*.jpg'):
                    _, face_descriptor2 = self.face_detect(path_img2)
                    dist = distance.euclidean(face_descriptor1, face_descriptor2)
                    if dist < similarity_threshold and dist != 0:
                        os.remove(path_img2)
                    path_img2_list[0] = path_img2
            except:
                continue
            shutil.move('data/crop_faces/' + folder_name + '/' + path_img1,
                        'data/original_images/' + folder_name + '/' + path_img1)
            print("save " + path_img1)
            if diff_save: save_in_diff_size(path_img1)

        if diff_save: save_in_diff_size(path_img2_list[0])


FOLDER_NAME = 'test'
weight, height = 178, 218

os.makedirs('data/images_' + str(weight) + 'x' + str(height) + '/' + FOLDER_NAME, exist_ok=True)
os.makedirs('data/original_images/' + FOLDER_NAME, exist_ok=True)
os.makedirs('data/detected_faces/' + FOLDER_NAME, exist_ok=True)
os.makedirs('data/crop_faces/' + FOLDER_NAME, exist_ok=True)

test = FaceCapture()
# test.youtube_face_detection(folder_name=FOLDER_NAME, url='https://www.youtube.com/watch?v=ppSPsvO19dU')
# test.crop_faces(folder_name=FOLDER_NAME)
test.face_filter(folder_name=FOLDER_NAME, diff_save=True)
