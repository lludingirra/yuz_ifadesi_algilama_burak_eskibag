#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("model.pkl", "rb") as f:
   model = pickle.load(f)

def draw_landmarks_on_image(rgb_image, detection_result):
  # bulunna yüzler ve o yüzler üzerindeki koordinatlar
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # print("Bulunan yüz sayısı", len(face_landmarks_list))
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    # print("Nokta sayısı", len(face_landmarks))

    # Sadece x,y ve z koordinatlarını al
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    # print(len([landmark.x for landmark in face_landmarks]))
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    koordinatlar= []
    for landmark in face_landmarks:
       koordinatlar.append(round(landmark.x, 4))
       koordinatlar.append(round(landmark.y,4))

    sonuc = model.predict([koordinatlar])
    annotated_image = cv2.putText(annotated_image, 
                                  sonuc[0], 
                                  (60,60),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                    2,
                                    (255, 255, 0),
                                    8)

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_tesselation_style())


    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())
    

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


# with open("veriseti.csv", "w") as f:
#     satir = ""
#     for i in range(1, 479):
#       satir = satir + f"x{i},y{i},"
#     satir = satir + "Etiket\n"
#     f.write(satir)

etiket = "happy"   


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Kameradan görüntü alımı
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Görüntüyü x ekseninde ters çevir
        flipped_frame = cv2.flip(frame, 1)  # 1: yatay (x ekseni) ters çevirme
        flipped_frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        mp_image_flipped = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_frame_rgb)

        detection_result_flipped = detector.detect(mp_image_flipped)

        annotated_image_flipped = draw_landmarks_on_image(mp_image_flipped.numpy_view(), detection_result_flipped)

        cv2.imshow("yuz_x_ters", cv2.cvtColor(annotated_image_flipped, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            exit(0)        