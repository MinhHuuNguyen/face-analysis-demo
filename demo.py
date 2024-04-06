import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from deepface import DeepFace
from deepface.modules.verification import find_cosine_distance


# Setting up mediapipe instances
face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5
)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
    model_selection=1
)
# face_stylizer = mp.tasks.vision.FaceStylizer.create_from_model_path(
#     model_path='face_stylizer_color_ink.task'
# )

# Streamlit title
st.set_page_config(page_title="Face Tracking Demo", layout="wide")
st.title("Face Tracking Demo")
col1, col2 = st.columns([4, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    run_face_detection = st.checkbox("Face Detection", value=True)
    run_emotion_analysis = st.checkbox("Emotion Analysis", value=False)
    run_age_analysis = st.checkbox("Age Analysis", value=False)
    run_gender_analysis = st.checkbox("Gender Analysis", value=False)
    run_race_analysis = st.checkbox("Race Analysis", value=False)
    run_face_mesh = st.checkbox("Face Mesh", value=False)
    run_background_removal = st.checkbox("Background Removal", value=False)
    # run_stylizer = st.checkbox("Face Stylizer", value=False)
    run_face_recognition = st.checkbox("Face Recognition", value=False)
    analysis_placeholder = st.empty()
    recognition_placeholder = st.empty()

# Prepare some constants
bg_image = None
BG_COLOR = (192, 192, 192) # gray
face_recognition_model = "Facenet"
face_recognition_threshold = 0.5
init_verifications_vector = DeepFace.represent(
    img_path="minh.jpg",
    model_name=face_recognition_model,
    enforce_detection=False
)[0]['embedding']


# Capture video from the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        st.write("Error accessing webcam.")
        break

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if run_face_detection:
        # Perform face detection
        results_detection = face_detection.process(image)
        if results_detection.detections is None:
            continue

        for detection in results_detection.detections:
            mp.solutions.drawing_utils.draw_detection(image, detection)
            # Assuming detection.location_data.relative_bounding_box gives the bounding box
            # Convert relative_bounding_box to pixel coordinates for cropping
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            # if run_stylizer:
            #     mp_face_image = mp.Image(
            #         image_format=mp.ImageFormat.SRGB,
            #         data=np.array(face_image)
            #     )
            #     face_stylizer_result = face_stylizer.stylize(mp_face_image)
            #     if face_stylizer_result is None:
            #         continue
            #     face_stylizer_result= face_stylizer_result.numpy_view()[:,:,:-1]
            #     face_stylizer_result = cv2.resize(face_stylizer_result, (bbox[2], bbox[3]))
            #     image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = face_stylizer_result

            if run_emotion_analysis or \
                run_age_analysis or \
                run_gender_analysis or \
                run_race_analysis or \
                run_face_recognition:

                actions = []
                if run_emotion_analysis:
                    actions.append("emotion")
                if run_age_analysis:
                    actions.append("age")
                if run_gender_analysis:
                    actions.append("gender")
                if run_race_analysis:
                    actions.append("race")
                if run_face_recognition:
                    checking_verifications_vector = DeepFace.represent(
                        img_path=face_image, 
                        model_name=face_recognition_model,
                        enforce_detection=False
                    )[0]['embedding']
                    distance = find_cosine_distance(
                        init_verifications_vector, 
                        checking_verifications_vector
                    )
                    if distance < face_recognition_threshold:
                        recognition_placeholder.text("This is Minhhh...")
                    else:
                        recognition_placeholder.text("This is not Minhhh...")

                try:
                    if len(actions) == 0:
                        continue
                    # Analyze the cropped face for multiple attributes
                    analysis = DeepFace.analyze(
                        img_path=face_image,
                        actions=actions,
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True
                    )[0]
                except Exception as e:
                    st.write("Error in deepface analysis:", e)
                
                # Display the analysis results on the image
                text = ""
                if run_emotion_analysis:
                    text += f"--{analysis['dominant_emotion']}--"
                if run_age_analysis:
                    text += f"--{analysis['age']}--"
                if run_gender_analysis:
                    text += f"--{analysis['dominant_gender']}--"
                if run_race_analysis:
                    text += f"--{analysis['dominant_race']}--"
                analysis_placeholder.text(text)

    if run_face_mesh:
        # Perform face mesh
        results_mesh = face_mesh.process(image)
        # Draw face mesh annotations on the image
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        thickness=1, circle_radius=1, color=(0,255,0)
                    )
                )
    
    if run_background_removal:
        # Perform selfie segmentation
        results_seg = selfie_segmentation.process(image)
        condition = np.stack((results_seg.segmentation_mask,) * 3, axis=-1) > 0.1
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        image = np.where(condition, image, bg_image)

    # Display the processed image in the Streamlit app
    frame_placeholder.image(image, channels="RGB", use_column_width=True)

# Release the webcam on app closure
cap.release()
