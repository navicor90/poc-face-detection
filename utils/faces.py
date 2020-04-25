from face_recognition import face_locations, face_encodings
import cv2
from scipy.spatial.distance import cosine


def faces_from_frame(frame):
    fl = face_locations(frame)
    fe = face_encodings(frame, fl)
    return fl, fe


def is_match(known_embedding, candidate_embedding, thresh=0.1):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False


def numpy_encodings_to_python_list(encodings):
    listed_encodings = []
    for e in encodings:
        listed_encodings.append(e.tolist())
    return listed_encodings


def exit_keys_pressed():
    return cv2.waitKey(1) & 0xFF == ord('q')


def debug_faces(frame, face_locs):
    # Draw a rectangle around the faces
    for (top, right, bottom, left) in face_locs:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)


class Camera:
    _video_capture = None

    def __init__(self):
        pass

    def rgb_frame_from_camera(self):
        self._video_capture = cv2.VideoCapture(0)
        ret, frame = self._video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._video_capture.release()
        return rgb_frame

    def release(self):
        self._video_capture.release()
        cv2.destroyAllWindows()
