import sys
from utils.faces import faces_from_frame, numpy_encodings_to_python_list, exit_keys_pressed, debug_faces, Camera, is_match
from utils.api_service import request_record_face_data
import time


def main(sleep_time=1, debug_mode=False):
    camera = Camera()
    last_embedding = []

    while not exit_keys_pressed():
        time.sleep(sleep_time)
        frame = camera.rgb_frame_from_camera()

        locations, encodings = faces_from_frame(frame)
        if debug_mode:
            debug_faces(frame, locations)
        encodings = numpy_encodings_to_python_list(encodings)

        are_different_faces = False
        if len(locations) > 0 and len(last_embedding) > 0:
            are_different_faces = not is_match(encodings[0], last_embedding)

        if are_different_faces or len(last_embedding) == 0:
            request_record_face_data(encodings[0], locations, frame)

        last_embedding = encodings[0] if len(locations) > 0 else last_embedding

    # When everything is done, release the capture
    camera.release()


if __name__ == "__main__":
    debug = False
    debug = True
    if len(sys.argv) > 1:
        sleep_time = int(sys.argv[1])
        main(sleep_time, debug_mode=debug)
    else:
        main(debug_mode=debug)
