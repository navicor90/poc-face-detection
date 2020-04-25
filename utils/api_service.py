import requests

HOST = "https://real-customer.herokuapp.com"


def request_record_face_data(encodings, locations, frame):
    url = HOST+'/customerEntrance'
    face_data = {'embedding_image': encodings}
             #'position': locations}
    #print(face_data)
    x = requests.post(url, json=face_data)
    print(x.text)
    response_ok = False
    if 200 <= x.status_code < 300:
        response_ok = True

    return response_ok
