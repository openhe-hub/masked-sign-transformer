import pickle
import os
import numpy as np

# params
input_folder = 'data/pkl_files'
output_folder = 'data/chatsign_200'

def handle_one_frame(data):
    body_kps = data['bodies']['candidate']  # (18, 2)
    face_kps = data['faces'][0] # (68, 2)
    hand_kps = data['hands'].reshape(42, 2) # (42, 2)

    all_kps = np.vstack([body_kps, face_kps, hand_kps])
    return all_kps

def handle_one_pkl(pkl):
    length = len(pkl)
    result = []
    for i in range(length):
        result.append(handle_one_frame(pkl[i]))
    return result

if __name__ == "__main__":
    for file in os.listdir(input_folder):
        print(f'handling {file}')
        with open(f'{input_folder}/{file}', 'rb') as fp:
            pkl = pickle.load(fp)
        pkl = handle_one_pkl(pkl)
        with open(f'{output_folder}/{file}', 'wb') as fp:
            pkl = pickle.dump(pkl, fp)

