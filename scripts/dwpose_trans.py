import pickle
import os
import numpy as np

# params
input_folder = 'data/pkl_wo_retarget'
output_folder = 'data/chatsign_200_wo_retarget'

def handle_one_frame(data):
    if data['hands'].shape[0] > 2 or data['hands_score'].shape[0] > 2: return None

    body_kps = np.hstack([data['bodies']['candidate'], data['bodies']['score'].T])
    face_kps = np.hstack([data['faces'][0], data['faces_score'].T])
    hand_kps = np.hstack([data['hands'].reshape(42, 2), data['hands_score'].reshape(42, 1)])

    all_kps = np.vstack([body_kps, face_kps, hand_kps])
    return {
        'keypoints': all_kps,
        'subset': data['bodies']['subset']
    }

def handle_one_pkl(pkl):
    length = len(pkl)
    result = []
    for i in range(length):
        all_kps = handle_one_frame(pkl[i])
        if all_kps is not None: result.append(all_kps)
    return result

if __name__ == "__main__":
    for file in os.listdir(input_folder):
        print(f'handling {file}')
        with open(f'{input_folder}/{file}', 'rb') as fp:
            pkl = pickle.load(fp)
        pkl = handle_one_pkl(pkl)
        with open(f'{output_folder}/{file}', 'wb') as fp:
            pkl = pickle.dump(pkl, fp)

