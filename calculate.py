from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
import pandas as pd
from dtaidistance import dtw

def calculate_euclidean_distance(pose1_vec, pose2_vec):
    return euclidean(pose1_vec.flatten(), pose2_vec.flatten())

def calculate_cosine_similarity(pose1_vec, pose2_vec):
    return np.dot(pose1_vec.flatten(), pose2_vec.flatten()) / (np.linalg.norm(pose1_vec.flatten()) * np.linalg.norm(pose2_vec.flatten()))

# 두 포즈 벡터 간 DTW(Dynamic Time Warping) 거리 계산 함수
def calculate_dtw_similarity(pose1_vec, pose2_vec):
    distance, path = fastdtw(pose1_vec, pose2_vec, dist=euclidean)
    return distance, path
