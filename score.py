import numpy as np
import main_joint_angle

### group_size 만큼 슬라이싱 후 평균
def average_by_group(array, group_size):
    n = len(array)
    num_groups = n // group_size
    remainder = n % group_size

    # 배열의 길이가 group_size의 배수가 아닌 경우, 남은 부분은 제외
    array = array[:n - remainder]

    # 배열을 group_size 만큼 묶은 후, 각 그룹의 평균을 계산
    grouped_array = np.mean(array.reshape(num_groups, group_size), axis=1)

    return grouped_array

### EXP Data
def exp_data_extract(arr):              ###기준&정답 / 기준&오답 일 때 알고리즘 다시 생각...
    arr_EXP = np.exp(arr)
    print('arr_EXP : ', arr_EXP)

    ### 여러가지 실시영상을 넣어보며 적당한 상한선을 정해야겠지만,
    ### np.exp(7) = 1096.xx, np.exp(8) = 2980.xx 이기 때문에
    ### 일단 상한선을 2000으로 잡고 2000 이상의 값이면 2000으로 때려놓고 정규화.

    arr_EXP[arr_EXP >= 2000] = 2000
    print('arr_EXP max 2000 : ', arr_EXP)

    normalized_arr_EXP = arr_EXP / 20

    print('normalized_arr_EXP : ', normalized_arr_EXP)
    print('normalized_arr_EXP 평균 : ', np.mean(normalized_arr_EXP))

    ### EXP_ACC
    exp_acc = (100 - np.mean(normalized_arr_EXP))
    print('!!!정확도 : ', exp_acc)

    return exp_acc


def ACCURACY(dtw_sim_exp, DTW_Alpha, JA_Alpha):
    Real_Acc = (1 - DTW_Alpha) * dtw_sim_exp + (JA_Alpha * 100)

    return Real_Acc


# 여기부터 사용 안함?
def dtw_score(dtw):
    dtw_min = np.min(dtw)
    dtw_max = np.max(dtw)
    dtw_normalized_distance = ((dtw-dtw_min) / (dtw_max-dtw_min))
    x = (dtw_normalized_distance-0.5) * 7
    score = 1 / (1+np.exp(x))
    score *= 100
    return dtw_normalized_distance, score

def cos_score(cos):
    score = np.array(cos)
    score *= 100
    return score
