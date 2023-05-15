import numpy as np
import main_joint_angle

### group_size 만큼 슬라이싱 평균
def average_by_group(array, group_size):
    n = len(array)
    num_groups = n // group_size
    remainder = n % group_size

    # 배열의 길이가 group_size의 배수가 아닌 경우, 남은 부분은 제외합니다.
    array = array[:n - remainder]

    # 배열을 group_size 만큼 묶은 후, 각 그룹의 평균을 계산합니다.
    grouped_array = np.mean(array.reshape(num_groups, group_size), axis=1)

    return grouped_array

### EXP Data
def exp_data_extract(arr, joint_range_acc):              ###기준&정답 / 기준&오답 일 때 알고리즘 다시 생각...
    arr_EXP = np.exp(arr)

    ### EXP_ACC
    lim = 0
    for i in range(len(arr_EXP)):
        if arr_EXP[i] > lim:
            lim = arr_EXP[i]

    # print('20프레임 : ', arr_EXP)
    # print('20avg_arr_EXP AVG : ', np.mean(arr_EXP))
    # print('lim : ', lim)

    arr_acc = (lim - arr_EXP) * 0.1

    # print('20프레임 별 정확도 : ', arr_acc)
    # print('20프레임 별 정확도의 평균 : ', np.mean(arr_acc))

    if joint_range_acc == True:
        exp_acc = 100 - np.mean(arr_acc)
    elif joint_range_acc == False:
        exp_acc = np.mean(arr_acc)

    return exp_acc


def ACCURACY(dtw_sim_exp, joint_within_range, Alpha):
    Real_Acc = (1 - Alpha) * dtw_sim_exp + (Alpha * 100) * joint_within_range

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
