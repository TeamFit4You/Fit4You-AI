import main_joint_angle
import score
import pose
import cv2
import mediapipe as mp
import calculate
import extract_pose_mod
import numpy as np
import os
import json
import requests


def Extract_Pose_Data(video_path1, video_path2):
    #각 동영상에서 추출한 pose_data 저장할 리스트
    pose_data1 = []
    pose_data2 = []

    # 두 동영상에서 각 프레임마다 포즈를 추출하여 pose_data1, pose_data2 리스트에 저장
    for pose_landmarks1, pose_landmarks2 in zip(extract_pose_mod.extract_pose(video_path1), extract_pose_mod.extract_pose(video_path2)):
        if pose_landmarks1 is not None and pose_landmarks2 is not None:
            pose_data1.append(pose_landmarks1)
            pose_data2.append(pose_landmarks2)

    return pose_data1, pose_data2


def DTW_similarity(pose_data1, pose_data2):
    # DTW 거리를 이용할 리스트
    dtw_similarities = []

    #코사인 유사도를 이용할 리스트
    #cosine_similarities = []

    # 두 동작의 포즈 데이터에서 유효한 랜드마크(x,y,z 값을 가지고 있는)만 추출하여 numpy 배열로 변환
    # 각 랜드마크의 좌표를 벡터 형태로 저장하여 pose1_vec, pose2_vec에 할당
    for pose1, pose2 in zip(pose_data1, pose_data2):
        if pose1 is not None and pose2 is not None:
            pose1_vec = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose1.landmark if
                                  landmark.HasField('x') and landmark.HasField('y') and landmark.HasField('z')])
            pose2_vec = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose2.landmark if
                                  landmark.HasField('x') and landmark.HasField('y') and landmark.HasField('z')])

            #euclidean_distance = calculate_euclidean_distance(pose1_vec, pose2_vec)
            #cosine_similarity = calculate.calculate_cosine_similarity(pose1_vec, pose2_vec)
            # print('pose1_vec', pose1_vec)
            # print('pose2_vec', pose2_vec)

            dtw_similarity, path = calculate.calculate_dtw_similarity(pose1_vec, pose2_vec)
            #euclidean_distances.append(euclidean_distance)
            #cosine_similarities.append(cosine_similarity)

            # 계산된 DTW 유사도 값을 리스트에 추가
            dtw_similarities.append(dtw_similarity)

    return dtw_similarities


def ACC_by_sim(dtw_similarities):
    # Using dtw_similarities / Int (20) = 20개의 list 데이터 평균치
    grouped_array = score.average_by_group(np.array(dtw_similarities), 20)
    print('grouped_array : ', grouped_array)
    print('grouped_array len : ', len(grouped_array))

    exp_acc = score.exp_data_extract(grouped_array)

    # Using cosine_similarities
    # score_cos = score.cos_score(cosine_similarities)

    return exp_acc


# def Joint_Range_is_OK():
#
#     return  #True/False


def save_similarity(score_dtw):                  ### 수정 필요. dtw_similarities, exp_acc 둘 다 각각 저장되도록
    f = open('similarity.txt', 'w')

    for i in range(len(score_dtw)):
        f.write('{{Frame : {} , similarity : {:.2f} }}\n'
        .format(i + 1, score_dtw[i]))

    print("dtw평균 유사도: ", np.mean(score_dtw))
    # pose.pose_drawing(video_path1, output_path1)
    # pose.pose_drawing(video_path2, output_path2)


if __name__ == '__main__':
    # 영상 가져오기 / video_path1 = 전문가 영상, video_path2 = 사용자 영상
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # video_path1 = 전문가 영상
    file_name = "허리_기준.mp4"
    print("전문가 영상 : {}".format(file_name))
    folder_name = "video"
    video_path1 = os.path.join(current_dir, folder_name, file_name)

    # video_path2 = 사용자 영상
    file_name = "허리_정답.mp4"
    print("사용자 영상 : {}".format(file_name))
    folder_name = "video"
    video_path2 = os.path.join(current_dir, folder_name, file_name)

    # 프레임별로 자른 영상 데이터(x,y,z,vis) 추출
    Expert_pose_data, User_pose_data = Extract_Pose_Data(video_path1, video_path2)

    # 관절 범위 정확도.
    print('운동 종목 : ')
    exercise = int(input())  # 운동 종류 입력

    #================================= input : exercise_id, filepath

    TF, consider_joint, alpha, feedback = main_joint_angle.Joint_Range_is_OK(exercise, video_path2)
    DTW_Alpha = consider_joint * alpha
    JA_Alpha = TF * alpha

    # # 임시 코드 / exercise = 1 : 정답, exercise = 2 : 오답
    # # exercise = 3 : 완전히 다른 영상, exercise = 4 : 완전히 다른 영상인데 고려할 각도 범위는 만족한 영상
    # if exercise == 1:
    #     joint_within_range, alpha = 1, 0.3
    # elif exercise == 2:
    #     joint_within_range, alpha = 0, 0.3
    # elif exercise == 3:
    #     joint_within_range, alpha = 0, 0.3
    # elif exercise == 4:
    #     joint_within_range, alpha = 1, 0.3

    # DTW similarity Data
    dtw_similarity = DTW_similarity(Expert_pose_data, User_pose_data)
    print('dtw_similarity : ', dtw_similarity)
    print('dtw_similarity len : ', len(dtw_similarity))

    # EXP 적용 정확도. 정답은 더 정답으로 오답은 더 오답으로
    exp_acc = ACC_by_sim(dtw_similarity)
    print('exp_acc : ', exp_acc)

    # 정확도 출력
    ACC = score.ACCURACY(exp_acc, DTW_Alpha, JA_Alpha)
    print("최종 정확도 : {}".format(ACC))

    data = {
        "accuracy": ACC,
        "feedback": feedback
    }

    json_data = json.dumps(data)

    url = "http://example.com/api"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code == 200:
        print("데이터 전송 성공")
    else:
        print("데이터 전송 실패")
    #================================= output : Accuracy, Feedback



# # 영상 출력하기
#output_path1 = '../1_output.mp4'
#output_path2 = '../2_output.mp4'
