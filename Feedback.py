import main_joint_angle as ja
import numpy as np
import json
import requests

# # JSON 요청 데이터 생성
# data = {
#     "key1": "value1",
#     "key2": "value2"
# }
#
# # JSON 형식으로 직렬화
# json_data = json.dumps(data)
#
# # 서버로 POST 요청 보내기
# url = "https://api.example.com"
# response = requests.post(url, data=json_data)
#
# # 응답 받은 JSON 데이터 해석
# response_data = response.json()
#
# # 응답 데이터 확인
# print(response_data)

# 응답으로 받은 JSON 데이터
response_data = '''
{
    "exercise_id": "1",
    "filepath": "ddd"
}
'''

# JSON 데이터 파싱
data = json.loads(response_data)

# 데이터 확인
print('exercise_id : ', data["exercise_id"])
print('filepath : ', data["filepath"])


# angle_List[0] = angle_elbow_r_idx
# angle_List[1] = angle_shoulder_r_idx
# angle_List[2] = angle_hip_r_idx
# angle_List[3] = angle_knee_r_idx
# angle_List[4] = angle_elbow_l_idx
# angle_List[5] = angle_shoulder_l_idx
# angle_List[6] = angle_hip_l_idx
# angle_List[7] = angle_knee_l_idx

exercise_id = data["exercise_id"]
expert_angle_list = [[132, 131, 130, 131, 131, 130, 129, 128, 128, 128, 128, 128, 127, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 128, 128, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 132, 132, 131, 131, 130, 130, 130, 130, 130, 130, 131, 131, 131, 132, 132, 132, 132, 132, 132, 132, 129, 130, 129, 130, 129, 129, 129, 129, 130, 130, 130, 129, 129, 129, 128, 128, 128, 129, 129, 130, 130, 131, 132, 132, 132, 132, 133, 133, 132, 132, 132, 133, 133, 135, 135, 136, 137, 135, 132, 132, 132, 132, 132, 132, 132, 131, 131, 131, 130, 129, 129, 129, 129, 130, 132, 133, 134, 134, 134, 134, 133, 132, 131, 130, 130, 131, 131, 131, 133, 135, 135, 134, 133, 134, 135, 135, 135, 135, 136, 136, 136, 136, 136, 135, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 136, 137, 137, 137, 137, 137, 138, 137, 137, 136, 136, 136, 136, 136, 135, 134, 134, 134, 134, 134, 134, 134, 134, 135, 136, 137, 137, 137, 137, 137, 137, 137, 137, 136, 136, 136, 137, 138, 138, 138, 138, 138, 139, 139, 139, 140, 140, 140, 140, 140, 140, 140, 139, 139, 139, 141, 142, 142, 142, 141, 141, 141, 141, 141, 141, 140, 140, 140, 140, 140, 140, 140, 140, 140, 141, 142, 143, 144, 145, 145, 146, 146, 146, 146, 146, 147, 147, 146, 145, 145, 145, 144, 143, 143, 143, 142, 142, 141, 141, 140, 142, 141, 142, 142, 146, 145, 145, 145, 145, 145, 143, 143, 144, 148, 147, 146, 145, 145, 144, 143, 142, 142, 140, 139, 139, 139, 139, 140, 140, 141, 140, 141, 140, 139, 139, 137, 137, 137, 137, 137, 136, 136, 136, 136, 135, 135, 135, 135, 135, 135, 135, 135, 134, 137, 138, 138, 137, 136, 137, 137, 137, 137, 137, 137, 136, 136, 135, 136, 136, 136, 136, 136, 136, 136, 136, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 136, 136, 136, 136, 136, 136], [25, 24, 25, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 20, 19, 19, 18, 19, 22, 21, 21, 21, 22, 22, 22, 21, 21, 22, 23, 23, 23, 23, 23, 24, 26, 26, 27, 26, 26, 26, 25, 25, 25, 25, 24, 24, 23, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 20, 20, 20, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 18, 19, 19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 18, 18, 16, 16, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 20, 20, 20, 18, 18, 18, 17, 16, 15, 15, 14, 14, 14, 13, 13, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 16, 17, 17, 16, 16, 15, 16, 13, 14, 14, 14, 14, 15, 16, 16, 16, 12, 13, 14, 15, 15, 16, 16, 16, 17, 18, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20], [175, 174, 175, 175, 175, 174, 175, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 174, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 171, 171, 171, 171, 170, 170, 169, 169, 168, 168, 168, 167, 167, 167, 167, 167, 167, 168, 167, 167, 167, 167, 166, 166, 166, 166, 166, 165, 165, 165, 165, 165, 165, 165, 165, 165, 164, 164, 164, 163, 163, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 161, 161, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 159, 159, 160, 160, 159, 159, 159, 159, 159, 158, 158, 158, 158, 158, 157, 158, 158, 158, 158, 158, 158, 157, 157, 157, 156, 156, 156, 157, 157, 157, 157, 156, 156, 156, 156, 156, 156, 155, 155, 155, 155, 155, 155, 154, 154, 154, 154, 153, 153, 153, 153, 153, 153, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 151, 151, 152, 152, 152, 152, 153, 153, 153, 153, 153, 154, 154, 154, 154, 155, 155, 155, 155, 155, 155, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 157, 157, 156, 156, 156, 156, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 156, 156, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 159, 159, 159, 160, 160, 161, 161, 161, 161, 162, 162, 162, 162, 163, 163, 163, 164, 163, 164, 165, 165, 166, 166, 167, 168, 167, 167, 168, 167, 167, 167, 168, 168, 168, 168, 168, 168, 168, 169, 169, 169, 170, 170, 171, 171, 172, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 176, 175, 175, 175, 175, 174, 174, 174, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 176, 175, 175], [177, 177, 177, 177, 177, 177, 176, 177, 177, 176, 176, 176, 176, 176, 176, 175, 175, 175, 175, 175, 175, 174, 174, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 176, 176, 176, 176, 176, 176, 176, 176, 176, 175, 176, 175, 176, 176, 176, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 175, 175, 174, 173, 173, 173, 172, 172, 172, 172, 172, 171, 171, 172, 172, 172, 171, 171, 171, 171, 170, 170, 171, 171, 171, 171, 171, 171, 171, 171, 171, 170, 171, 171, 172, 172, 172, 172, 173, 173, 173, 173, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 173, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 172, 172, 172, 172, 171, 171, 170, 170, 170, 169, 169, 169, 169, 169, 169, 168, 168, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 167, 167, 167, 167, 168, 168, 168, 168, 168, 168, 168, 168, 168, 167, 168, 168, 169, 169, 169, 170, 171, 171, 171, 171, 171, 171, 171, 171, 172, 172, 173, 173, 173, 173, 173, 174, 174, 176, 176, 176, 176, 175, 176, 176, 176, 177, 177, 177, 177, 178, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 176, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179], [113, 113, 112, 111, 111, 110, 110, 110, 109, 109, 109, 109, 109, 109, 109, 109, 108, 108, 108, 108, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 106, 106, 106, 106, 106, 106, 106, 107, 106, 107, 107, 107, 107, 107, 108, 108, 108, 109, 109, 110, 109, 109, 109, 109, 108, 109, 109, 109, 110, 110, 110, 110, 111, 112, 112, 112, 112, 113, 112, 113, 112, 112, 112, 113, 113, 114, 114, 113, 113, 112, 112, 113, 114, 114, 115, 115, 116, 116, 116, 117, 117, 118, 118, 117, 118, 118, 118, 119, 120, 120, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 124, 125, 125, 125, 125, 124, 124, 124, 123, 123, 122, 122, 122, 121, 121, 120, 120, 119, 119, 119, 119, 119, 119, 118, 118, 117, 117, 117, 118, 118, 118, 118, 117, 117, 116, 117, 118, 117, 117, 117, 116, 117, 118, 118, 118, 118, 118, 117, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 118, 118, 118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 118, 118, 118, 118, 118, 117, 117, 117], [39, 38, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 37, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 35, 35, 35, 36, 36, 37, 36, 36, 35, 35, 35, 35, 34, 33, 33, 33, 32, 32, 32, 32, 32, 32, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 30, 30, 30, 29, 29, 29, 28, 28, 28, 28, 27, 28, 28, 28, 28, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 26, 27, 26, 26, 26, 26, 26, 27, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 29, 29, 29, 29, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 33, 32, 32, 31, 31, 32, 33, 33, 33, 32, 32, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 36, 36, 36, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36], [177, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 178, 178, 178, 178, 178, 178, 179, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 177, 177, 177, 177, 177, 176, 176, 176, 176, 176, 175, 175, 175, 175, 175, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 172, 172, 172, 172, 171, 171, 171, 171, 170, 170, 170, 169, 169, 169, 169, 169, 168, 168, 168, 167, 167, 167, 166, 166, 166, 166, 166, 165, 165, 165, 165, 165, 165, 165, 164, 164, 164, 164, 163, 162, 162, 161, 161, 161, 161, 161, 161, 161, 161, 160, 160, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 158, 158, 158, 158, 158, 158, 158, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 156, 156, 156, 156, 156, 155, 155, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 153, 153, 153, 153, 153, 154, 154, 154, 154, 154, 154, 154, 154, 154, 155, 155, 155, 155, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158, 158, 158, 159, 160, 160, 160, 161, 161, 161, 162, 162, 163, 164, 165, 165, 165, 165, 166, 166, 167, 167, 167, 167, 168, 168, 169, 169, 170, 170, 170, 170, 171, 172, 172, 173, 173, 174, 174, 175, 175, 176, 176, 177, 177, 177, 177, 177, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178], [178, 176, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 174, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 175, 174, 173, 173, 173, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 171, 171, 171, 171, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 173, 174, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 172, 172, 172, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 171, 171, 171, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179]]
correct_angle_list = [[135, 134, 133, 134, 134, 133, 131, 130, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 129, 130, 131, 131, 132, 132, 132, 132, 132, 132, 132, 131, 131, 131, 131, 132, 132, 134, 134, 135, 135, 135, 134, 134, 134, 135, 133, 133, 133, 133, 134, 134, 133, 134, 134, 134, 135, 135, 135, 136, 136, 137, 137, 137, 137, 138, 138, 139, 139, 139, 139, 139, 139, 137, 137, 137, 136, 136, 137, 138, 139, 139, 140, 140, 140, 141, 141, 141, 141, 141, 142, 142, 142, 141, 140, 140, 140, 139, 138, 137, 138, 140, 140, 139, 138, 138, 138, 138, 139, 139, 139, 140, 140, 140, 140, 140, 141, 141, 141, 141, 142, 142, 142, 143, 144, 144, 144, 144, 144, 144, 144, 144, 144, 145, 144, 144, 143, 143, 143, 143, 143, 143, 144, 144, 143, 143, 143, 143, 143, 143, 143, 143, 143, 144, 144, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 144, 143, 144, 144, 144, 144, 144, 145, 145, 145, 145, 146, 146, 146, 146, 146, 146, 145, 144, 144, 144, 143, 143, 143, 144, 144, 144, 146, 146, 146, 147, 147, 147, 148, 148, 149, 149, 150, 150, 150, 149, 148, 147, 147, 147, 148, 148, 148, 149, 149, 149, 149, 147, 146, 146, 146, 145, 143, 145, 147, 149, 148, 147, 147, 144, 142, 140, 140, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 138, 138, 138, 138, 138, 138, 140, 141, 141, 141, 141, 141, 141, 141, 140, 140, 139, 139, 139, 139, 138, 138, 138, 139, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139], [21, 21, 21, 20, 20, 20, 21, 22, 23, 23, 23, 23, 23, 22, 22, 22, 22, 23, 23, 23, 23, 22, 21, 21, 20, 20, 20, 20, 20, 19, 20, 20, 20, 20, 20, 20, 20, 18, 17, 17, 16, 16, 16, 17, 17, 17, 18, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 17, 17, 16, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 15, 15, 14, 15, 15, 15, 15, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 11, 11, 11, 10, 11, 10, 11, 11, 11, 12, 12, 13, 13, 12, 11, 10, 10, 11, 11, 13, 15, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 19, 19, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16], [171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 171, 171, 170, 169, 169, 169, 168, 168, 168, 168, 168, 168, 168, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 165, 165, 165, 164, 164, 164, 164, 163, 163, 163, 161, 161, 162, 162, 161, 161, 161, 161, 161, 160, 160, 160, 161, 161, 161, 160, 160, 160, 160, 160, 160, 160, 161, 161, 161, 161, 161, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 160, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 158, 158, 158, 158, 158, 158, 158, 158, 158, 159, 159, 158, 158, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 158, 158, 159, 159, 159, 159, 159, 159, 159, 160, 160, 161, 161, 162, 162, 163, 163, 163, 164, 164, 165, 165, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 167, 166, 166, 166, 167, 167, 167, 167, 167, 167, 168, 168, 168, 169, 169, 169, 169, 169, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172], [178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 176, 176, 176, 177, 176, 176, 176, 176, 176, 177, 177, 176, 176, 176, 176, 176, 176, 176, 176, 176, 175, 175, 175, 175, 175, 174, 174, 173, 173, 173, 173, 173, 173, 172, 171, 172, 172, 171, 171, 171, 171, 172, 172, 171, 171, 171, 171, 171, 171, 171, 169, 169, 169, 168, 168, 168, 168, 168, 169, 169, 169, 169, 169, 169, 169, 169, 170, 170, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 172, 171, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 172, 173, 172, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178], [107, 107, 107, 107, 107, 107, 108, 108, 107, 107, 107, 106, 106, 106, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 106, 106, 107, 107, 108, 108, 108, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 110, 111, 111, 111, 111, 112, 113, 113, 113, 113, 114, 114, 114, 114, 114, 115, 116, 116, 117, 118, 119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 128, 128, 128, 128, 128, 127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 127, 127, 126, 126, 125, 125, 124, 123, 122, 121, 120, 120, 119, 119, 119, 119, 118, 118, 118, 118, 118, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 117, 117, 117, 117, 117, 117, 116, 116, 116, 116, 115, 115, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 114, 113, 113, 113, 113, 113, 113, 113, 113, 113, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112], [42, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 41, 41, 41, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 37, 37, 36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 35, 35, 35, 35, 35, 34, 34, 35, 35, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 35, 36, 36, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 36, 36, 37, 37, 37], [177, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 177, 177, 177, 177, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 174, 174, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 173, 173, 173, 173, 172, 172, 171, 170, 170, 170, 170, 170, 170, 170, 170, 169, 169, 169, 169, 168, 168, 168, 168, 168, 168, 167, 167, 167, 167, 166, 166, 166, 166, 166, 166, 166, 165, 165, 165, 165, 165, 165, 165, 165, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 165, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165, 166, 166, 167, 167, 168, 169, 169, 170, 170, 171, 172, 172, 173, 173, 173, 173, 174, 174, 174, 174, 174, 175, 175, 175, 175, 176, 176, 176, 177, 176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178], [179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 177, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 175, 175, 174, 174, 174, 174, 174, 174, 174, 173, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 173, 173, 173, 173, 172, 172, 172, 172, 171, 171, 172, 171, 171, 171, 171, 171, 171, 171, 171, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 169, 169, 169, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 170, 171, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176]]
wrong_angle_list = [[134, 138, 138, 137, 136, 137, 138, 138, 139, 139, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 140, 141, 141, 140, 141, 142, 142, 143, 140, 141, 142, 143, 143, 140, 141, 141, 141, 142, 142, 143, 143, 143, 144, 144, 144, 143, 143, 144, 144, 143, 143, 143, 143, 144, 144, 144, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 144, 143, 143, 142, 142, 142, 142, 141, 141, 141, 141, 142, 143, 143, 143, 143, 143, 143, 142, 142, 142, 142, 143, 142, 141, 141, 140, 141, 143, 144, 144, 145, 145, 145, 145, 145, 145, 144, 142, 142, 142, 141, 141, 142, 144, 145, 145, 146, 145, 146, 145, 144, 145, 145, 145, 143, 143, 144, 144, 144, 144, 145, 146, 147, 148, 149, 149, 149, 149, 150, 150, 150, 150, 151, 151, 152, 152, 151, 151, 151, 151, 152, 151, 150, 150, 150, 151, 150, 149, 150, 149, 148, 149, 149, 150, 150, 152, 151, 150, 149, 149, 148, 148, 148, 146, 146, 146, 149, 149, 149, 150, 148, 149, 150, 149, 149, 147, 146, 147, 147, 147, 146, 146, 145, 145, 144, 145, 145, 145, 142, 142, 141, 141, 141, 140, 139, 139, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139, 138, 138, 138, 138, 138, 138, 137, 137, 137, 137, 137], [23, 19, 18, 18, 19, 18, 17, 17, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 14, 14, 14, 15, 15, 14, 13, 12, 11, 11, 11, 11, 11, 11, 12, 12, 14, 15, 14, 15, 15, 14, 12, 12, 12, 11, 12, 11, 11, 11, 10, 11, 10, 12, 12, 12, 12, 13, 12, 12, 11, 10, 10, 9, 9, 9, 9, 8, 8, 8, 9, 9, 9, 8, 9, 10, 9, 9, 9, 9, 10, 10, 10, 9, 9, 10, 11, 11, 12, 13, 13, 13, 12, 12, 10, 10, 11, 12, 12, 13, 13, 13, 14, 15, 15, 13, 13, 12, 12, 12, 12, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17, 17], [176, 175, 175, 174, 174, 174, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 172, 171, 170, 169, 169, 169, 169, 168, 168, 168, 168, 168, 168, 168, 167, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 166, 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 166, 167, 167, 166, 166, 165, 164, 163, 163, 162, 162, 162, 162, 161, 161, 160, 159, 158, 155, 153, 152, 153, 149, 148, 147, 145, 144, 144, 143, 142, 141, 140, 140, 139, 138, 141, 139, 141, 140, 139, 142, 140, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 140, 142, 146, 143, 143, 143, 143, 141, 143, 143, 146, 146, 145, 146, 147, 147, 149, 150, 150, 150, 149, 149, 151, 152, 153, 154, 155, 155, 156, 157, 157, 157, 158, 159, 160, 160, 161, 161, 162, 163, 164, 165, 166, 167, 168, 168, 168, 168, 168, 169, 169, 169, 169, 169, 170, 170, 170, 170, 171, 171, 171, 171, 171, 172, 172, 173, 173, 174, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177], [179, 179, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 176, 176, 176, 176, 176, 176, 176, 175, 175, 175, 174, 173, 173, 172, 171, 170, 169, 167, 165, 161, 158, 157, 156, 155, 155, 155, 154, 153, 147, 149, 149, 149, 150, 149, 149, 148, 148, 147, 146, 145, 147, 147, 141, 142, 137, 139, 138, 132, 134, 135, 136, 135, 135, 134, 134, 134, 134, 135, 134, 134, 134, 134, 134, 134, 133, 129, 135, 136, 138, 139, 143, 142, 144, 142, 147, 149, 149, 150, 151, 150, 150, 152, 154, 159, 162, 162, 163, 163, 163, 164, 166, 170, 169, 171, 173, 175, 175, 177, 179, 179, 178, 178, 176, 176, 177, 176, 177, 176, 176, 175, 175, 175, 175, 175, 174, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 174, 175, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 178], [120, 118, 117, 117, 116, 116, 116, 116, 116, 116, 115, 115, 115, 115, 115, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 120, 120, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 122, 122, 122, 123, 124, 124, 124, 125, 126, 127, 127, 127, 128, 129, 129, 130, 131, 132, 132, 133, 132, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144, 145, 149, 150, 151, 150, 149, 148, 148, 149, 149, 149, 150, 151, 151, 151, 151, 150, 151, 151, 152, 154, 153, 154, 152, 153, 153, 153, 152, 151, 150, 150, 150, 148, 147, 145, 143, 143, 143, 142, 140, 138, 136, 135, 136, 135, 134, 134, 133, 132, 131, 130, 129, 128, 127, 125, 124, 123, 121, 119, 119, 119, 120, 118, 117, 116, 117, 116, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 117, 118, 118, 119, 119, 119, 119, 119, 119, 119, 119, 118, 118, 118, 118, 118, 117, 116, 116, 116, 115, 115, 115, 115, 115, 115], [36, 36, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 33, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 30, 31, 31, 30, 30, 29, 30, 30, 30, 30, 30, 29, 28, 28, 28, 28, 29, 28, 27, 27, 27, 27, 26, 26, 25, 25, 24, 23, 23, 22, 21, 20, 19, 18, 17, 17, 15, 13, 13, 13, 14, 15, 16, 15, 15, 14, 14, 13, 13, 13, 13, 13, 13, 13, 12, 13, 12, 12, 11, 13, 13, 13, 13, 14, 14, 15, 15, 15, 17, 17, 18, 19, 19, 19, 20, 21, 22, 23, 23, 22, 22, 23, 23, 23, 23, 24, 25, 25, 25, 26, 26, 27, 28, 28, 29, 29, 29, 28, 29, 30, 31, 30, 31, 31, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37], [176, 177, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 176, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 174, 174, 174, 173, 173, 173, 173, 173, 173, 173, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 171, 170, 170, 170, 169, 169, 169, 169, 169, 169, 169, 169, 170, 170, 170, 170, 170, 169, 169, 169, 169, 169, 168, 167, 166, 165, 164, 163, 162, 160, 159, 157, 156, 155, 154, 152, 152, 150, 149, 146, 144, 144, 142, 141, 139, 138, 136, 135, 134, 132, 132, 132, 132, 131, 130, 131, 131, 131, 132, 131, 131, 132, 131, 131, 131, 131, 131, 132, 132, 132, 133, 133, 134, 134, 134, 134, 133, 134, 133, 134, 135, 135, 136, 137, 137, 137, 139, 140, 141, 143, 145, 147, 148, 149, 150, 152, 153, 154, 157, 158, 159, 159, 160, 161, 162, 164, 165, 166, 167, 169, 170, 170, 171, 172, 173, 173, 173, 174, 174, 175, 175, 175, 175, 175, 175, 175, 176, 176, 177, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 179, 179, 179, 178, 178, 178, 178, 178, 178, 178, 178], [178, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 178, 177, 177, 177, 177, 177, 176, 176, 177, 177, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 175, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 175, 175, 175, 175, 174, 174, 174, 174, 175, 175, 174, 174, 173, 173, 173, 171, 170, 169, 168, 167, 167, 166, 165, 163, 162, 161, 160, 158, 156, 156, 158, 158, 158, 158, 158, 158, 158, 159, 159, 160, 161, 160, 158, 157, 156, 155, 152, 152, 150, 148, 149, 148, 146, 147, 147, 147, 148, 148, 147, 147, 147, 147, 147, 146, 147, 148, 150, 151, 152, 153, 155, 155, 157, 158, 158, 159, 161, 161, 162, 163, 163, 164, 165, 166, 166, 166, 166, 168, 170, 169, 169, 169, 170, 171, 172, 173, 174, 174, 175, 176, 175, 176, 177, 177, 178, 178, 179, 179, 179, 179, 179, 179, 178, 178, 178, 178, 178, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 177, 178, 178, 178, 178, 178, 178, 179, 179, 179]]

#min_angle

#
# if exercise_id == 1:
#     print('기준영상 & 정답영상 운동')
#
# elif exercise_id == 2:
#     print('기준영상 & 오답영상 운동')