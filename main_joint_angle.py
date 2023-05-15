import cv2
import mediapipe as mp
import numpy as np
import os

# 주요 관절 8개 list
angle_elbow_r_idx = []
angle_shoulder_r_idx = []
angle_hip_r_idx = []
angle_knee_r_idx = []
angle_elbow_l_idx = []
angle_shoulder_l_idx = []
angle_hip_l_idx = []
angle_knee_l_idx = []


def init_media_pipe():
    # mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 동영상 파일 열기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "허리_정답.mp4"
    folder_name = "video"
    video_path = os.path.join(current_dir, folder_name, file_name)
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일의 프레임 수, 프레임 크기 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 출력 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('허리_정답_output.mp4', fourcc, fps, frame_size)

    return cap, mp_pose, mp_drawing, out

# def Shoulder_Cor(User_angle_shoulder_r_idx, Expert_max_R_shoulder, Expert_min_R_shoulder):
#     for i in range(len(User_angle_shoulder_r_idx)):
#         if User_angle_shoulder_r_idx[i] < Expert_min_R_shoulder and User_angle_shoulder_r_idx[i] > Expert_max_R_shoulder:
#             print('가동범위 맞아?? ㄹㅇ? ')
#             break
#     print('가동범위 ㅇㅈ ㄱㅊ음')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)


def Joint_List(cap, mp_pose, mp_drawing, out):
    # MediaPipe Pose 실행
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            # 프레임 읽기
            ret, frame = cap.read()

            if not ret:
                break

            # MediaPipe Pose 처리 전 BGR을 RGB로 변경
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 어깨 landmark 출력
            image_hight, image_width, _ = frame.shape
            if not results.pose_landmarks:
                continue

            # 결과 그리기
            annotated_frame = frame.copy()
            landmarks = results.pose_landmarks.landmark

            # 위치 값 받아오기
            # 오른쪽
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ancle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            # 왼쪽
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ancle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # 각도계산
            # 오른쪽
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle_shoulder_r = calculate_angle(hip_r, shoulder_r, elbow_r)
            angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
            angle_knee_r = calculate_angle(hip_r, knee_r, ancle_r)
            # 왼쪽
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_shoulder_l = calculate_angle(hip_l, shoulder_l, elbow_l)
            angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
            angle_knee_l = calculate_angle(hip_l, knee_l, ancle_l)

            # 주요 관절 8개 list append
            angle_elbow_r_idx.append(angle_elbow_r)
            angle_shoulder_r_idx.append(angle_shoulder_r)
            angle_hip_r_idx.append(angle_hip_r)
            angle_knee_r_idx.append(angle_knee_r)
            angle_elbow_l_idx.append(angle_elbow_l)
            angle_shoulder_l_idx.append(angle_shoulder_l)
            angle_hip_l_idx.append(angle_hip_l)
            angle_knee_l_idx.append(angle_knee_l)

            # 각도 시각화
            # 오른쪽
            cv2.putText(annotated_frame, str(angle_shoulder_r),
                        tuple(np.multiply(shoulder_r, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(annotated_frame, str(angle_elbow_r),
                        tuple(np.multiply(elbow_r, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(annotated_frame, str(angle_hip_r),
                        tuple(np.multiply(hip_r, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(annotated_frame, str(angle_knee_r),
                        tuple(np.multiply(knee_r, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            # 왼쪽
            cv2.putText(annotated_frame, str(angle_shoulder_l),
                        tuple(np.multiply(shoulder_l, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(annotated_frame, str(angle_elbow_l),
                        tuple(np.multiply(elbow_l, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(annotated_frame, str(angle_hip_l),
                        tuple(np.multiply(hip_l, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(annotated_frame, str(angle_knee_l),
                        tuple(np.multiply(knee_l, [image_width, image_hight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))

            # 결과 동영상 파일에 추가
            out.write(annotated_frame)

    angle_List = [[]*8]
    angle_List[0] = angle_elbow_r_idx
    angle_List[1] = angle_shoulder_r_idx
    angle_List[2] = angle_hip_r_idx
    angle_List[3] = angle_knee_r_idx
    angle_List[4] = angle_elbow_l_idx
    angle_List[5] = angle_shoulder_l_idx
    angle_List[6] = angle_hip_l_idx
    angle_List[7] = angle_knee_l_idx

    return angle_List


def save_joint_list():
    f = open('angle_어깨.txt', 'w')

    for i in range(len(angle_elbow_r_idx)):
        f.write('{{Frame : {} , r_elbow : {} , l_elbow : {} , r_shoulder : {} ,'
                'l_shoulder : {} , r_hip : {} , l_hip : {} , r_knee : {} , l_knee : {} }}\n'
        .format(
            i + 1, angle_elbow_r_idx[i], angle_elbow_l_idx[i], angle_shoulder_r_idx[i], angle_shoulder_l_idx[i],
                angle_hip_r_idx[i], angle_hip_l_idx[i], angle_knee_r_idx[i], angle_knee_l_idx[i]))

    f.close()

def Joint_Range_is_OK(exercise):   ### consider_joint 는 main_jointangle 파일의 Joint_Range_is_OK 함수에서 return T / F 값으로 도출.
    # consider_joint : 고려할 관절의 개수
    # TF : 관절이 가동 범위 내에 존재 하는지 아닌지
    # Alpha : 고려하는 관절에 따른 운동별 정확도 가중치

    # 초기화
    cap, mp_pose, mp_drawing, out = init_media_pipe()

    # 주요 관절 8개 list up
    angle_List = Joint_List(cap, mp_pose, mp_drawing, out)


    if exercise == 1:
        ### 해당 운동에서 주요하게 고려할 관절의 가동범위
        consider_joint = 1
        if angle_List[3] < 170 and angle_List[7] < 170:
            TF = 1
            print('TF 까지 들어옴')


    if consider_joint == 1:
        Alpha = 0.3
    elif consider_joint == 2:
        Alpha = 0.25
    elif consider_joint == 3:
        Alpha = 0.2


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return TF, Alpha


if __name__ == '__main__':
    # 초기화
    cap, mp_pose, mp_drawing, out = init_media_pipe()

    # 주요 관절 8개 list up
    angle_List = Joint_List(cap, mp_pose, mp_drawing, out)

    print('angle_List : ', angle_List)

    # 주요 관절 8개 list save
    #save_joint_list()





    cap.release()
    out.release()
    cv2.destroyAllWindows()
