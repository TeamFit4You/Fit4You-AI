import mediapipe as mp
import cv2

# mediapipe를 활용해 동영상에서 포즈를 추출하는 함수
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_pose(video_path):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)# 동영상 파일 로드
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.resize(image, (640, 480))# 이미지 사이즈 조절
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)# 이미지 색상 조절
            image.flags.writeable = False
            results = pose.process(image)# 이미지에서 포즈 추출
            image.flags.writeable = True
            if results.pose_landmarks:
                yield results.pose_landmarks