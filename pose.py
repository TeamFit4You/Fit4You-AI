import cv2
import mediapipe as mp
import numpy as np

def pose_drawing(video_path, output_path):
    # mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    # 동영상 파일의 프레임 수, 프레임 크기 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 출력 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            # 프레임 읽기
            ret, frame = cap.read()

            if not ret:
                break

            # MediaPipe Pose 처리 전 BGR을 RGB로 변경
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            image_hight, image_width, _ = frame.shape
            if not results.pose_landmarks:
                continue

            # 결과 그리기
            annotated_frame = frame.copy()

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))



            # 결과 동영상 파일에 추가
            out.write(annotated_frame)
            # 결과 출력
            cv2.imshow("MediaPipe Pose", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
