import cv2

cap = cv2.VideoCapture(10)          #Realsense RGB
    
if not cap.isOpened():
    print("Realsense open failed")
    exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 1프레임과 다음 프레임 사이의 간격 설정
delay = round(1000/fps)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter("/home/unicon4/yolov5/output.avi", fourcc, fps, (w, h))

# 제대로 열렸는지 확인
if not out.isOpened():
    print('File open failed!')
    cap.release()
    sys.exit()
    
while True:
    ret,frame = cap.read()
    
    if not ret:
        print("Can't read IR camera")
        break
    
    cv2.imshow('frame',frame)
    out.write(frame)


        
    if cv2.waitKey(delay) == 27:
        break
        
#cap.release()
#cv2.destroyAllWindows()
