import cv2


class MotionDetector():
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def alert(self):
        print("Motion detected!")

    def detect(self):
        ...

    def background_subtraction(self, cap, frame):
        fgMask = self.backSub.apply(frame)
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        return fgMask

