import cv2


class MotionDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def alert(self):
        print("Motion detected!")

    def background_subtraction(self, cap, frame):
        fgMask = self.backSub.apply(frame)
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            (15, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        return fgMask

    def contour_detection(self, frame, fg_mask):
        cont, hier = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_ct = cv2.drawContours(frame, cont, -1, (0, 255, 0), 2)
        return frame_ct

    def threshold(self, frame): ...

    def gaussian_blur(self): ...
