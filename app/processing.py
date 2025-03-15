import cv2


class MotionDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def alert(self):
        print("Motion detected!")

    def background_subtraction(self, cap, frame):
        fg_mask = self.backSub.apply(frame)
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            (15, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        return fg_mask

    def threshold(self, fg_mask):
        ret, mask_th = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
        return (ret, mask_th)

    def erosion_dilation(self, mask_th):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        mask_ed = cv2.morphologyEx(
            mask_th, cv2.MORPH_OPEN, kernel, iterations=2
        )
        return mask_ed

    def contour_detection(self, frame, mask_ed):
        cnt, hier = cv2.findContours(
            mask_ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filter_cnt = self.filter_contour(cnt)
        frame_cnt = cv2.drawContours(frame, filter_cnt, -1, (0, 255, 0), 2)
        return frame_cnt

    def filter_contour(self, contours):
        min_cnt_area = 500
        large_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_cnt_area
        ]
        return large_contours
