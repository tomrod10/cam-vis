import os
import cv2
import processing

file_path = "test_video.mov"
abs_path = os.path.abspath(file_path)


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(abs_path)
        self.md = processing.MotionDetector()

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break

            fg_mask = self.md.background_subtraction(self.cap, frame)
            _, mask_th = self.md.threshold(fg_mask.copy())
            mask_ed = self.md.erosion_dilation(mask_th)
            frame_cnt = self.md.contour_detection(frame, mask_ed)

            cv2.imshow("threshold", mask_th)
            cv2.imshow("contour", frame_cnt)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
