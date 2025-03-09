import cv2
import Processing


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.md = Processing.MotionDetector()

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break

            fgMask = self.md.background_subtraction(self.cap, frame)
            frame_ct = self.md.contour_detection(frame, fgMask)
            cv2.imshow("FG Mask", frame_ct)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
