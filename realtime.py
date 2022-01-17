import cv2
from motion import MotionDetector
from style_transfer import StyleTransfer
from threading import Thread
import sys
import getopt


class RealtimeTransfer():
    def __init__(self, src) -> None:
        self.detector = MotionDetector(default=True, tolerance=5)
        self.cap = cv2.VideoCapture(0)
        W, H = 300, 300
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        cv2.namedWindow('BGR')
        self.B = 0
        self.G = 0
        cv2.createTrackbar("Blending", "BGR", self.B,
                           100, self.set_sliders('B'))
        cv2.createTrackbar("Detail", "BGR", self.G, 100, self.set_sliders('G'))
        self.st = StyleTransfer()
        self.st.initialize_reference(src)
        self.grabbed, self.frame = self.cap.read()
        #self.t = Thread(target=self.update, args=())
        # self.t.daemon = True  # daemon threads run in background

    def set_sliders(self, name):
        def _set_sliders(val):
            if name == "B":
                self.B = val
            else:
                self.G = val

        return _set_sliders

    def start(self):
        self.stopped = False
        self.t.start()
    # method passed to thread to read next available frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.cap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def transfer(self):
        while True:
            _, image = self.cap.read()
            isMoving = self.detector.detect(image)
            b = self.B / 100
            d = self.G / 100

            if isMoving:
                try:
                    self.st.update_face(image, "raw")
                    self.st.warp_img('affine')
                    self.st.process_reference_layers()

                    image = self.st.transform("", "", b, d, 1 - d)
                except:
                    print('bla')

            cv2.imshow('BGR', image)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()

    def main(self, argv):
        ref = "poses/makeup_tryon.jpeg"

        opts, args = getopt.getopt(
            argv,
            "a:g:s:r:w:d:p",
            [
                "alpha=",
                "gamma=",
                "ref=",
            ],
        )

        for curr_opt, curr_arg in opts:
            if curr_opt == "-a" or curr_opt == "--alpha":
                alpha = float(curr_arg)
                self.B = alpha * 100

            if curr_opt == "-g" or curr_opt == "--gamma":
                gamma = float(curr_arg)
                self.G = gamma * 100

            if curr_opt == "--ref":
                ref = curr_arg

        self.st.initialize_reference(ref)
        self.transfer()


if __name__ == "__main__":
    rt = RealtimeTransfer("poses/makeup_tryon.jpeg")
    rt.main(sys.argv[1:])
