import numpy as np
import cv2


class MotionDetector:
    """
    Motion detector object. From the internet. 
    """

    def __init__(self, default=False, threshold=10, tolerance=200, kernel_size=7, erosions=3):
        self.default = default
        self.threshold = threshold
        self.tolerance = tolerance
        self.kernel = np.ones(kernel_size, np.uint8)
        self.erosion_iterations = erosions
        self.frames = []

        self.gray = None
        self.diff = None
        self.threshed = None
        self.denoised = None
        self.total_diff = 0
        self.is_moving = False

    def detect(self, frame):
        """
        Motion motion_detector

        Takes a frame, compares to previous frame received to check for change, performs some noise cleanup, then
        returns a boolean indicating whether the two frames are sufficiently different to be constitute movement.
        """
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frames.insert(0, self.gray)

        if len(self.frames) < 2:
            return self.default

        while len(self.frames) > 2:
            del self.frames[-1]

        self.diff = cv2.absdiff(self.frames[0], self.frames[1])
        _, self.threshed = cv2.threshold(
            self.diff, self.threshold, 255, cv2.THRESH_BINARY)
        self.denoised = cv2.erode(
            self.threshed, self.kernel, iterations=self.erosion_iterations)
        self.total_diff = cv2.countNonZero(self.denoised)

        self.is_moving = self.total_diff > self.tolerance

        return self.is_moving

    def reset_defaults(self):
        self.default = False
        self.threshold = 10
        self.tolerance = 200
        self.kernel = np.ones(7, np.uint8)
        self.erosion_iterations = 3
