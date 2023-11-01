from typing import Tuple
import numpy as np
import cv2
import random


class BackgroundRemover:
    def __init__(self) -> None:
        pass

    def find_background(self, video: np.ndarray, sample_size: int) -> np.ndarray:
        """Find the background of the random 500 frames by getting the median value for each pixel

        Args:
            frames (np.ndarray): a list of frames, generally the entire video
            sample_size (int): the number of frames to sample from the video. Must be less than the number of frames in the video

        Returns:
            np.ndarray: a single frame that represents the background
        """
        if sample_size > len(video):
            raise ValueError(
                f"Sample size {sample_size} must be less than the number of frames in the video {len(video)}"
            )
        if sample_size < 1:
            raise ValueError(f"Sample size {sample_size} must be greater than 0")

        background_image = np.median(random.sample(video, sample_size), axis=0).astype(
            dtype=np.uint8
        )
        cv2.imshow("Background Image", background_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return background_image

    # Delete background
    def delete_background(self, original, background):
        filtered = []
        for frame in original:
            filtered.append(cv2.absdiff(frame, background))
        return filtered


class NoiseReducer:
    def __init__(self) -> None:
        self.erosion_size = 0
        # max_elem = 2
        self.max_kernel_size = 21
        self.title_trackbar_element_type = (
            "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse"
        )
        self.title_trackbar_kernel_size = "Kernel size:\n 2n +1"
        self.title_erosion_window = "Erosion"
        self.title_dilatation_window = "Dilation"

    def apply_filter(
        self,
        original: np.ndarray,
        filter_type: str,
        kernel_size: int | Tuple[int, int],
        sigma: float | None = None,
    ):
        """Applies a 2D filter to each frame in a video. Gaussian filtering for original video. Median filtering for salt-and-pepper noise.

        Args:
            original (np.ndarray): A list of frames, generally the entire video
            filter_type (str): One of "median", "gaussian", or "bilateral"
            kernel_size (int | Tuple[int, int]): The size of the kernel to use for the filter. If filter_type is "median", this should be an int. Otherwise, it should be a tuple of ints.
            sigma (float | None, optional): The sigma value to use for the filter. Defaults to None.

        Returns:
            np.ndarray: A list of frames that have been filtered
        """
        if filter_type not in ["median", "gaussian", "bilateral"]:
            raise ValueError(f"Filter type {filter_type} not supported")

        elif filter_type != "median":
            if sigma is None:
                raise ValueError(f"Filter type {filter_type} requires sigma")
            elif isinstance(kernel_size, int):
                raise ValueError(
                    f"Filter type {filter_type} requires kernel size tuple (x, y)"
                )
        if filter_type == "median":
            if isinstance(kernel_size, tuple):
                raise ValueError(f"Filter type {filter_type} requires kernel size int")
            elif kernel_size % 2 == 0:
                raise ValueError("Median filter requires odd kernel size")
            elif sigma is not None:
                raise Warning("Median filter does not support sigma")

        filtered = []
        for frame in original:
            if filter_type == "median":
                filtered.append(cv2.medianBlur(frame, kernel_size))
            elif filter_type == "gaussian":
                filtered.append(cv2.GaussianBlur(frame, kernel_size, sigma))
            elif filter_type == "bilateral":
                filtered.append(cv2.bilateralFilter(frame, kernel_size, sigma, sigma))
            else:
                raise ValueError(f"Filter type {filter_type} not supported")
        return filtered

    # Thresholding
    def low_pass(self, orig):
        filtered = []
        for frame in orig:
            lp = frame > 250
            frame[lp] = 0
            filtered.append(frame)
        return filtered

    # Eroding
    def erosion(self, src, val):
        erosion_size = cv2.getTrackbarPos(
            self.title_trackbar_kernel_size, self.title_erosion_window
        )
        erosion_type = 0
        val_type = cv2.getTrackbarPos(
            self.title_trackbar_element_type, self.title_erosion_window
        )
        if val_type == 0:
            erosion_type = cv2.MORPH_RECT
        elif val_type == 1:
            erosion_type = cv2.MORPH_CROSS
        elif val_type == 2:
            erosion_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(
            erosion_type,
            (2 * erosion_size + 1, 2 * erosion_size + 1),
            (erosion_size, erosion_size),
        )
        erosion_dst = cv2.erode(src, element)
        cv2.imshow(self.title_erosion_window, erosion_dst)

    # Dilating
    def dilation(self, src, val):
        dilatation_size = cv2.getTrackbarPos(
            self.title_trackbar_kernel_size, self.title_dilatation_window
        )
        dilatation_type = 0
        val_type = cv2.getTrackbarPos(
            self.title_trackbar_element_type, self.title_dilatation_window
        )
        if val_type == 0:
            dilatation_type = cv2.MORPH_RECT
        elif val_type == 1:
            dilatation_type = cv2.MORPH_CROSS
        elif val_type == 2:
            dilatation_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(
            dilatation_type,
            (2 * dilatation_size + 1, 2 * dilatation_size + 1),
            (dilatation_size, dilatation_size),
        )
        dilatation_dst = cv2.dilate(src, element)
        cv2.imshow(self.title_dilatation_window, dilatation_dst)
