from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def main() -> None:
    config = OpenCVCameraConfig(
        index_or_path="/dev/video1",
        fps=15,
        width=1920,
        height=1080,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG",
    )

    with OpenCVCamera(config) as camera:
        frame = camera.read()
        print("read() call returned frame with shape:", frame.shape)

        try:
            for i in range(10):
                frame = camera.async_read(timeout_ms=200)
                print(f"async_read call returned frame {i} with shape:", frame.shape)
        except TimeoutError as e:
            print(f"No frame received within timeout: {e}")

        try:
            initial_frame = camera.read_latest(max_age_ms=1000)
            for i in range(10):
                frame = camera.read_latest(max_age_ms=1000)
                print(f"read_latest call returned frame {i} with shape:", frame.shape)
                print(f"Was a new frame received by the camera? {not (initial_frame == frame).all()}")
        except TimeoutError as e:
            print(f"Frame too old: {e}")


if __name__ == "__main__":
    main()
