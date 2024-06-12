import cv2 as cv
import numpy as np
import keyboard
from djitellopy.tello import Tello
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from enum import Enum
from threading import Thread


class TelloState(Enum):
    AVAIL = 0
    FLYING = 1
    LANDING = 2


class TelloMission(Enum):
    NONE = 0
    DETECT = 1
    FLY = 2


class TelloDrone:
    def __init__(self) -> None:
        self.drone = Tello()
        self.drone.connect()

        self.state = TelloState.AVAIL
        self.mission = TelloMission.NONE
        self.circle_count = 0
        self.circle_params = []

        self.middle_x = None
        self.drone_frame = None
        self.drone.streamon()


    def __del__(self):
        self.drone.streamoff()
        cv.destroyAllWindows()
        self.drone.end()

    def check_battery(self) -> int:
        return self.drone.get_battery()


    def check_temperature(self) -> float:
        return self.drone.get_temperature()


    def fly(self, x: int, y: int, z: int, yaw: int) -> None:
        self.mission = TelloMission.FLY if self.circle_count == 2 else TelloMission.DETECT
        self.drone.send_rc_control(y, x, z, yaw)


    def Move_forward(self, dist: int) -> None:
        self.drone.move_forward(dist)


    def Set_speed(self, speed: int) -> None:
        self.drone.set_speed(speed)


    def Takeoff(self) -> None:
        self.state = TelloState.FLYING
        self.mission = TelloMission.DETECT
        self.drone.takeoff()


    def Land(self) -> None:
        self.mission = TelloMission.NONE
        self.state = TelloState.LANDING
        self.drone.land()
        self.state = TelloState.AVAIL


    def show_camera(self, delay: int = 1) -> None:
        HEIGHT, WIDTH = self.drone_frame.shape[0:2]
        CYAN, RED, GREEN, SCALE, FONT, THICKNESS = (150, 150, 10), (0, 0, 255), (
        0, 255, 0), 2.5, cv.FONT_HERSHEY_PLAIN, 4

        state_pos, mission_pos = (int(0.01 * WIDTH), int(0.05 * HEIGHT)), (int(0.01 * WIDTH), int(0.12 * HEIGHT))
        circle_count_pos, temp_pos = (int(0.01 * WIDTH), int(0.19 * HEIGHT)), (int(0.75 * WIDTH), int(0.05 * HEIGHT))
        bat_pos, land_pos = (int(0.83 * WIDTH), int(0.12 * HEIGHT)), (int(0.36 * WIDTH), int(0.9 * HEIGHT))

        state_text, mission_text = f"STATE: {self.state.name}", f"MISSION: {self.mission.name}"
        circle_count_text, temp_text = f"COUNT: {self.circle_count}", f"TEMP: {self.check_temperature()}"
        bat_text, land_text = f"BAT: {self.check_battery()}", "LAND ASAP"

        cv.putText(self.drone_frame, state_text, state_pos, FONT, SCALE, CYAN, THICKNESS)

        cv.putText(self.drone_frame, mission_text, mission_pos, FONT, SCALE, CYAN, THICKNESS)

        cv.putText(self.drone_frame, circle_count_text, circle_count_pos, FONT, SCALE, CYAN, THICKNESS)

        # If critical value of temperature or battery is exceeded, show it in red

        if self.check_temperature() < 90:
            cv.putText(self.drone_frame, temp_text, temp_pos, FONT, SCALE, CYAN, THICKNESS)

        else:
            cv.putText(self.drone_frame, temp_text, temp_pos, FONT, SCALE, RED, THICKNESS)
            cv.putText(self.drone_frame, land_text, land_pos, FONT, 3.0, RED, THICKNESS)

        if self.check_battery() > 20:
            cv.putText(self.drone_frame, bat_text, bat_pos, FONT, SCALE, CYAN, THICKNESS)

        else:
            cv.putText(self.drone_frame, bat_text, bat_pos, FONT, SCALE, RED, THICKNESS)
            cv.putText(self.drone_frame, land_text, land_pos, FONT, 3.0, RED, THICKNESS)

        # Draw circles on the current frame
        self.draw_circles()

        # Draw a horizontal and vertical line between the only two circles
        self.draw_lines()

        # Show current frame with all annotations
        cv.imshow("Drone Camera", self.drone_frame)
        cv.waitKey(delay = delay)


    def detect_circles(self, model: YOLO) -> None:
        frame_handle = self.drone.get_frame_read()
        self.drone_frame = frame_handle.frame
        self.state = TelloMission.DETECT

        self.circle_count = 0
        self.circle_params = []
        self.middle_x = None

        # Canny by default returns a single-channel image

        canny = cv.Canny(self.drone_frame, threshold1=250, threshold2=250)
        canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        # Predict on the canny image (3-channel)

        results = model(canny, conf = 0.6)

        for result in results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes_xyxy, boxes_conf):
                x1, y1, x2, y2 = box[:]
                conf = round(float(conf), 2)

                radius_x = 0.5 * (x2 - x1)
                radius_y = 0.5 * (y2 - y1)
                radius = round(0.5 * (radius_x + radius_y))

                center_x = round(x1 + radius_x)
                center_y = round(y1 + radius_y)

                self.circle_params.append((center_x, center_y, radius, conf))

        self.circle_count = len(self.circle_params)

        if self.circle_count == 2:
            center_1x = self.circle_params[0][0]
            center_2x = self.circle_params[1][0]

            self.middle_x = round(0.5 * (center_1x + center_2x))


    def draw_circles(self) -> None:
        for circle in self.circle_params:
            center_x, center_y, radius, conf = circle[:]
            cv.circle(self.drone_frame, (center_x, center_y), radius, (255, 255, 0), 2)


    def draw_lines(self) -> None:
        # Lines are drawn only when we are having two circles being detected

        if self.circle_count == 2:
            center_1x, center_1y = self.circle_params[0][0:2]
            center_2x, center_2y = self.circle_params[1][0:2]

            # A connecting line is drawn between the centers of the two main circles
            # Just for aesthetics the length of the vertical line is equal to the length of the connecting line

            line_len = np.sqrt(pow((center_1x - center_2x), 2) + pow((center_1y - center_2y), 2))
            half_len = round(0.5 * line_len)

            GREEN, RED = (0, 255, 0), (0, 0, 255)

            cv.line(self.drone_frame, (center_1x, center_1y), (center_2x, center_2y), GREEN, 2)

            cv.line(self.drone_frame,
                    (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y)) - half_len),
                    (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y)) + half_len), GREEN, 2)

            cv.circle(self.drone_frame, (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y))), 4,
                      RED, -1)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
