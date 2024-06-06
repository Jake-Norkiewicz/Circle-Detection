import cv2 as cv
import numpy as np
from djitellopy.tello import Tello
from ultralytics import YOLO
from enum import Enum

class TelloState(Enum):
    READY = 0
    FLYING = 1
    LANDING = 2

class TelloMission(Enum):
    NONE = 0
    DETECT = 1
    FLY = 2

class TelloDrone():
    def __init__(self) -> None:
        self.drone = Tello()
        self.drone.connect()

        self.state = TelloState.READY
        self.mission = TelloMission.NONE
        self.circle_count = 0

        self.circle_params = []
        self.drone_frame = None
        self.drone.streamon()


    def check_battery(self) -> int:
        return self.drone.get_battery()


    def check_temperature(self) -> float:
        return self.drone.get_temperature()


    def fly(self, x: int = 0, y: int = 0, z: int = 0, yaw: int = 0) -> None:
        self.mission = TelloMission.FLY
        self.drone.send_rc_control(y, x, z, yaw)


    def Takeoff(self) -> None:
        self.state = TelloState.FLYING
        self.drone.takeoff()


    def Land(self) -> None:
        self.mission = TelloMission.NONE
        self.state = TelloState.LANDING
        self.drone.land()
        self.drone.streamoff()


    def show_camera(self, delay: int = 0) -> None:
        temperature = self.check_temperature()
        battery = self.check_battery()

        HEIGHT, WIDTH = self.drone_frame.shape[0:2]
        CYAN, RED, GREEN, SCALE, FONT, THICKNESS = (150, 150, 10), (0, 0, 255), (0, 255, 0), 2.5, cv.FONT_HERSHEY_PLAIN, 4

        state_pos = (int(0.01 * WIDTH), int(0.05 * HEIGHT))
        mission_pos = (int(0.01 * WIDTH), int(0.12 * HEIGHT))
        circle_count_pos = (int(0.01 * WIDTH), int(0.19 * HEIGHT))
        temp_pos = (int(0.75 * WIDTH), int(0.05 * HEIGHT))
        bat_pos = (int(0.83 * WIDTH), int(0.12 * HEIGHT))
        land_pos = (int(0.36 * WIDTH), int(0.9 * HEIGHT))

        state_text = f"STATE: {self.state.name}"
        mission_text = f"MISSION: {self.mission.name}"
        circle_count_text = f"COUNT: {self.circle_count}"
        temp_text = f"TEMP: {temperature}"
        bat_text = f"BAT: {battery}"
        land_text = "LAND ASAP"

        cv.putText(self.drone_frame, state_text, state_pos, FONT, SCALE, CYAN, THICKNESS)
        cv.putText(self.drone_frame, mission_text, mission_pos, FONT, SCALE, CYAN, THICKNESS)
        cv.putText(self.drone_frame, circle_count_text, circle_count_pos, FONT, SCALE, CYAN, THICKNESS)

        # If critical value of temperature or battery is exceeded, show it in red

        if temperature < 90:
            cv.putText(self.drone_frame, temp_text, temp_pos, FONT, SCALE, CYAN, THICKNESS)

        else:
            cv.putText(self.drone_frame, temp_text, temp_pos, FONT, SCALE, RED, THICKNESS)
            cv.putText(self.drone_frame, land_text, land_pos, FONT, 3.0, RED, THICKNESS)

        if battery > 20:
            cv.putText(self.drone_frame, bat_text, bat_pos, FONT, SCALE, CYAN, THICKNESS)

        else:
            cv.putText(self.drone_frame, bat_text, bat_pos, FONT, SCALE, RED, THICKNESS)
            cv.putText(self.drone_frame, land_text, land_pos, FONT, 3.0, RED, THICKNESS)

        # Draw circles on the current frame

        for circle in self.circle_params:
            center_x, center_y, radius, conf=circle[:]
            cv.circle(self.drone_frame, (center_x, center_y), radius, GREEN, 2)

        # Draw a horizontal and vertical line between the only two circles

        if self.circle_count == 2:
            center_1x, center_1y = self.circle_params[0][0:2]
            center_2x, center_2y = self.circle_params[1][0:2]

            line_len = round(np.sqrt(pow((center_1x - center_2x), 2) + pow((center_1y - center_2y), 2)))
            half_len = round(0.5 * line_len)

            cv.line(self.drone_frame, (center_1x, center_1y), (center_2x, center_2y), GREEN, 2)
            cv.line(self.drone_frame, (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y)) - half_len),
                    (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y)) + half_len), GREEN, 2)
            cv.circle(self.drone_frame, (int(0.5 * (center_1x + center_2x)), int(0.5 * (center_1y + center_2y))), 4, RED, -1)

        # Show current frame with all annotations

        cv.imshow("Drone Camera", self.drone_frame)
        cv.waitKey(delay = delay)


    def detect_circles(self, model: YOLO) -> None:
        frame_handle = self.drone.get_frame_read()
        self.drone_frame = frame_handle.frame
        self.state = TelloMission.DETECT

        self.circle_count = 0
        self.circle_params = []

        # Canny by default returns a single-channel image

        canny = cv.Canny(self.drone_frame, threshold1 = 250, threshold2 = 250)
        canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        results = model(canny, conf = 0.7)

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
                self.circle_count = len(boxes_xyxy)


def main() -> None:
    pass

if __name__ == "__main__":
    main()