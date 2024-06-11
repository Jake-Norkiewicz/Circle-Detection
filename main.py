from tellodrone import TelloDrone
from ultralytics import YOLO
from threading import Thread
from enum import Enum
import time

class Side(Enum):
    LEFT = 0
    RIGHT = 1

def which_side(drone: TelloDrone, half_width: int) -> Side:
    if drone.middle_x >= half_width:
        return Side.RIGHT

    return Side.LEFT


def detect_and_show(drone: TelloDrone, model: YOLO) -> None:
    while True:
        try:
            drone.detect_circles(model)
            drone.show_camera(delay = 1)

        except:
            return

def main() -> None:
    drone = TelloDrone()
    model = YOLO("circles600last.pt")
    half_width = drone.camera_dimensions()[0] // 2

    time.sleep(5)

    camera_thread = Thread(target = detect_and_show, args = [drone, model])
    camera_thread.start()

    time.sleep(2)

    # TAKING OFF!
    drone.Takeoff()

    time.sleep(1)

    positioning = True
    rotating = False
    finding_circles = False
    initial_side = None
    middle_yaw = None

    while positioning:
        if drone.circle_count == 2:

            if finding_circles:
                drone.fly(0, 0, 0, 0)
                drone.set_console_msg("STOP")

                finding_circles = False
                time.sleep(1)

            if initial_side is None:
                initial_side = which_side(drone, half_width)

            if rotating:
                current_side = which_side(drone, half_width)

                if not current_side == initial_side:
                    middle_yaw = drone.check_yaw()
                    drone.fly(0, 0, 0, 0)
                    drone.set_console_msg("STOP")

                    positioning = False
                    time.sleep(1)


            if not rotating and drone.middle_x < half_width:
                drone.fly(0, 0, 0, -15)
                drone.set_console_msg("TURN RIGHT")

                rotating = True
                time.sleep(1)

            if not rotating and drone.middle_x > half_width:
                drone.fly(0, 0, 0, 15)
                drone.set_console_msg("TURN LEFT")

                rotating = True
                time.sleep(1)

        else:
            drone.fly(0, 0, 0, 25)
            drone.set_console_msg("FINDING CIRCLES")

            finding_circles = True
            time.sleep(1)

    current_yaw = drone.check_yaw()
    current_side = which_side(drone, half_width)
    rot_angle = abs(current_yaw - middle_yaw)

    if current_side == Side.LEFT:
        drone.rotate_CW(rot_angle)

    else:
        drone.rotate_CCW(rot_angle)

    time.sleep(2)

    drone.fly(20, 0, 0, 0)
    drone.set_console_msg("STRAIGHT AHEAD")

    time.sleep(5)

    drone.fly(0, 0, 0, 0)
    drone.set_console_msg("LANDING")

    drone.Land()
    del drone


if __name__ == "__main__":
    main()