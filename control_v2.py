from tellodrone import TelloDrone
from ultralytics import YOLO
#from threading import Thread
#from enum import Enum
import time
#from multiprocessing import Process


def main() -> None:
    drone = TelloDrone()
    model = YOLO("circles600last.pt", verbose=False)
    half_width = 480
    tolerance = 20

    print("waiting for takeoff...")

    #drone.Takeoff()

    time.sleep(5)

    command_sent = False

    while True:
        drone.detect_circles(model)
        drone.show_camera(delay = 1)

        if drone.circle_count == 2:
            print("2 circles found")

            # go left
            if drone.middle_x > half_width + tolerance and not command_sent:
                drone.fly(0, 0, 0, -10)
                print("rotating left")
                command_sent = True

            # go right
            elif drone.middle_x < half_width - tolerance and not command_sent:
                drone.fly(0, 0, 0, 10)
                print("rotating right")
                command_sent = True

            # in tolerance field
            else:
                drone.fly(0, 0, 0, 0)
                drone.fly(0, 0, 0, 0)
                drone.fly(0, 0, 0, 0)
                print("centered")
                break

    print("out of loop")

    drone.fly(25, 0, 0, 0)
    time.sleep(5)

    #camera_thread.join()

    del drone


if __name__ == "__main__":
    main()