from tellodrone import TelloDrone
import cv2 as cv
from ultralytics import YOLO
import time
from threading import Thread

def detect_and_show(drone, model) -> None:
    while True:
        drone.detect_circles(model)
        drone.show_camera(delay = 1)

def main():
    drone = TelloDrone()

    model = YOLO("circles600last.pt")
    w_half = 480
    tolerance = 20

    cam_thread = Thread(target = detect_and_show, args = [drone, model])
    cam_thread.start()

    time.sleep(5)

    drone.Takeoff()

    while True:
        if drone.two_circles() and drone.middle_x < w_half - tolerance:
            drone.fly(yaw = -15)

        elif drone.two_circles() and drone.middle_x > w_half + tolerance:
            drone.fly(yaw = 15)

        elif drone.two_circles():
            drone.fly(0, 0, 0, 0)
            time.sleep(2)
            drone.fly(x = 20)
            time.sleep(3)
            drone.fly(0, 0, 0, 0)

            break

        else:
            drone.fly(0, 0, 0, 0)

    drone.Land()


if __name__ == "__main__":
    main()