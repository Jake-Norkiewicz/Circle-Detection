from tellodrone_cleared import TelloDrone
from ultralytics import YOLO
import time


def main() -> None:
    drone = TelloDrone()
    model = YOLO("circles600last.pt", verbose = False)
    half_width = 480
    tolerance = 10

    print("waiting for takeoff...")

    drone.Takeoff()
    drone.fly(0, 0, 0, 0)

    while True:
        drone.detect_circles(model)
        drone.show_camera(delay = 1)
        print(f"count: {drone.circle_count}")

        if drone.circle_count == 2:
            # go left
            if drone.middle_x > half_width + tolerance:
                drone.fly(0, 10, 0, 0)
                print("go right")

            # go right
            if drone.middle_x < half_width - tolerance:
                drone.fly(0, -10, 0, 0)
                print("go left")

            print(f"middle_x: {drone.middle_x}")
            print(f"half_width - tolerance: {half_width - tolerance}")
            print(f"half_width + tolerance: {half_width + tolerance}")

            # centered
            if half_width - tolerance <= drone.middle_x <= half_width + tolerance:
                drone.fly(0, 0, 0, 0)
                print("centered")
                break

    print("out of loop")

    drone.Set_speed(15)
    drone.Move_forward(200)

    time.sleep(3)
    drone.Land()

if __name__ == "__main__":
    main()