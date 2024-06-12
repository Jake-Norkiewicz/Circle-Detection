from tellodrone import TelloDrone
from ultralytics import YOLO
from threading import Thread
import time

def detect_and_show(drone, model) -> None:
    while True:
        drone.detect_circles(model)
        drone.show_camera(delay = 1)

def main() -> None:
    drone = TelloDrone()
    model = YOLO("circles600last.pt", verbose = False)
    half_width = 480
    tolerance = 20

    print("waiting for takeoff...")
    time.sleep(5)

    drone.Takeoff()

    camera_thread = Thread(target = detect_and_show, args = [drone, model])
    print("Starting threads")
    camera_thread.start()

    time.sleep(5)

    command_sent = False
    #first_rotate = False
    #start_time = time.time()

    drone.fly(0, 0, 0, 0)

    while True:
        time.sleep(0.1)

        print(f"middle_x: {drone.middle_x}")

        if drone.circle_count >= 2:
            print("2 circles found")

            # go left
            if drone.middle_x > half_width + tolerance and not command_sent:
                drone.fly(0, -5, 0, 0)
                print("rotating left")
                command_sent = True

            # go right
            elif drone.middle_x < half_width - tolerance and not command_sent:
                drone.fly(0, 5, 0, 0)
                print("rotating right")
                command_sent = True

            # in tolerance field
            else:
                drone.fly(0, 0, 0, 0)
                drone.fly(0, 0, 0, 0)
                drone.fly(0, 0, 0, 0)
                print("stopping")
                break
        
        else:
            drone.fly(0, 0, 0, 0)


    print("out of loop")
    time.sleep(2)

    command_sent = False
    # drone.fly(25, 0, 0, 0)
    # time.sleep(8)

    while True:
        if drone.circle_count >= 2 and not command_sent:
            drone.fly(25, 0, 0, 0)
            command_sent = True

        elif not drone.circle_count >= 2 and not command_sent:
            drone.fly(25, 0 ,0, 0)
            command_sent = True
            time.sleep(3)
            break

    drone.fly(0, 0, 0, 0)
    time.sleep(1)
    drone.Land()

    camera_thread.join()

    del drone


if __name__ == "__main__":
    main()