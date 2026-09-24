from inputs import get_gamepad
import threading


class Controller:
    def __init__(self):
        self.x = 128
        self.y = 128
        self.z = 128
        # -1 while the D-pad is held up, 1 while held down
        self.dpad_y = 0
        # -1 while the D-pad is held left, 1 while held right
        self.dpad_x = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=()
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(
        self,
        max_vel_front=1.0,
        max_vel_back=0.75,
        max_vel_side=0.75,
        max_vel_yaw=0.75,
    ):
        vel_x = -(self.x - 128) / 128 * max_vel_side
        vel_y = -(self.y - 128) / 128
        vel_z = -(self.z - 128) / 128 * max_vel_yaw

        if vel_y > 0.0:
            vel_y *= max_vel_front
        else:
            vel_y *= 2 / 3 * max_vel_back

        return [vel_y, vel_x, vel_z]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_Z":
                    self.z = event.state
                elif event.code == "ABS_Y":
                    self.y = event.state
                elif event.code == "ABS_X":
                    self.x = event.state
                elif event.code == "ABS_HAT0Y":
                    self.dpad_y = event.state
                elif event.code == "ABS_HAT0X":
                    self.dpad_x = event.state


if __name__ == "__main__":
    from time import sleep

    controller = Controller()

    while True:
        print(controller.read())
        sleep(0.1)
