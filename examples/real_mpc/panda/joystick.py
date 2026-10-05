from inputs import get_gamepad
import threading


class Controller:
    def __init__(self):
        self.x = 128
        self.y = 128
        self.up = 0
        self.down = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=()
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(
        self,
    ):
        return [self.x, self.y, self.up, self.down]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "BTN_SOUTH":
                    self.up = event.state
                elif event.code == "BTN_C":
                    self.down = event.state
                elif event.code == "ABS_Y":
                    self.y = event.state
                elif event.code == "ABS_X":
                    self.x = event.state
