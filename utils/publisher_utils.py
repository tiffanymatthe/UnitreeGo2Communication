from unitree_sdk2py.core.channel import ChannelPublisher
from typing import Any
import time

class ChannelLogPublisher(ChannelPublisher):
    def __init__(self, name: str, type: Any):
        super().__init__(name, type)
        # list of timestamps and published commands
        self.logs = []

    def Write(self, sample: Any, timeout: float = None):
        success = self.__channel.Write(sample, timeout)
        self.logs.append((time.time(), sample))
        return success