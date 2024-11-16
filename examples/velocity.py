import sys
import time
import constants.unitree_legged_const as go2
import utils.client_utils as client_utils

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

def HighStateHandler(msg: SportModeState_):
    print(msg)

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rsc = client_utils.get_robot_state_client()
    client_utils.set_service(rsc, "sport_mode", True)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)

    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    while True:
        time.sleep(0.01)