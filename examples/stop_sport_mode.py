from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import client_utils as client_utils

import sys

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rsc = client_utils.get_robot_state_client()
    client_utils.print_service_list(rsc)
    client_utils.set_service(rsc, "sport_mode", False)
