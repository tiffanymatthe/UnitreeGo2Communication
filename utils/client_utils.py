import time
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

def get_robot_state_client():
    rsc = RobotStateClient()
    rsc.SetTimeout(3.0)
    rsc.Init()
    return rsc

def print_service_list(rsc: RobotStateClient):
    code, lst = rsc.ServiceList()
    
    if code != 0:
        print("list service error. code:", code)
    else:
        print("list service success. len:", len(lst))
        for s in lst:
            print("name:", s.name, ", protect:", s.protect, ", status:", s.status)

    time.sleep(3)

def set_service(rsc: RobotStateClient, service_name: str, mode: bool):
    code = rsc.ServiceSwitch(service_name, mode)
    if code != 0:
        print(f"service stop {service_name} error. code: {code}")
    else:
        print(f"service stop {service_name} success. code: {code}")

    time.sleep(3)
