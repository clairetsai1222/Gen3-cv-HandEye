#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading
import gen3_utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

class GripperPose:
    def __init__(self):
        self.args = gen3_utilities.parseConnectionArguments()

    def get_current_gripper_position(self, base, base_cyclic): 
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = "" 

        feedback = base_cyclic.RefreshFeedback() # 从base_cyclic客户端获取当前的反馈信息

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x          # (meters)
        cartesian_pose.y = feedback.base.tool_pose_y + 0.1    # (meters)
        cartesian_pose.z = feedback.base.tool_pose_z + 0.2    # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)夹爪角度：+往下；-往上
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)夹爪角度：+逆时针；-顺时针
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)夹爪角度：+左转；-右转
        
        gripper_pose = [cartesian_pose.x, cartesian_pose.y, cartesian_pose.z, cartesian_pose.theta_x, cartesian_pose.theta_y, cartesian_pose.theta_z]
        return gripper_pose


    def return_gripper_pose(self):
        
        # Import the gen3_utilities helper module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        # Parse arguments
        # args = gen3_utilities.parseConnectionArguments()
        # Create connection to the device and get the router
        with gen3_utilities.DeviceConnection.createTcpConnection(self.args) as router:

            # Create required services
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)
            try:
                gripper_pose = self.get_current_gripper_position(base, base_cyclic)
            except Exception as e:  
                print("Error getting current gripper position: {}".format(e))

            return gripper_pose

if __name__ == "__main__":
    gp = GripperPose()
    print(gp.return_gripper_pose())
