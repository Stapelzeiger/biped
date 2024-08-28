#!/usr/bin/env python
# -*- coding: utf-8 -*-

#*******************************************************************************
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************


#*******************************************************************************
#***********************     Read and Write Example      ***********************
#  Required Environment to run this example :
#    - Protocol 2.0 supported DYNAMIXEL(X, P, PRO/PRO(A), MX 2.0 series)
#    - DYNAMIXEL Starter Set (U2D2, U2D2 PHB, 12V SMPS)
#  How to use the example :
#    - Select the DYNAMIXEL in use at the MY_DXL in the example code. 
#    - Build and Run from proper architecture subdirectory.
#    - For ARM based SBCs such as Raspberry Pi, use linux_sbc subdirectory to build and run.
#    - https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/
#  Author: Ryu Woon Jung (Leon)
#  Maintainer : Zerom, Will Son
# *******************************************************************************

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import * # Uses Dynamixel SDK library

MY_DXL = 'X_SERIES' # X330 (5.0 V recommended), X430, X540, 2X430


# Control table address
if MY_DXL == 'X_SERIES' or MY_DXL == 'MX_SERIES':
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    DXL_MINIMUM_POSITION_VALUE = 0 # Refer to the Minimum Position Limit of product eManual
    DXL_MAXIMUM_POSITION_VALUE = 4095      # Refer to the Maximum Position Limit of product eManual
    BAUDRATE = 57600

PROTOCOL_VERSION = 2.0

# Factory default ID of all DYNAMIXEL is 1
DXL_ID = 1
DEVICENAME = sys.argv[1]  # Check which port is being used on your controller

TORQUE_ENABLE = 1 # Value for enabling the torque
TORQUE_DISABLE = 0 # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20 # Dynamixel moving status threshold

OP_CURRENT_BASED_POSITION = 6
OP_POSITION = 1
ADDR_OPERATING_MODE = 11

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

def communication_valid(comm_result, err):
    if comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(comm_result))
    elif err != 0:
        print("%s" % packetHandler.getRxPacketError(err))
    else:
        return True
    return False

# Disable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if communication_valid(dxl_comm_result, dxl_error) == True:
    print("Torque has been successfully disabled")

# Set operating mode to current-based position control
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, OP_POSITION)
if communication_valid(dxl_comm_result, dxl_error) == True:
    print("Operating mode has been successfully changed to current-based position control")

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if communication_valid(dxl_comm_result, dxl_error) == True:
    print("Torque has been successfully enabled")

while 1:
    # Read present position
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    communication_valid(dxl_comm_result, dxl_error)

    # Hold position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, dxl_present_position)
    communication_valid(dxl_comm_result, dxl_error)

# Disable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

# Close port
portHandler.closePort()
