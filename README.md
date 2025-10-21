
# NearLink RSSI Fingerprinting Dataset
This RSSI Dataset is a comprehensive set of Received Signal Strength Indicator (RSSI) readings gathered from three different types of scenarios. Three wireless technologies were used which consisted of:
- Wi-Fi (IEEE 802.11. AX 2.4GHz band),
- Bluetooth Low Energy (BLE 5.2),
- NearLink (SLE 1.0).

In addition, we also combBefore running the script, you need to execute the following instructions:ined datasets collected using three wireless technologies, including:

- Wi-Fi+BLE, 
- Wi-Fi+NearLink, 
- BLE+NearLink,
- Wi-Fi+BLE+NearLink.
# Hardware
In the experiment, the equipment used includes BearPi Pico H3863 for reading RSSI data and an autonomous robot for automated data collection.

![image](https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/autonomous_robot.png)
<h3 align="center">The hardware of BearPi Pico H3863 and autonomous robot</h3>

The autonomous robot integrates multiple sensors and computing modules, including a [BearPi-Pico H3863 development board](https://github.com/Hny0305Lin/Bearpi_Hi3863_Pico?tab=readme-ov-file), a RoboSense RS-Helios-16P LiDAR, an Intel RealSense D435 RGB-D camera, a HiPNUC CH110 6-axis IMU, and wheel odometry. The chassis is built on an AgileX Scout Mini, and the onboard computer is an NVIDIA Jetson AGX Xavier with 32 GB RAM. The BearPi-Pico H3863 development board incorporates the [HiSilicon Hi3863V100 chipset](https://www.hisilicon.com/en/products/connectivity/short-range-iot/wifi-nearlink-ble/hi3863v100), which supports tri-mode communication (NearLink, BLE, and Wi-Fi) and is compatible with SLE 1.0, BLE 5.2, and Wi-Fi 6 protocols.
# Scripts
You need install ros packages on AgileX scout mini, Generally speschematic diagramaking, the package is already installed if you buy the product. If not, the package can be found at: https://github.com/agilexrobotics/scout_ros 
Before running the script, you need to execute the following instructions on the terminal:
```
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000
roscore
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/scout_base/scout_bringup/launch
roslaunch scount_minimal.launch
sudo chmod 666/dev/ttyUSB1
```
# Environment
<!-- <h3 align="center">Classroom scenario</h3>

| Schematic diagram | Real picture |
|:-----------------:|:-------------:|
| <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_classroom.png" height="800"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_classroom.jpg" height="800"> |
| <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_parking_lot.png" height="400"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_parking_lot.jpg" height="400"> |
| <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_helipad.png" height="400"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_helipad.jpg" height="400"> | -->
