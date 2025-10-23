
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
Three experimental scenarios were selected: two indoor environments (classroom and parking lot) and one outdoor environment (helipad). All experiments were conducted at the National Elite Institute of Engineering, Chongqing University.
<!-- <h3 align="center">Classroom scenario</h3>

| Classroom | Parking lot | Helipad |
|:-----------------:|:-------------:|:-------------:|
| Schematic diagram | Schematic diagram | Schematic diagram |
| <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_classroom.png" width="255"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_parking_lot.png" height="300"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_helipad.png" height="300"> |
| <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_classroom.jpg" width="255"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_parking_lot.png" height="340"> | <img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_helipad.png" height="340"> | -->
- The classroom has an area of approximately 60 m² (8.2 × 7.2 m). 4 transmitters were installed at the corners of the classroom, forming a square with a side length of 5 m. The central area enclosed by the transmitters, measuring 4.5 × 4.5 m, was defined as the target area for fingerprint database construction. Within this region, 81 reference points were arranged at 0.5 m intervals. Each reference point was marked with a 5 cm × 5 cm blue square sticker to indicate the ground-truth coordinates. The ground-truth positions were measured using a laser rangefinder (SNDWAY SW-120GS, ±2 mm accuracy). 16 points were randomly selected as test points (marked with pink stickers). 

- The parking lot covers an area of approximately 1200 m² (30 × 40 m). 8 transmitters were installed around the central area, forming a rectangle measuring 28.5 × 22.5 m. The central target area for fingerprint database construction measured 27 × 21 m. A total of 252 reference points were arranged at 1.5 m intervals, marked with blue stickers, and 50 test points were randomly selected and marked with pink stickers. 

- The helipad covers an area of approximately 1600 m² (40 × 40 m). 8 transmitters were installed around the central area, forming a square with a side length of 16 m. The central target region area 15 × 15 m. Within this region, 225 reference points were arranged at 1 m intervals and marked with blue stickers, while 45 test points were randomly selected and marked with pink stickers. 

In this work, we adopted the open-source [H-IPS](https://github.com/6trem/H-IPS) algorithm for range based Multilateration. For the fingerprinting method, we adopted one machine learning algorithm ([KNN](https://github.com/SensorOrgNet/A_Soft_Range_Limited_K_Nearest_Neighbors_Algorithm_for_Indoor_Localization_Enhancement)) and four deep learning algorithms (MLP, [LSTM-RNN](https://github.com/SensorOrgNet/Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization), [HADNN](https://codeocean.com/capsule/6488957/tree/v1) and [GconvLoc](https://github.com/dongdokee/GConvLoc).

<table>
  <tr>
    <th>Classroom</th>
    <th>Parking lot</th>
    <th>Helipad</th>
  </tr>

  <!-- 示意图图片行 -->
  <tr>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_classroom.png" width="255"></td>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_parking_lot.png" height="300"></td>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/scenario_helipad.png" height="300"></td>
  </tr>
  <!-- 示意图标题行 -->
  <!-- <tr>
    <td colspan="3" align="center"><b>Schematic diagram</b></td>
  </tr> -->


  <!-- 实际场景图片行 -->
  <tr>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_classroom.jpg" width="255"></td>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_parking_lot.png" height="340"></td>
    <td><img src="https://github.com/CQU-UISC/NearLink-RSSI-Fingerprinting-Dataset/blob/main/IMG/real_helipad.png" height="340"></td>
  </tr>
  <!-- 实际场景标题行 -->
  <!-- <tr>
    <td colspan="3" align="center"><b>Real scenario</b></td>
  </tr> -->
</table>
<h3 align="center">Schematic diagram and real scenario in 3 scenarios</h3>

# Related Paper
S. Xie, X. Long, F. Guo, Y. Guo and F. Gu, "Comparison of RSSI-Based Indoor/Outdoor Localization Using NearLink Technology," IEEE Transactions on Mobile Computing. To be published.


# Acknowledgments
This project leverages the following works:
- A. C. Eyng, O. K. Rayel, E. Oroski, and J. L. Rebelatto, “Kalman filtering-aided hybrid indoor positioning system with fingerprinting and multilateration,” in 2020 IEEE 91st vehicular technology conference (VTC2020-Spring). IEEE, 2020, pp. 1–5. Project: [H-IPS](https://github.com/6trem/H-IPS)
- M. T. Hoang, Y. Zhu, B. Yuen, T. Reese, X. Dong, T. Lu, R. Westendorp, and M. Xie, “A soft range limited k-nearest neighbors algorithm for indoor localization enhancement,” IEEE Sensors Journal, vol. 18, no. 24, pp. 10 208–10 216, 2018. Project: [KNN](https://github.com/SensorOrgNet/A_Soft_Range_Limited_K_Nearest_Neighbors_Algorithm_for_Indoor_Localization_Enhancement)
- Y. Etiabi, W. Njima, and E. M. Amhoud, “Federated learning based hierarchical 3d indoor localization,” in 2023 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2023, pp. 1–6.
- M. T. Hoang, B. Yuen, X. Dong, T. Lu, R. Westendorp, and K. Reddy, “Recurrent neural networks for accurate rssi indoor localization,” IEEE Internet of Things Journal, vol. 6, no. 6, pp. 10 639–10 651, 2019. Project: [LSTM-RNN](https://github.com/SensorOrgNet/Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization)
- J. Cha and E. Lim, “A hierarchical auxiliary deep neural network architecture for large-scale indoor localization based on wi-fi fingerprinting,” Applied Soft Computing, vol. 120, p. 108624, 2022. Project: [HADNN](https://codeocean.com/capsule/6488957/tree/v1)
- D. Kim and Y.-J. Suh, “Gconvloc: Wifi fingerprinting-based indoor localization using graph convolutional networks,” IEICE TRANSACTIONS on Information and Systems, vol. 106, no. 4, pp. 570–574, 2023. Project: [GconvLoc](https://github.com/dongdokee/GConvLoc)


