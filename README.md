# 基于YOLO的多目标检测实现

## Introduction
此项目是本人学习机器视觉的一个项目，对yolo的学习，感兴趣的可以自行修改，本质上没有进行自己的模型训练
## Background
该项目开始探索各种OpenCV跟踪api，如KCF、MOSSE、CSRT和MedianFlow。虽然这些工具在某些条件下提供了鲁棒性，但它们在速度和准确性方面提出了挑战，特别是在从不同角度跟踪物体时。
参考https://github.com/irmuun8881/YoloTracking
原项目是一个比较经典的例子，在此之上做了一些改进添加许多模块
### YOLOv5 Model
- **yolo5**: 实现了基于yolo5的多目标检测，在原有基础上添加了对目标类型的跟踪输出。
- **classlist**:打印yolo基本检测类型的对应编号，
### YOLO8
- **yolo8**:基于yolo8的目标跟踪，
- **yolo8**:添加了对于目标路径的部分，可以给出目标的移动路径
## Code Overview
The code integrates YOLOv5 for object detection and CSRT for object tracking. It initializes a CSRT tracker for each object detected by YOLOv5 and updates these trackers as the video progresses. The detection frequency and tracker update mechanism ensure computational efficiency.

## Conclusion
This project is a testament to the efficacy of combining different technologies to achieve robust solutions in computer vision, particularly in object tracking. It demonstrates the potential of neural networks in enhancing traditional computer vision techniques.

## Installation Instructions
To set up and run this project, follow these steps:

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/irmuun8881/YoloTracking.git
   cd YoloTracking
Create and Activate Virtual Environment (Optional)

Windows:
sh
Copy code
python -m venv venv
venv\Scripts\activate
macOS/Linux:
sh
Copy code
python3 -m venv venv
source venv/bin/activate
Install Required Packages

sh
Copy code
pip install -r requirements.txt
Running the Project

sh
Copy code
python yolo_csrt.py  # Replace with the name of your main script
shell
Copy code

### requirements.txt
opencv-python
torch
torchvision

vbnet
Copy code

These files provide a comprehensive guide for understanding and running your project. The README offers insights into the project's purpose, technologies used, and setup instructions. The `requirements.txt` file lists essential Python packages for the project's environment.