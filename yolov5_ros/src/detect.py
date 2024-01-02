#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
import os
import sys
from rostopic import get_topic_type
import message_filters
from scipy import stats

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from detection_msgs.msg import BoundingBox, BoundingBoxes


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup

        # Initialize CV_Bridge
        self.bridge = CvBridge()
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.image_sub = rospy.Subscriber(
            input_image_topic, Image, self.callback, queue_size=1)


        # 创建彩色和深度图像的订阅者
        #color_sub = message_filters.Subscriber(rospy.get_param("~input_image_topic"), Image)
        #depth_sub = message_filters.Subscriber(rospy.get_param("~input_depth_topic"), Image)

        # 创建一个时间同步器
        #ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=2, slop=300)
        #ts.registerCallback(self.callback)



        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )

        # 在Yolov5Detector类的初始化中添加成员变量来存储内参
        self.depth_camera_intrinsics = None



        # 然后，在初始化中订阅/camera/depth/camera_info话题
        self.depth_info_sub = rospy.Subscriber(rospy.get_param("~depth_camera_info"), CameraInfo, self.depth_info_callback)

    # 添加内参的回调函数
    def depth_info_callback(self, camera_info_msg):
        self.depth_camera_intrinsics = np.array(camera_info_msg.K).reshape(3, 3)

    def get_median_depth_circle(self, depth_frame, center, radius):
        """
        Get the median depth value within the specified radius with the center point as the center of the circle.
        """
        cx, cy = center
        depth_values = []

        for x in range(max(0, cx - radius), min(cx + radius + 1, depth_frame.shape[0])):
            for y in range(max(0, cy - radius), min(cy + radius + 1, depth_frame.shape[1])):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    depth = depth_frame[y,x]
                    if depth > 0:  # # Ignore invalid depth values
                        depth_values.append(depth)

        if depth_values:
            return np.median(depth_values)
        else:
            return 0

    def get_3d_coord(self, x, y, depth, depth_scale):
        """
        Get 3D coordinates from 2D pixel coordinates and depth values
        """
        # 获取内参
        fx = self.depth_camera_intrinsics[0, 0]
        fy = self.depth_camera_intrinsics[1, 1]
        cx = self.depth_camera_intrinsics[0, 2]
        cy = self.depth_camera_intrinsics[1, 2]

        depth = depth * depth_scale
        x3d = (x - cx) / fx * depth
        y3d = (y - cy) / fy * depth
        z3d = depth

        return x3d, y3d, z3d


    def callback(self, image_sub):
        """adapted from yolov5/detect.py"""
        # 确保我们有内参数据
        if self.depth_camera_intrinsics is None:
            rospy.logwarn("No depth camera intrinsics available.")
            return

        # 先获取深度图像的话题
        depth_image_topic = rospy.get_param("~input_depth_topic")
        # 调用一次订阅者来获取最新的深度图像消息
        depth_image_msg = rospy.wait_for_message(depth_image_topic, Image)
        try:
            # 转换深度图像消息到OpenCV图像
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        im = self.bridge.imgmsg_to_cv2(image_sub, desired_encoding="bgr8")

        im, im0 = self.preprocess(im)
        # print(im.shape)
        # print(img0.shape)
        # print(img.shape)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = image_sub.header
        bounding_boxes.image_header = image_sub.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                # 计算边界框中心点的像素坐标
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]

                # 设置圆的半径为检测框的一半宽度或高度中较小的那个
                radius = int(min(width, height) / 2)

                median_depth = self.get_median_depth_circle(depth_image, (int(x_center), int(y_center)), int(radius))

                x3d, y3d, z3d = self.get_3d_coord(x_center, y_center, median_depth, 0.0010000000474974513)

                # 输出三维坐标
                rospy.loginfo(f"{self.names[c]}, depth: Z={z3d:.2f}")

                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))



    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()

'''
import pyrealsense2 as rs
pipeline = rs.pipeline()
pipeline.start()

profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("Depth Scale is: ", depth_scale)
pipeline.stop()
'''