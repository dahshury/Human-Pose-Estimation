<h1>Human Pose Estimation</h1>

<img src="./media/screenshot1.png" alt="Alt text" height="400"/>

Human pose estimation expirement using several models:

+ [Hourglass network (from-scratch)](https://github.com/princeton-vl/pytorch_stacked_hourglass)
+ [YOLOv8 Pose-nano (fine-tuned)](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo8n-pose.pt)
+ [keypointrcnn_resnet50_fpn w18 (fine-tuned)](https://pytorch.org/vision/0.18/models/keypoint_rcnn.html)

<h3>Built With</h3>

+ [Python](https://www.python.org/downloads/)
+ [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
+ [Albumentations](https://albumentations.ai/)
  
 <h3>About the Dataset</h3>

The [dataset](https://universe.roboflow.com/poseestimation-wzidb/dataset-ridimensionato/dataset/41) consists of 25 keypoints of human body pose. 

<details>
  <summary>Keypoints List</summary>

  1. Right eye  
  2. Nose  
  3. Left eye  
  4. Neck  
  5. Right shoulder  
  6. Right elbow  
  7. Right wrist  
  8. Right hand thumb  
  9. Right hand pinky  
  10. Left shoulder  
  11. Left elbow  
  12. Left wrist  
  13. Left hand pinky  
  14. Left hand thumb  
  15. Torso  
  16. Right hip  
  17. Left hip  
  18. Right knee  
  19. Right foot  
  20. Right foot fingertips  
  21. Right ankle  
  22. Left knee  
  23. Left ankle  
  24. Left foot  
  25. Left foot fingertips

</details>

| Split | Image Count | Percentage |
|-------|-------------|------------|
| train | 1578        | 88.01%     |
| valid | 143         | 7.98%      |
| test  | 72          | 4.02%      |

## How to run

+ STEP 01- Clone the repository

```bash
git clone https://github.com/dahshury/Human-Pose-Estimation
```

+ STEP 02- Create a conda environment after opening the repository (optional)

```bash
conda create -n pose python=3.9 -y
```

```bash
conda activate pose
```

+ STEP 03- install the requirements

```bash
pip install -r requirements.txt
```

+ STEP 04- install training requirements using GPU (optional)

```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

You can now experiment with the notebook, and train any YOLO dataset on several models and compare the results with evaluation metrics and visualizations.

### Results

| Model                                                                                 | mAP (P)<sup>val<br>50 | mAP (P)<sup>val<br>50-95 |
| ------------------------------------------------------------------------------------- | ----------------- | -------------------- |
| [Hourglass network (from-scratch)](https://github.com/princeton-vl/pytorch_stacked_hourglass) | 0.6848           | 0.3971             |
| [YOLOv8-nano (from-scratch)](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 0.664           | 0.2106              |
| [keypointrcnn_resnet50_fpn w18](https://pytorch.org/vision/0.18/models/keypoint_rcnn.html) | 0.783             | 0.3637                |

### Contact Me

[Linkedin](https://www.linkedin.com/in/dahshory/)

### Acknowledgements

Thanks to [Samir Gouda](github.com/SamirGouda) & [Omar Eldahshoury](github.com/omareldahshoury) for thier support.

### License

This project uses code from the YOLOv8 library by Ultralytics, which is licensed under the AGPL-3.0 License.

+ **Repository:** [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
+ **License:** AGPL-3.0 License
+ **License Text:** See [LICENSE](LICENSE) for the full text.

### Citation

```
@misc{dataset-ridimensionato_dataset,
    title = {dataset ridimensionato Dataset},
    type = {Open Source Dataset},
    author = {poseestimation},
    howpublished = {\url{https://universe.roboflow.com/poseestimation-wzidb/dataset-ridimensionato}},
    url = {https://universe.roboflow.com/poseestimation-wzidb/dataset-ridimensionato},
    journal = {Roboflow Universe},
    publisher = {Roboflow},
    year = {2024},
    month = {sep},
    note = {visited on 2024-10-03},
}

@software{Jocher_Ultralytics_YOLO_2023,
    author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
    license = {AGPL-3.0},
    month = {jan},
    title = {{Ultralytics YOLO}},
    url = {https://github.com/ultralytics/ultralytics},
    version = {8.0.0},
    year = {2023}
}
```
