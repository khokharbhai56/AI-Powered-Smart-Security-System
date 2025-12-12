@echo off
echo Creating project directories...

REM Create main directories
if not exist "project\models" mkdir "project\models"
if not exist "project\logs" mkdir "project\logs"
if not exist "project\screenshots" mkdir "project\screenshots"
if not exist "project\data" mkdir "project\data"

REM Create model subdirectories
if not exist "project\models\yolov12_security" mkdir "project\models\yolov12_security"
if not exist "project\models\yolov12_security\weights" mkdir "project\models\yolov12_security\weights"
if not exist "project\models\mask_rcnn_output" mkdir "project\models\mask_rcnn_output"
if not exist "project\models\cnn_models" mkdir "project\models\cnn_models"

REM Create data subdirectories
if not exist "project\data\yolo_dataset" mkdir "project\data\yolo_dataset"
if not exist "project\data\yolo_dataset\images" mkdir "project\data\yolo_dataset\images"
if not exist "project\data\yolo_dataset\images\train" mkdir "project\data\yolo_dataset\images\train"
if not exist "project\data\yolo_dataset\images\val" mkdir "project\data\yolo_dataset\images\val"
if not exist "project\data\yolo_dataset\images\test" mkdir "project\data\yolo_dataset\images\test"
if not exist "project\data\yolo_dataset\labels" mkdir "project\data\yolo_dataset\labels"
if not exist "project\data\yolo_dataset\labels\train" mkdir "project\data\yolo_dataset\labels\train"
if not exist "project\data\yolo_dataset\labels\val" mkdir "project\data\yolo_dataset\labels\val"
if not exist "project\data\yolo_dataset\labels\test" mkdir "project\data\yolo_dataset\labels\test"

if not exist "project\data\coco" mkdir "project\data\coco"
if not exist "project\data\action_dataset" mkdir "project\data\action_dataset"
if not exist "project\data\videos" mkdir "project\data\videos"

REM Create log subdirectories
if not exist "project\logs\training" mkdir "project\logs\training"
if not exist "project\logs\inference" mkdir "project\logs\inference"
if not exist "project\logs\alerts" mkdir "project\logs\alerts"

echo Directory structure created successfully!
echo.
echo Project structure:
echo project/
echo ├── models/
echo │   ├── yolov12_security/
echo │   │   └── weights/
echo │   ├── mask_rcnn_output/
echo │   └── cnn_models/
echo ├── logs/
echo │   ├── training/
echo │   ├── inference/
echo │   └── alerts/
echo ├── screenshots/
echo └── data/
echo     ├── yolo_dataset/
echo     │   ├── images/
echo     │   │   ├── train/
echo     │   │   ├── val/
echo     │   │   └── test/
echo     │   └── labels/
echo     │       ├── train/
echo     │       ├── val/
echo     │       └── test/
echo     ├── coco/
echo     ├── action_dataset/
echo     └── videos/
echo.
pause
