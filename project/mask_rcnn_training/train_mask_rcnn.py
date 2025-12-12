"""
Mask R-CNN Training Script for Person Segmentation
"""

import os
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import mlflow
import wandb
from pathlib import Path
from utils.config import Config
from utils.logger import setup_logger

def setup_detectron_config(config):
    """Setup Detectron2 configuration"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coco_person_train",)
    cfg.DATASETS.TEST = ("coco_person_val",)
    cfg.DATALOADER.NUM_WORKERS = config.mask_rcnn_config['num_workers']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = config.mask_rcnn_config['ims_per_batch']
    cfg.SOLVER.BASE_LR = config.mask_rcnn_config['base_lr']
    cfg.SOLVER.MAX_ITER = config.mask_rcnn_config['max_iter']
    cfg.SOLVER.STEPS = config.mask_rcnn_config['steps']
    cfg.SOLVER.GAMMA = config.mask_rcnn_config['gamma']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only person class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = "./models/mask_rcnn_output"

    return cfg

def register_datasets(config):
    """Register COCO datasets for person segmentation"""
    # Register COCO person dataset
    register_coco_instances(
        "coco_person_train",
        {},
        str(Path(config.dataset_paths['coco']) / "annotations" / "person_keypoints_train2017.json"),
        str(Path(config.dataset_paths['coco']) / "train2017")
    )

    register_coco_instances(
        "coco_person_val",
        {},
        str(Path(config.dataset_paths['coco']) / "annotations" / "person_keypoints_val2017.json"),
        str(Path(config.dataset_paths['coco']) / "val2017")
    )

class CustomTrainer(DefaultTrainer):
    """Custom trainer with MLflow logging"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = setup_logger('mask_rcnn_training')

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def run_step(self):
        super().run_step()

        # Log metrics every 100 iterations
        if self.iter % 100 == 0:
            metrics = self.storage.latest()
            mlflow.log_metrics({
                'iteration': self.iter,
                'total_loss': metrics.get('total_loss', 0),
                'loss_cls': metrics.get('loss_cls', 0),
                'loss_box_reg': metrics.get('loss_box_reg', 0),
                'loss_mask': metrics.get('loss_mask', 0),
                'loss_rpn_cls': metrics.get('loss_rpn_cls', 0),
                'loss_rpn_loc': metrics.get('loss_rpn_loc', 0)
            }, step=self.iter)

def train_mask_rcnn(config_path='config.yaml'):
    """Train Mask R-CNN for person segmentation"""
    config = Config(config_path)
    logger = setup_logger('mask_rcnn_training')

    # Setup MLflow
    if config.mlops_config['mlflow_tracking_uri']:
        mlflow.set_tracking_uri(config.mlops_config['mlflow_tracking_uri'])
        mlflow.set_experiment(config.mlops_config['experiment_name'] + '_mask_rcnn')

    # Setup WandB
    if config.mlops_config['wandb_project']:
        wandb.init(project=config.mlops_config['wandb_project'],
                  name='mask_rcnn_training')

    with mlflow.start_run():
        # Register datasets
        register_datasets(config)

        # Setup config
        cfg = setup_detectron_config(config)

        # Create output directory
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Log parameters
        mlflow.log_params(config.mask_rcnn_config)

        # Setup trainer
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)

        # Training
        logger.info("Starting Mask R-CNN training...")
        trainer.train()

        # Evaluation
        logger.info("Evaluating model...")
        evaluator = COCOEvaluator("coco_person_val", output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "coco_person_val")
        results = inference_on_dataset(trainer.model, val_loader, evaluator)

        # Log final metrics
        mlflow.log_metrics({
            'bbox_AP': results['bbox']['AP'],
            'bbox_AP50': results['bbox']['AP50'],
            'bbox_AP75': results['bbox']['AP75'],
            'segm_AP': results['segm']['AP'],
            'segm_AP50': results['segm']['AP50'],
            'segm_AP75': results['segm']['AP75']
        })

        # Save model
        final_model_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
        if final_model_path.exists():
            mlflow.log_artifact(str(final_model_path), 'models')

        logger.info("Mask R-CNN training completed")
        return trainer.model

if __name__ == '__main__':
    train_mask_rcnn()
