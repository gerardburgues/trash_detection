import os
import cv2
import random
import argparse
import requests
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--realtime", "-rt", action="store_true")
    args = parser.parse_args()

    setup_logger()
    cfg = configure()
    
    register_coco_instances("taco_dataset", {}, "../TACO/data/annotations.json", "../TACO/data")
    taco_dataset_metadata = MetadataCatalog.get("taco_dataset")
    dataset_dicts = DatasetCatalog.get("taco_dataset")

    # show_ground_truth_sample(taco_dataset_metadata, dataset_dicts)

    if args.train:
        train(cfg)
    else:
        predictor = DefaultPredictor(cfg)
        if args.realtime:
            predict_real_time(predictor, taco_dataset_metadata)
        else:
            predict_random_samples(predictor, taco_dataset_metadata, dataset_dicts)

def configure():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    cfg = get_cfg()
    cfg.merge_from_file(
        "../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("taco_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ("taco_dataset", )
    return cfg

def train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def predict_random_samples(predictor, dataset_metadata, dataset_dicts):
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
                im[:, :, ::-1],
                metadata=dataset_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image = v.get_image()[:, :, ::-1]

        while True:
            cv2.imshow("Prediction", image)
            if cv2.waitKey(0) == 27: # Press Esc key to exit
                break

def predict_real_time(predictor, dataset_metadata):
    url = "http://192.168.1.38:8080/shot.jpg"

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        im = cv2.imdecode(img_arr, -1)
        outputs = predictor(im)
        v = Visualizer(
                im[:, :, ::-1],
                metadata=dataset_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        im = v.get_image()[:, :, ::-1]
        cv2.imshow("Android_cam", im)
      
        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            break
      
    cv2.destroyAllWindows()

def show_ground_truth_sample(dataset_metadata, dataset_dicts):
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        
        image = vis.get_image()[:, :, ::-1]
        while True:
            cv2.imshow("Prediction", image)
            if cv2.waitKey(0) == 27: # Press Esc key to exit
                break

if __name__ == "__main__":
    main()
