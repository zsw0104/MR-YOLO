import os
import warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore", category=FutureWarning)

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8n_t_i.yaml")
    model = YOLO("MR-YOLO.yaml")

    # print(model)

    # Train the model
    train_results = model.train(

        # data="MydataI.yaml",  # path to dataset YAML
        data="mydata.yaml",

        epochs=200,  # number of training epochs
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=16,
        # save=True,  # whether to save the trained model
        # save_period=50,  # save the model every 10 epochs
        # csbqa_loss=True,  # use Class- and Scale-Balanced Quality-Aware loss
    )

    # Evaluate model performance on the validation set
    metrics = model.val()
