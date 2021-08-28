from tensorflow.keras import callbacks

from yolov4.tf import YOLOv4, YOLODataset, SaveWeightsCallback

cfg = "yolov4"
weight = "yolov4.weights"
if __name__ == "__main__":
    yolo = YOLOv4()

    yolo.config.parse_names("coco.names")
    yolo.config.parse_cfg(f"config/{cfg}.cfg")

    yolo.make_model()
    yolo.load_weights(
        f"weights/{weight}",
        weights_type="yolo",
    )
    yolo.summary(summary_type="yolo")

    for i in range(29):
        yolo.model.get_layer(index=i).trainable = False

    yolo.summary()

    train_dataset = YOLODataset(
        config=yolo.config,
        dataset_list="data2/_annotations_train.txt",
        image_path_prefix="data2/train/images",
        training=True,
    )

    val_dataset = YOLODataset(
        config=yolo.config,
        dataset_list="data2/_annotations_valid.txt",
        image_path_prefix="data2/valid/images",
        training=False,
    )

    yolo.compile()

    _callbacks = [
        callbacks.TerminateOnNaN(),
        callbacks.TensorBoard(
            log_dir=f"logs/{cfg}",
            update_freq=200,
            histogram_freq=1,
        ),
        SaveWeightsCallback(
            yolo=yolo,
            dir_path=f"trained/{cfg}",
            weights_type="yolo",
            step_per_save=2000,
        ),
    ]

    yolo.fit(
        train_dataset,
        callbacks=_callbacks,
        validation_data=val_dataset,
        verbose=3,  # 3: print step info
    )