from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8m.pt")  # build a new model from scratch

    # Use the model
    results = model.train(data="config.yaml", epochs=200, optimizer='Adam', lr0=0.003, lrf=0.00001, device=0)  # train the model

if __name__ == '__main__':
    main()
