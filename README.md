# NicePoseGUI
A [NiceGUI](https://nicegui.io/) wrapper for tracking individuals in videos using pose estimation tools - primarily [ultralytics](https://github.com/ultralytics) YOLO models.


NicePoseGUI provides an interface to select videos, indicate which individuals should be tracked and have their joint positions recorded and then outputs the data for the selected indivudals in a simple json format.

The JSON format looks like this:
```JSON
{
    "video" : "original video name",
    "model" : "model used to generate data",
    "framerate" : "framerate of video",
    "points" : {
        "name1" : {
            "L_Ankle": {
                "c" : [0.1, 0.2, 0.3, "..."],
                "x" : [101, 102, 103, "..."],
                "y" : [104, 105, 106, "..."]
            },
            "R_Ankle" : {
                "c" : [0.1, 0.2, 0.3, "..."],
                "x" : [101, 102, 103, "..."],
                "y" : [104, 105, 106, "..."]
            },
            etc...
        },
        "name2" : {
            "L_Ankle" : {"..."}
        }
    }
}
```
