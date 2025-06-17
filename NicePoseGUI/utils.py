from ultralytics import YOLO
import pandas as pd
import numpy as np
import os.path as OSpath
import cv2
import platform
from pathlib import Path
from typing import Literal, Optional
from nicegui import ui, events, run
from PIL import Image
from multiprocessing import Queue
import json


yolo_keypoint_mapping = {
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Wrist": 9,
    "R_Wrist": 10,
    "L_Hip": 11,
    "R_Hip": 12,
    "L_Knee": 13,
    "R_Knee": 14,
    "L_Ankle": 15,
    "R_Ankle": 16,
    "Nose": 0
}
val_names = ["x", "y", "c"]

class PersonSelectionDialog(ui.dialog):
    def __init__(self, image: np.ndarray, boxes: np.ndarray, ids: np.ndarray,
                 name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props('persistent')

        self.ids = ids
        self.boxes = boxes
        self.name = name
        self.selected_ids = []
        self.selected_names = []
        self.name_inputs = []

        with self, ui.card():
            ui.label(f"Select Individuals to Track from video {name}:")
            self.ii = ui.interactive_image(
                image,
                on_mouse=self._click_select
            )
            with ui.row():
                ui.button("OK", icon="arrow_forward_ios", on_click=self._handle_ok)
                ui.button("Reset", icon="refresh", color="secondary",
                          on_click=self._reset_selection)
                ui.button("Cancel", icon="close", color="negative", on_click=self._handle_cancel)

    def _reset_selection(self):
        self.selected_ids.clear()
        self.ii.set_content("")
        self.name_inputs = [name_input.delete() for name_input in self.name_inputs]
        self.name_inputs.clear()

    def _click_select(self, event):
        x, y = event.image_x, event.image_y
        # Add logic to select the person based on the click coordinates
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                person_id = int(self.ids[i])
                if person_id in self.selected_ids:
                    # self.selected_ids.remove(person_id)
                    ui.notify(f"Person ID {person_id} already selected")
                else:
                    self.selected_ids.append(person_id)
                    # Mark Rectangle on image
                    self.ii.content += f"<rect x={x1} y={y1} height={y2-y1} width={x2-x1} fill='none' stroke=lime stroke-width='5'/>"
                    # Remove file extension from video name
                    name = self.name.split(".")[0]

                    self.name_inputs.append(ui.input(
                        label=f"Specifiy Name for id:{person_id}",
                        value=name if len(self.ids) == 1 else f"{name}_{person_id}",
                        autocomplete=[name],
                        validation={
                            "Cannot Be Empty": lambda value: len(value) >= 1,
                            "Cannot contain ':'": lambda value: ":" not in value,
                            "Cannot contain ','": lambda value: "," not in value,
                            "Cannot contain '.'": lambda value: "." not in value,
                        }
                    ).props("clearable")
                    )

                    ui.notify(f"Person ID {person_id} selected")
                break

    async def _handle_ok(self):
        if any([not name_input.validate() for name_input in self.name_inputs]):
            ui.notify("Given names not valid. See error for guidance.", type="warning")
        else:
            names = [name_input.value for name_input in self.name_inputs]
            self.submit([self.selected_ids, names])

    def _handle_cancel(self):
        self.close()
        self.clear()

def yolo_single(frame, model_name : str):
    model = YOLO(model_name)
    result = model.track(frame,
                         show=False,
                        #  device='cuda' if opt_GpuCpu.value else 'cpu',
                        save=False,
                        show_boxes=False)[0]
    return result

def run_yolo(video_path: str, model_name: str,
             ids: list[int], names: list[str],
             queue: Queue,
             save_path: str = None,
             conf: float = 0.5, device: Literal['cpu', 'cuda'] = 'cpu',
             show: bool = False, verbose: bool = False) -> dict:
    """
    Run YOLOv8 inference on a video file and save the results.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLOv8 model file.
        save_path (str): Folder path to save the output video with detections.
        conf (float): Confidence threshold for detections. Default is 0.5.
        device (str): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        show (bool): Whether to display the video with detections. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing the joint locations and confidence scores.
    """
    # Load the YOLOv8 model
    model = YOLO(model_name)

    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ui.notify(f"Error: Unable to open video file {video_path}")
        return

    # Get video framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set Up output Dataframe for joint locations
    multi_i = pd.MultiIndex.from_product([list(yolo_keypoint_mapping.keys()), val_names],
                                         names=["Point", "Coords"])
    # joint_locs_df = pd.DataFrame(columns=multi_i)

    output = {
        "video": video_path.split("\\")[-1],
        "model": model_name,
        "framerate": fps,
        "points": {name: pd.DataFrame(columns=multi_i) for name in names}
    }

    results = model.track(source=video_path, conf=conf,
                          device=device, show=show,
                          save= False if save_path is None else True, 
                          project=None if save_path is None else save_path,
                          verbose=verbose, stream=True)

    # Function for finding an int in a list of ints.
    # Returns None with error - i.e. id is not found it list
    def find_idx(id: int, id_list: list[int]):
        try:
            return id_list.index(id)
        except:
            return None

    for n, result in enumerate(results):
        for name, id in zip(names, ids):
            # Get index of id of detected person
            idx = find_idx(id, ids)

            if idx is not None:
                xy_all = result.keypoints.xy[idx].tolist()
                conf_all = result.keypoints.conf[idx].tolist()

                xys = [xy_all[i] for i in yolo_keypoint_mapping.values()]
                cs = [conf_all[i] for i in yolo_keypoint_mapping.values()]

                output["points"][name].loc[n] = [item for xy,
                                                 c in zip(xys, cs) for item in [*xy, c]]
            else:
                output["points"][name].loc[n] = None

        queue.put_nowait((n+1)/n_frames)

        # Useful when debugging
        # if n >=5:
        #     break
    return output

# Extend JSON encoder to deal with dataframes with MultiIndexed columns
# by creating a nested dict.
# Can then be read by pd.Dataframe.from_dict(dict, orient="index").stack().to_frame()


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            if isinstance(obj.columns, pd.MultiIndex):
                return {level0: {level1: obj[level0, level1].to_list() for level1 in obj.columns.levels[1]} for level0 in obj.columns.levels[0]}
            else:
                return obj.to_dict('records')
        else:
            return super().default(self, obj)

class settingsDialog(ui.dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open('NicePoseGUI_settings.json', 'r') as f:
            self.settings = json.load(f)
        
        with ui.row():
            ui.label("Settings")
            ui.button("Save", icon="save", on_click=self._save_settings)
            ui.button("Close", icon="close", color="negative", on_click=self.close)
            ui.button("Reset", icon="refresh", color="secondary", on_click=self._reset_settings)
        
        ui.input(
            label='Default Video Path',
            value=self.settings['default_paths']['video_path']
        )
        ui.input(
            label='Default Video List File Path',
            value=self.settings['default_paths']['vid_txt_path'],
        )

class local_file_picker(ui.dialog):

    def __init__(self, directory: str, *,
                 upper_limit: Optional[str] = ..., multiple: bool = False, show_hidden_files: bool = False,
                 extension_filter : str | tuple[str] = None) -> None:
        """Local File Picker

        This is a simple file picker that allows you to select a file from the local filesystem where NiceGUI is running.

        :param directory: The directory to start in.
        :param upper_limit: The directory to stop at (None: no limit, default: same as the starting directory).
        :param multiple: Whether to allow multiple files to be selected.
        :param show_hidden_files: Whether to show hidden files.#
        :param extension_filter: Filter so that only files ending with specified extensions are shown
        :return: List of selected file-paths
        :rtype: list[str]
        """
        super().__init__()
        self.props('persistent')
        self.classes('h=1/2')

        self.path = Path(directory).expanduser()
        self.extension_filter = extension_filter
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(directory if upper_limit == ... else upper_limit).expanduser()
        self.show_hidden_files = show_hidden_files

        with self, ui.card():
            self.add_drives_toggle()
            self.grid = ui.aggrid({
                'columnDefs': [{'field': 'name', 'headerName': 'File'}],
                'rowSelection': 'multiple' if multiple else 'single',
            }, html_columns=[0]).classes('w-96').on('cellDoubleClicked', self.handle_double_click)
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=self.close).props('outline')
                ui.button('Ok', on_click=self._handle_ok)
        self.update_grid()

    def add_drives_toggle(self):
        if platform.system() == 'Windows':
            import win32api
            drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
            self.drives_toggle = ui.toggle(drives, value=drives[0], on_change=self.update_drive)

    def update_drive(self):
        self.path = Path(self.drives_toggle.value).expanduser()
        self.update_grid()

    def update_grid(self) -> None:
        paths = list(self.path.glob('*'))
        # if not self.show_hidden_files:
        #     paths = [p for p in paths if not p.name.startswith('.')]
        # if self.extension_filter is not None:
        #     paths = [p for p in paths if OSpath.isdir(p) or p.name.endswith(self.extension_filter)]
        paths.sort(key=lambda p: p.name.lower())
        paths.sort(key=lambda p: not p.is_dir())

        self.grid.options['rowData'] = [
            {
                'name': f'📁 <strong>{p.name}</strong>' if p.is_dir() else p.name,
                'path': str(p),
            }
            for p in paths
        ]
        if (self.upper_limit is None and self.path != self.path.parent) or \
                (self.upper_limit is not None and self.path != self.upper_limit):
            self.grid.options['rowData'].insert(0, {
                'name': '📁 <strong>..</strong>',
                'path': str(self.path.parent),
            })
        self.grid.update()

    def handle_double_click(self, e: events.GenericEventArguments) -> None:
        self.path = Path(e.args['data']['path'])
        if self.path.is_dir():
            self.update_grid()
        else:
            self.submit([str(self.path)])

    async def _handle_ok(self):
        rows = await self.grid.get_selected_rows()
        self.submit([r['path'] for r in rows])