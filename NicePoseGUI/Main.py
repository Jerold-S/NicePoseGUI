from nicegui import ui, run
from ultralytics import YOLO
import json
import cv2
from PIL import Image
from multiprocessing import Manager
import utils

# os.chdir(os.path.dirname(__file__))

settings_path = "NicePoseGUI_Settings.json"
with open(settings_path, "r") as settings_file:
    settings_json = json.load(settings_file)

yolo_models = ["yolo11x-pose", "yolov8x-pose"]


async def delete_selected_rows(table: ui.table):
    table.rows = [row for row in table.rows if row not in table.selected]
    table.update()


@ui.page('/')
def RowMech_PoseEstimation():
    queue = Manager().Queue()

    async def add_video_to_table():
        paths = await utils.local_file_picker('../../Pose_Estimation/Data_Aquisition', multiple=True,
                                              extension_filter=('.mp4', '.avi', '.mov', '.MOV', '.AVI', 'MP4'))
        if paths is None:
            return
        else:
            for path in paths:
                video_table.add_row({
                    "video": path.split('\\')[-1],
                    "path": '\\'.join(path.split('\\')[:-1])
                })

            video_table.run_method('scrollTo', len(video_table.rows)-1)
            video_table.update()

    async def add_video_to_table_fromfile():
        path = await utils.local_file_picker('', multiple=False)
        if path is None:
            return
        else:
            with open(*path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    video_table.add_row({
                        "video": line.strip().split('\\')[-1],
                        "path": '\\'.join(line.strip().split('\\')[:-1])
                    })

                video_table.run_method('scrollTo', len(video_table.rows)-1)
                video_table.update()

    async def select_people(table: ui.table, selected: bool = False):
        """Select people in the selected videos using YOLOv8 tracking.

        :param table: The table containing video information.
        :param selected: If True, only process selected rows; otherwise, process all rows.
        :return: None
        """
        # Check some videos are in table or selected
        if selected and len(table.selected) == 0:
            ui.notify("No videos selected", color="negative")
            return
        elif not selected and len(table.rows) == 0:
            ui.notify("No videos in table", color="negative")
            return
        elif selected and len(table.selected):
            rows = table.selected
        else:
            rows = table.rows
        # Cycle through video rows and run Person Selection
        for row in rows:
            row['progress'] = 'Selecting...'
            table.update()

            video_name = row['video']
            video_folder_path = row['path']
            video_path = '\\'.join([video_folder_path, video_name])

            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            if not success:
                ui.notify(f'Error reading video Path: {video_path}', type='warning')
                return

            model = YOLO(opt_modelSelect.value + '.pt')

            result = model.track(
                source=frame,
                show=False,
                device='cuda' if opt_GpuCpu.value else 'cpu',
                save=False,
                show_boxes=False)[0]

            selected_ids, names = await utils.PersonSelectionDialog(
                image=Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)),
                boxes=result.boxes.xyxy.numpy(),
                ids=result.boxes.id.numpy(),
                name=video_name.split('.')[0]
            )

            row["ids"] = ",".join([":".join([id, name])
                                    for id, name in zip(map(str, selected_ids), names)])

            row['progress'] = ''
            table.update()

    async def run_pose_detection(vid_table: ui.table):
        # Check some videos are in table
        if len(vid_table.rows) == 0:
            ui.notify("No videos selected", color="negative")
            return

        # Cycle through video rows and run Pose Estimation
        for row in vid_table.rows:
            folder_path = row['path']
            video_name = row['video']
            video_path = "\\".join([folder_path, video_name])

            idx, names = map(list, zip(*[comb.split(":") for comb in row["ids"].split(",")]))
            idx = list(map(int, idx))  # Convert IDs from string to int

            row['progress'] = "Processing..."
            vid_table.update()
            prog_bar.set_value(0)
            prog_bar.update()

            # Run YOLOv8 inference on the video
            print(opt_modelSelect.value)
            output = await run.cpu_bound(
                utils.run_yolo,
                video_path=video_path, model_name=opt_modelSelect.value + '.pt',
                ids=idx, names=names,
                queue=queue,
                save_path=video_path.replace(
                    video_name.split('.')[-1], f'_pose_{opt_modelSelect.value}.mp4') if opt_WriteVideo.value else None,
                conf=0.5,
                device='cuda' if opt_GpuCpu.value else 'cpu',
                show=opt_display.value, verbose=opt_Verbose.value
            )

            with open('\\'.join([
                folder_path,
                video_name.replace('.mp4', f'_pose_{opt_modelSelect.value}.json')
            ]), "w") as file:
                json.dump(output, file, cls=utils.JSONEncoder, indent=4)

            row['progress'] = "Done"
            video_table.update()

    ui.separator()

    with ui.row(align_items='center'):
        ui.button("Select Video Files", icon='folder', on_click=add_video_to_table)
        ui.button("Add from List File", icon='folder', on_click=add_video_to_table_fromfile).tooltip(
            "Specifiy a .txt file with video paths, one per line")
        ui.label("Add Video from path:").classes("text-lg")
        vid_path_in = ui.input(placeholder="Video Location")
        ui.button("Add Video", icon='add', color="secondary",
                  on_click=lambda: video_table.add_row({"video": vid_path_in.value.split('\\')[-1], "path": vid_path_in.value}))
        # ui.image("NicePoseGUI.png").classes("w-1/6 absolute-top-right")

    with ui.dropdown_button("Video Options", icon="menu", color="secondary"):
        ui.item("Delete Selected Videos", on_click=lambda: delete_selected_rows(video_table))
        ui.item("Select People in Selected Videos",
                on_click=lambda: select_people(video_table, selected=True))

    ui.separator()

    video_table = ui.table(
        columns=[
            {"name": "video", "label": "Video", "field": "video"},
            {"name": "path", "label": "Path", "field": "path"},
            {"name": "ids", "label": "IDs:Names", "field": "ids"},
            {"name": "progress", "label": "Progress", "field": "progress"},
        ],
        rows=[],
        selection="multiple",
        row_key="path",
        column_defaults={
            'align': 'left',
            'headerClasses': 'uppercase text-primary',
        }
    ).props("virtual-scroll").classes("h-72 w-3/4")

    prog_bar = ui.linear_progress(show_value=False, color="secondary",
                                  value=0).classes("w-1/2")
    ui.timer(0.1, lambda: prog_bar.set_value(queue.get() if not queue.empty() else prog_bar.value))

    ui.separator()

    ui.label()
    with ui.row(align_items='center'):
        ui.label("Select Model:").classes("text-lg")
        opt_modelSelect = ui.select(yolo_models, value=yolo_models[0])
        opt_GpuCpu = ui.switch('GPU', value=True)
        opt_display = ui.switch("Display", value=False)
        opt_Verbose = ui.switch("Verbose", value=True)
        opt_WriteVideo = ui.switch("Write Video", value=False)

    ui.button("Select People", icon="group", on_click=lambda e: select_people(video_table))

    with ui.row(align_items='center'):
        ui.button("Run Pose Detection", icon="play_arrow",
                  on_click=lambda e: run_pose_detection(video_table))


ui.run()
