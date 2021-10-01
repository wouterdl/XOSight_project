# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Generate offline synthetic dataset
"""


import asyncio
import copy
import os
import torch
import signal

import carb
import omni
from omni.isaac.python_app import OmniKitHelper
#from pxr import Usd, Sdf
#from pxr import UsdGeom

# Default rendering parameters
RENDER_CONFIG = {
    "renderer": "RayTracedLighting",
    "samples_per_pixel_per_frame": 12,
    "headless": True,
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 256,
    "height": 256,
}


class RandomScenario(torch.utils.data.IterableDataset):
    def __init__(self, scenario_path, writer_mode, data_dir, max_queue_size, train_size, classes):

        self.kit = OmniKitHelper(config=RENDER_CONFIG)
        from omni.isaac.synthetic_utils import SyntheticDataHelper, DataWriter, KittiWriter, DomainRandomization

        self.sd_helper = SyntheticDataHelper()
        self.dr_helper = DomainRandomization()
        self.writer_mode = writer_mode
        self.writer_helper = KittiWriter if writer_mode == "kitti" else DataWriter
        self.dr_helper.toggle_manual_mode()
        self.stage = self.kit.get_stage()
        self.result = True

        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        if scenario_path is None:
            self.result, nucleus_server = find_nucleus_server()
            if self.result is False:
                carb.log_error("Could not find nucleus server with /Isaac folder")
                return
            self.asset_path = nucleus_server + "/Library"
            #scenario_path = self.asset_path + "/Samples/Synthetic_Data/Stage/warehouse_with_sensors.usd"
            scenario_path = self.asset_path + "/Simple_Warehouse/warehouse_GUI.usd"
        self.scenario_path = scenario_path
        self.max_queue_size = max_queue_size
        self.data_writer = None
        self.data_dir = data_dir
        self.train_size = train_size
        self.classes = classes

        self._setup_world(scenario_path)
        self.cur_idx = 0
        self.exiting = False
        self._sensor_settings = {}

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)

    def setup_dr(self):
        from pxr import Gf

        #DR variables
        cam_ymin = -220
        cam_ymax = 1000
        camera_points_list = []


        light_prim_path = ["/Root/SM_LampCeilingA_23/RectLight", "/Root/SM_LampCeilingA_24/RectLight", "/Root/SM_LampCeilingA_33/RectLight"]

        # self.light_comp = self.dr_helper.create_light_comp(light_prim_path, first_color_range=(0.9, 0., 0.), second_color_range=(1.0, 0.0, 0.0))

        # result, prim = omni.kit.commands.execute(
        #     "CreateLightComponentCommand",
        #     light_paths=[light_prim_path],
        #     first_color_range=(1, 1., 1.),
        #     second_color_range=(1.0, 1.0, 1.0),
        #     intensity_range=(0, 70000.0),
        #     temperature_range=(1500.0, 6500.0),
        #     enable_temperature=True,
        #     duration=1.0,
        #     include_children=False,
        #     )
        
        


        for i in range(cam_ymin, cam_ymax):
            
            camera_points_list.append(Gf.Vec3f(-250, i, 500))
            # camera_vector = Gf.Vec3f(-250, i, 500)
            # prim = "/Root/Camera_01"
            # UsdGeom.XformCommonAPI(prim).SetTranslate(camera_vector)

        # result, prim = omni.kit.commands.execute(
        #     "CreateTransformComponentCommand",
        #     prim_paths="/Root/Camera_01",
        #     #prim_paths="/Root/SM_WallA_InnerCorner34/SM_WallB_6M",
        #     path='/Root/Camera_01/transform_component_0',
        #     target_points=camera_points_list,
        #     enable_sequential_behavior=False
        #     )

        # omni.kit.commands.execute('AddRelationshipTarget',
        # relationship=Usd.Prim(</Root/Camera_01/transform_component_0>).GetRelationship('primPaths'),
        # target=Sdf.Path('/Root/Camera_01'))





        # result, prim = omni.kit.commands.execute(
        #     'CreateVisibilityComponentCommand',
        #     prim_paths=['/Root/Group_05/SM_PaletteA_359/SM_PaletteA_01'],
        #     num_visible_range=(0, 0),
        #     duration=1.0,
        #     include_children=True,
        #     seed=12345
        # )

    def _setup_world(self, scenario_path):
        # Load scenario
        setup_task = asyncio.ensure_future(self.load_stage(scenario_path))
        while not setup_task.done():
            self.kit.update()
        ###USE DR FUNCTION HERE###
        self.setup_dr()
        self.kit.update()
        self.kit.setup_renderer()
        self.kit.update()

    def __iter__(self):
        return self

    def __next__(self):
        # step once and then wait for materials to load
        self.dr_helper.randomize_once()
        self.kit.update()
        while self.kit.is_loading():
            self.kit.update()

        # Enable/disable sensor output and their format
        self._enable_rgb = True
        self._enable_depth = True
        self._enable_instance = False
        self._enable_semantic = True
        self._enable_bbox_2d_tight = True
        self._enable_bbox_2d_loose = True 
        self._enable_depth_colorize = True
        self._enable_instance_colorize = True
        self._enable_semantic_colorize = True
        self._enable_bbox_2d_tight_colorize = True
        self._enable_bbox_2d_loose_colorize = True
        self._enable_depth_npy = True
        self._enable_instance_npy = False
        self._enable_semantic_npy = True
        self._enable_bbox_2d_tight_npy = True
        self._enable_bbox_2d_loose_npy = True
        self._num_worker_threads = 4
        self._output_folder = self.data_dir

        sensor_settings_viewport = {
            "rgb": {"enabled": self._enable_rgb},
            "depth": {
                "enabled": self._enable_depth,
                "colorize": self._enable_depth_colorize,
                "npy": self._enable_depth_npy,
            },
            "instance": {
                "enabled": self._enable_instance,
                "colorize": self._enable_instance_colorize,
                "npy": self._enable_instance_npy,
            },
            "semantic": {
                "enabled": self._enable_semantic,
                "colorize": self._enable_semantic_colorize,
                "npy": self._enable_semantic_npy,
            },
            "bbox_2d_tight": {
                "enabled": self._enable_bbox_2d_tight,
                "colorize": self._enable_bbox_2d_tight_colorize,
                "npy": self._enable_bbox_2d_tight_npy,
            },
            "bbox_2d_loose": {
                "enabled": self._enable_bbox_2d_loose,
                "colorize": self._enable_bbox_2d_loose_colorize,
                "npy": self._enable_bbox_2d_loose_npy,
            },
        }
        self._sensor_settings["Viewport"] = copy.deepcopy(sensor_settings_viewport)

        # Write to disk
        if self.data_writer is None:
            print(f"Writing data to {self._output_folder}")
            if self.writer_mode == "kitti":
                self.data_writer = self.writer_helper(
                    self._output_folder, self._num_worker_threads, self.max_queue_size, self.train_size, self.classes
                )
            else:
                self.data_writer = self.writer_helper(
                    self._output_folder, self._num_worker_threads, self.max_queue_size, self._sensor_settings
                )
            self.data_writer.start_threads()

        viewport_iface = omni.kit.viewport.get_viewport_interface()
        viewport_name = "Viewport"
        viewport = viewport_iface.get_viewport_window(viewport_iface.get_instance(viewport_name))
        groundtruth = {
            "METADATA": {
                "image_id": str(self.cur_idx),
                "viewport_name": viewport_name,
                "DEPTH": {},
                "INSTANCE": {},
                "SEMANTIC": {},
                "BBOX2DTIGHT": {},
                "BBOX2DLOOSE": {},
            },
            "DATA": {},
        }

        gt_list = []

        if self._enable_rgb:
            gt_list.append("rgb")
        if self._enable_depth:
            gt_list.append("depthLinear")
        if self._enable_bbox_2d_tight:
            gt_list.append("boundingBox2DTight")
        if self._enable_bbox_2d_loose:
            gt_list.append("boundingBox2DLoose")
        if self._enable_instance:
            gt_list.append("instanceSegmentation")
        if self._enable_semantic:
            gt_list.append("semanticSegmentation")

        # gt_list.append("normals")


        # Render new frame
        self.kit.update()

        # Collect Groundtruth
        gt = self.sd_helper.get_groundtruth(gt_list, viewport)

        # RGB
        image = gt["rgb"]
        if self._enable_rgb:
            groundtruth["DATA"]["RGB"] = gt["rgb"]

        # Depth
        if self._enable_depth:
            groundtruth["DATA"]["DEPTH"] = gt["depthLinear"].squeeze()
            groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = self._enable_depth_colorize
            groundtruth["METADATA"]["DEPTH"]["NPY"] = self._enable_depth_npy

        # Instance Segmentation
        if self._enable_instance:
            instance_data = gt["instanceSegmentation"][0]
            instance_data_shape = instance_data.shape
            groundtruth["DATA"]["INSTANCE"] = instance_data
            groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data_shape[1]
            groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data_shape[0]
            groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self._enable_instance_colorize
            groundtruth["METADATA"]["INSTANCE"]["NPY"] = self._enable_instance_npy

        # Semantic Segmentation
        if self._enable_semantic:
            semantic_data = gt["semanticSegmentation"]
            semantic_data_shape = semantic_data.shape
            groundtruth["DATA"]["SEMANTIC"] = semantic_data
            groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data_shape[1]
            groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data_shape[0]
            groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = self._enable_semantic_colorize
            groundtruth["METADATA"]["SEMANTIC"]["NPY"] = self._enable_semantic_npy

        # 2D Tight BBox
        if self._enable_bbox_2d_tight:
            groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = self._enable_bbox_2d_tight_colorize
            groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = self._enable_bbox_2d_tight_npy

        # 2D Loose BBox
        if self._enable_bbox_2d_loose:
            groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = self._enable_bbox_2d_loose_colorize
            groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = self._enable_bbox_2d_loose_npy
            groundtruth["METADATA"]["BBOX2DLOOSE"]["WIDTH"] = RENDER_CONFIG["width"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["HEIGHT"] = RENDER_CONFIG["height"]

        # Normals
        # groundtruth["DATA"]["NORMALS"] = gt["normals"]

        self.data_writer.q.put(groundtruth)

        self.cur_idx += 1
        return image


if __name__ == "__main__":
    "Typical usage"
    import argparse

    parser = argparse.ArgumentParser("Dataset generator")
    parser.add_argument("--scenario", type=str, help="Scenario to load from omniverse server")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to record")
    parser.add_argument("--writer_mode", type=str, default="npy", help="Specify output format - npy or kitti")
    parser.add_argument(
        "--data_dir", type=str, default=os.getcwd() + "/testing/output", help="Location where data will be output"
    )
    parser.add_argument("--max_queue_size", type=int, default=500, help="Max size of queue to store and process data")
    parser.add_argument(
        "--train_size", type=int, default=8, help="Number of frames for training set, works when writer_mode is kitti"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=[],
        help="Which classes to write labels for, works when writer_mode is kitti.  Defaults to all classes",
    )
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    dataset = RandomScenario(
        args.scenario, args.writer_mode, args.data_dir, args.max_queue_size, args.train_size, args.classes
    )

    if dataset.result:
        # Iterate through dataset and visualize the output
        print("Loading materials. Will generate data soon...")
        for image in dataset:
            print("ID: ", dataset.cur_idx)
            if dataset.cur_idx == args.num_frames:
                break
            if dataset.exiting:
                break
        # cleanup
        dataset.kit.shutdown()



























# # Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.

# import time
# import os
# from omni.isaac.python_app import OmniKitHelper
# import carb
# import omni
# import omni.kit.commands
# #import omni.isaac.dr as dr
# #dr_interface = dr._dr.acquire_dr_interface()
# #from pxr import Sdf, Usd
# #from omni.isaac.synthetic_utils import SyntheticDataHelper, DomainRandomization

# # import omni.kit
# # import omni.usd

# # from pxr import Sdf

# # This sample loads a usd stage and starts simulation
# print('test1')
# CONFIG = {
#     "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
#     "width": 1280,
#     "height": 720,
#     "sync_loads": True,
#     "headless": False,
#     "renderer": "RayTracedLighting",
# }
# print('test2')
# if __name__ == "__main__":
#     import argparse

#     #Set variables
#     light_prim_path = "/Warehouse/SM_LampCeilingA_23/RectLight"

#     # Set up command line arguments
#     parser = argparse.ArgumentParser("Usd Load sample")
#     parser.add_argument("--usd_path", type=str, help="Path to usd file", required=True)
#     parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
#     parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

#     args, unknown = parser.parse_known_args()
#     # Start the omniverse application
#     CONFIG["headless"] = args.headless
#     kit = OmniKitHelper(config=CONFIG)
#     print('111111111111111111111111111111111111111111111111111111111111111111')
#     # Locate /Isaac folder on nucleus server to load sample
#     from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
#     print('222222222222222222222222222222222222222222222222222222222222222222222')
#     result, nucleus_server = find_nucleus_server()
#     if result is False:
#         carb.log_error("Could not find nucleus server with /Isaac folder, exiting")
#         exit()
#     asset_path = nucleus_server + "/Isaac"
#     usd_path = asset_path + args.usd_path
#     omni.usd.get_context().open_stage(usd_path, None)

#     print('33333333333333333333333333333333333333333333333333333333333333333333333')
#     #time.sleep(200)
#     # Wait two frames so that stage starts loading
#     kit.app.update()
#     kit.app.update()
#     #ime.sleep(100) 
#     print('4444444444444444444444444444444444444444444444444444444444444444444444444444')
#     print("Loading stage...")
#     while kit.is_loading():
#         kit.update(1.0 / 60.0)
#     print("Loading Complete")

#     print('5555555555555555555555555555555555555555555555555555555555555555555555555')
#     kit.play()
#     print('66666666666666666666666666666666666666666666666666666666666666666666666')
#     #time.sleep(200)

#     result, prim = omni.kit.commands.execute(
#             "CreateLightComponentCommand",
#             light_paths=[light_prim_path],
#             first_color_range=(0.9, 0.9, 0.9),
#             second_color_range=(1.0, 1.0, 1.0),
#             intensity_range=(0, 70000.0),
#             temperature_range=(1500.0, 6500.0),
#             enable_temperature=True,
#             duration=1.0,
#             include_children=False,
#             )

#     # Run in test mode, exit after a fixed number of steps
#     if args.test is True:
#         for i in range(10):
#             # Run in realtime mode, we don't specify the step size
#             kit.update()
#     else:
#         while kit.app.is_running():
#             # Run in realtime mode, we don't specify the step size
            
            
            
#             kit.update()

#     kit.stop()
#     kit.shutdown()



# # import omni.kit.commands
# # from pxr import Sdf, Usd

# # omni.kit.commands.execute('AddRelationshipTarget',
# # 	relationship=Usd.Prim(</Warehouse/SM_LampCeilingA_23/RectLight/light_component_0>).GetRelationship('primPaths'),
# # 	target=Sdf.Path('/Warehouse/SM_LampCeilingA_23/RectLight'))

# # omni.kit.commands.execute('CreateLightComponentCommand',
# # 	path='/Warehouse/SM_LampCeilingA_23/RectLight/light_component_0',
# # 	light_paths=[],
# # 	first_color_range=(0.0, 0.0, 0.0),
# # 	second_color_range=(1.0, 1.0, 1.0),
# # 	intensity_range=(40000.0, 70000.0),
# # 	temperature_range=(6500.0, 6500.0),
# # 	enable_temperature=False,
# # 	duration=1.0,
# # 	include_children=False,
# # 	seed=12345)