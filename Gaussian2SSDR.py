#  created by Isabella Liu (lal005@ucsd.edu) at 2024/05/29 18:21.
#
#  Rendering the trained model on the test dataset


import os, sys
import json
import datetime
import os.path as osp
import torch
import uuid
import datetime
from tqdm import tqdm
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mayavi import mlab
import trimesh

from own_ssdr import SSDR
from scene import Scene
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene import DeformModelNormalSep as deform_model_sep
from scene import AppearanceModel as appearance_model
from utils.renderer import mesh_renderer, mesh_shape_renderer
from utils.general_utils import safe_state
from utils.system_utils import load_config_from_file, merge_config
from arguments import ModelParams, PipelineParams, OptimizationParams

import nvdiffrast.torch as dr

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_pointcloud(pc,name="point_cloud.ply"):
    cloud = trimesh.points.PointCloud(pc)
    cloud.export(name)



@torch.no_grad()
def rendering_trajectory(dataset, opt, pipe, checkpoint, fps=24):
    args.model_path = dataset.model_path

    # Load models
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    glctx = dr.RasterizeGLContext()
    scene = Scene(dataset, gaussians, shuffle=False)
    ## Deform forward model
    deform = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform"
    )
    deform_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_normal",
    )
    ## Deform backward model
    deform_back = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform_back"
    )
    deform_back_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_back_normal",
    )
    ## Appearance model
    appearance = appearance_model(is_blender=dataset.is_blender)
    ## Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=-1)
        deform.load_weights(checkpoint, iteration=-1)
        deform_normal.load_weights(checkpoint, iteration=-1)
        deform_back.load_weights(checkpoint, iteration=-1)
        deform_back_normal.load_weights(checkpoint, iteration=-1)
        appearance.load_weights(checkpoint, iteration=-1)

    # Compose camera trajectory
    viewpoint_cam_stack = scene.getTrainCameras().copy()
    # Create folders
    image_folder = osp.join(dataset.model_path, "images")
    os.makedirs(image_folder, exist_ok=True)
    final_images = []
    gaussian_point_cloud = []

    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_cam_stack)):

        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians
        d_xyz, d_rotation, d_scaling, _ = deform.step(
            gaussians.get_xyz.detach(), time_input
        )
        gaussian_point_cloud.append(gaussians.get_xyz.detach().cpu() + d_xyz.detach().cpu())
    
    # 拿到gaussian点云序列之后，送进SSDR
    
    # read skeleton from Converage Axis
    Init_Bone = trimesh.load("/home/wjx/research/code/Coverage_Axis/output/mesh_selected_inner_points.obj").vertices
    # print(Init_Bone.shape)
    poses = np.array(gaussian_point_cloud)
    rest_pose = poses[0]
    # save_pointcloud(rest_pose,"outputs/ssdr/pc/pc.ply")



    W, bone_transforms, rest_bones_t = SSDR(poses, rest_pose, num_bones=Init_Bone.shape[0],initial_Bone=Init_Bone)
    np.save("outputs/ssdr/ssdr_result.npy",{"W":W, "Bone_Trans":bone_transforms, "rest_bone":rest_bones_t,"rest_pose":rest_pose})
    ssdr_visualization(rest_pose,rest_bones_t,W,bone_transforms)

    # 对B和W进行可视化
def ssdr_visualization(rest_pose=None,rest_bone=None,W=None,bone_trans=None,eval=False):
    if eval:
        results = np.load("./ssdr_result.npy",allow_pickle=True).item()
        rest_pose = results["rest_pose"]
        W = results["W"]
        rest_bone = results["rest_bone"]
        bone_trans = results["Bone_Trans"]
        print("W shape:", W.shape)
        print("Bone Transforms shape:", bone_trans.shape)
        print("Rest Bones shape:", rest_bone.shape)
        print("Rest Pose shape:", rest_pose.shape)

    # # 可视化点云
    # mlab.figure(bgcolor=(1, 1, 1))
    # mlab.points3d(rest_pose[:, 0], rest_pose[:, 1], rest_pose[:, 2], scale_factor=0.01)
    # mlab.title("Point Cloud")
    # mlab.show()

    # # 假设bone_transforms是形状为[K, 4, 4]的骨骼变换矩阵
    # # 假设rest_bones_t是rest pose下骨骼的位置

    # K = rest_bone.shape[0]
    # mlab.figure(bgcolor=(1, 1, 1))

    # for i in range(K):
    #     # 获取骨骼在rest pose下的位置
    #     bone_pos = rest_bone[i]
        
    #     # 绘制骨骼的关节
    #     mlab.points3d(bone_pos[0], bone_pos[1], bone_pos[2], scale_factor=0.05, color=(1, 0, 0))
        
    #     # 如果骨骼之间有连接，绘制骨骼
    #     for j in range(i+1, K):
    #         parent_pos = rest_bone[j]
    #         mlab.plot3d([bone_pos[0], parent_pos[0]], [bone_pos[1], parent_pos[1]], [bone_pos[2], parent_pos[2]], color=(0, 0, 1))

    # mlab.title("Skeleton")
    # mlab.show()
    for i in range(bone_trans.shape[0]):
        bone_index = i  # 选择一个骨骼的索引
        weights = W[:, bone_index]  # 获取对应骨骼的weights

        mlab.figure(bgcolor=(1, 1, 1))
        points = rest_pose
        # 通过颜色显示skin weights
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], weights, scale_factor=0.01, colormap='viridis')
        mlab.title(f"Skinning Weights for Bone {bone_index}")
        mlab.show()
    # 计算变换后的点云
    transformed_points = apply_bone_transforms(rest_pose, W, bone_trans)

    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    mlab.view(azimuth=45, elevation=80, distance=10)

    # 迭代每一帧点云，保存为图片
    for i, frame_points in enumerate(transformed_points):
        # 清除之前的点云
        mlab.clf()

        # 绘制当前帧的点云
        mlab.points3d(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2],
                    scale_factor=0.01, color=(0, 0, 1))
        
        # 保存图片
        mlab.savefig(f'outputs/ssdr/frames/frame_{i:03d}.png')

    # 关闭 mayavi 图形窗口
    mlab.close()





# 使用B和W驱动gaussian
# 将骨骼变换应用到点云
def apply_bone_transforms(rest_pose, W, bone_trans):
    num_frames = bone_trans.shape[1]
    transformed_points = []

    for t in range(num_frames):
        transformed_frame = np.zeros_like(rest_pose)
        for i in range(W.shape[1]):
            transform = bone_trans[i, t]
            rotation = transform[:3, :3]
            translation = transform[:3, 2]
            transformed_frame += W[:, i:i+1] *( (rotation @ rest_pose.T).T + translation)
        transformed_points.append(transformed_frame)

    return transformed_points



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--camera_radius", type=float, default=4.0)
    parser.add_argument("--camera_lookat", type=float, nargs="+", default=[0, 0, 0])
    parser.add_argument("--camera_elevation", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--total_frames", type=int, default=100)
    parser.add_argument("--eva", type=bool, default=False)

    
    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])
    if not args.eva:
        # Load config file
        if args.config:
            config_data = load_config_from_file(args.config)
            combined_args = merge_config(config_data, args)
            args = Namespace(**combined_args)

        lp = lp.extract(args)
        op = op.extract(args)
        pp = pp.extract(args)

        # Updating save path
        unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_name = osp.basename(lp.source_path)
        folder_name = f"rendering-traj-{data_name}-{unique_str}"
        if not lp.model_path:
            if os.getenv("OAR_JOB_ID"):
                unique_str = os.getenv("OAR_JOB_ID")
            else:
                unique_str = str(uuid.uuid4())
            lp.model_path = os.path.join("./output/", unique_str[0:10])
        lp.model_path = osp.join(lp.model_path, folder_name)
        # Set up output folder
        print("Output folder: {}".format(lp.model_path))
        os.makedirs(lp.model_path, exist_ok=True)

        # Initialize system state (RNG)
        safe_state(args.quiet)

        # Save all parameters into file
        combined_args = vars(Namespace(**vars(lp), **vars(op), **vars(pp)))
        # Convert namespace to JSON string
        args_json = json.dumps(combined_args, indent=4)
        # Write JSON string to a text file
        with open(osp.join(lp.model_path, "cfg_args.txt"), "w") as output_file:
            output_file.write(args_json)

        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        rendering_trajectory(lp, op, pp, args.start_checkpoint, args.fps)

        # All done
        print("\nRendering complete.")
    else:
        ssdr_visualization(eval=args.eva)






