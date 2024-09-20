# 使用mayavi可视化点云，可以指定一系列参数，如相机参数、渲染图像大小、支持渲染单个点云文件，也支持渲染一个文件夹下的所有点云（支持多种格式输入），并保存结果，参数包括输入path和输出path
import os
import numpy as np
import trimesh
from mayavi import mlab
import subprocess



def normalize_points(points):
    # 计算最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 归一化操作：将所有点的值缩放到 [0, 1] 范围
    normalized_points = (points - min_vals) / (max_vals - min_vals) - 0.5
    return normalized_points


def check_path(path):
    # 将相对路径转换为绝对路径
    absolute_path = os.path.abspath(path)
    
    # 判断路径是否存在
    if not os.path.exists(absolute_path):
        # 如果路径不存在，创建路径
        os.makedirs(absolute_path)
        print(f"路径不存在，已创建路径: {absolute_path}")
    else:
        print(f"路径已存在: {absolute_path}")


import os
import subprocess

def images_to_gif(path, fps, output_name="output.gif"):
    # 检查输入路径是否存在
    if not os.path.exists(path):
        raise ValueError(f"路径 {path} 不存在")
    
    # 使用 ffmpeg 进行图片合成 gif
    # 假设图片的命名为 img_001.png, img_002.png, img_003.png 等等
    # 根据你的图片命名格式调整 pattern
    pattern = os.path.join(path, "frame_%d.png")
    
    # 构建 ffmpeg 命令
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),         # 设置帧率
        "-i", pattern,                  # 输入图片的路径模式
        "-vf", "scale=800:-1",          # 可选：调整gif大小，800px宽度，高度根据宽度自动调整
        "-loop", "0",                   # 设置 gif 循环播放，0 表示无限循环
        f"outputs/ssdr/{output_name}"                     # 输出文件名
    ]
    
    # 调用 subprocess 来执行命令
    try:
        subprocess.run(cmd, check=True)
        print(f"GIF 成功生成，保存路径为: {output_name}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 命令执行失败: {e}")

# 示例调用
# images_to_gif("outputs/ssdr/images", fps=10, output_name="animation.gif")




def visualize_and_save_point_cloud(input_path, output_path, camera_params=None, image_size=(800, 600)):
    # 检查输入路径是文件还是文件夹
    if os.path.isdir(input_path):
        point_cloud_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.ply', '.obj', '.stl'))]
    else:
        point_cloud_files = [input_path]

    # 创建输出文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in point_cloud_files:
        # 使用trimesh加载点云
        mesh = trimesh.load(file)
        
        # 获取点云数据
        points = np.array(mesh.vertices)
        
        # 使用Mayavi可视化点云
        mlab.figure(bgcolor=(1, 1, 1), size=image_size)
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 1))

        # 设置相机参数（可选）
        if camera_params:
            mlab.view(azimuth=camera_params.get('azimuth', 0),
                      elevation=camera_params.get('elevation', 90),
                      distance=camera_params.get('distance', 10),
                      focalpoint=camera_params.get('focalpoint', (0, 0, 0)))

        # 构建输出文件路径
        output_file = os.path.join(output_path, os.path.splitext(os.path.basename(file))[0] + '.png')

        # 保存渲染图像
        mlab.savefig(output_file)
        mlab.close()

    print(f"Rendered images saved to: {output_path}")


def visualize_multiple_point_clouds(input_paths, output_path, colors=None, opacity=None, sizes=None, camera_params=None, image_size=(800, 600)):
    """
    在同一个图像上渲染多个点云，并为每个点云指定颜色和大小。
    
    参数:
    - input_paths: list of str, 点云文件路径列表（支持文件夹路径或文件路径）
    - output_path: str, 输出渲染图像的路径
    - colors: list of tuples, 每个点云的颜色（RGB格式），例如 [(1, 0, 0), (0, 1, 0)]
    - sizes: list of floats, 每个点云的点大小，例如 [0.1, 0.2]
    - camera_params: dict, 相机参数
    - image_size: tuple, 渲染图像大小
    """
    # 初始化 Mayavi 图像
    mlab.figure(bgcolor=(1, 1, 1), size=image_size)

    # 确保颜色和大小列表与输入路径数量一致
    if colors is None:
        colors = [(1, 0, 0)] * len(input_paths)  # 默认红色
    if sizes is None:
        sizes = [5] * len(input_paths)  # 默认大小

    for idx, path in enumerate(input_paths):
        # 检查是否是文件夹路径
        if os.path.isdir(path):
            point_cloud_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.ply', '.obj', '.stl'))]
        else:
            point_cloud_files = [path]

        for file in point_cloud_files:
            # 使用trimesh加载点云
            mesh = trimesh.load(file)
            points = np.array(mesh.vertices)

            # 渲染点云
            mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=colors[idx], scale_factor=sizes[idx],opacity=opacity[idx])

    # 设置相机参数（可选）
    if camera_params:
        mlab.view(azimuth=camera_params.get('azimuth', 90),
                  elevation=camera_params.get('elevation', 90),
                  distance=camera_params.get('distance', 10),
                  focalpoint=camera_params.get('focalpoint', (0, 0, 0)))

    # 保存渲染图像
    # mlab.show()
    mlab.savefig(output_path)
    mlab.close()

    print(f"Rendered image saved to: {output_path}")


def reconstruct(rest_pose, bone_transforms, rest_bones_t, W):
    """
    Computes the skinned vertex positions on some poses given bone transforms and weights.

    inputs : rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
             bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
             rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
             W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.

    return: |num_poses| x |num_verts| x 3 Vertex positions for all poses: sum{bones} (w * (R @ p + T)) 
            |num_bones| x |num_poses| x 3 Max weighted vertex positions
    """
    # Points in rest pose without rest bone translations
    p_corrected = rest_pose[np.newaxis, :, :] - rest_bones_t[:, np.newaxis, :]  # |num_bones| x |num_verts| x 3

    # Compute the transformed vertex positions
    constructions = np.einsum('bijk,blk->bilj', bone_transforms[:, :, :3, :],
                              p_corrected)  # |num_bones| x |num_poses| x |num_verts| x 3
    constructions += bone_transforms[:, :, np.newaxis, 3, :]  # |num_bones| x |num_poses| x |num_verts| x 3
    constructions *= (W.T)[:, np.newaxis, :, np.newaxis]  # |num_bones| x |num_poses| x |num_verts| x 3
    vertex_positions = np.sum(constructions, axis=0)  # |num_poses| x |num_verts| x 3

    # 从W数组中提取出每个骨骼权重最大的点
    # 找到每个骨骼在每个姿势下最大权重的顶点索引
    max_weight_indices = np.argmax(W, axis=0)  # |num_verts| x |num_bones|
    print("Max Weights:",W[max_weight_indices[0],0],W[max_weight_indices[1],1],W[max_weight_indices[2],2],W[max_weight_indices[3],3],W[max_weight_indices[4],4],W[max_weight_indices[5],5],W[max_weight_indices[6],6],W[max_weight_indices[7],7],W[max_weight_indices[8],8],W[max_weight_indices[9],9])

    # 使用这些索引从vertex_positions中提取对应的点
    max_weight_points = np.zeros((bone_transforms.shape[1], bone_transforms.shape[0], 3))
    for bone in range(W.shape[1]):
        for pose in range(vertex_positions.shape[0]):
            idx = max_weight_indices[bone]
            max_weight_points[pose, bone] = vertex_positions[pose, idx]

    return vertex_positions


def get_transformed_bones(bone_transforms, rest_bones_t):
    # # 提取旋转部分
    # rotation_matrices = bone_transforms[:, :, :3, :]
    
    # # 计算变换后的骨骼位置
    # # transformed_bones = np.einsum('bijk,bk->bik', rotation_matrices, rest_bones_t) + bone_transforms[:, :, 3, :] # Bones, Frames, 3
    # transformed_bones = np.einsum('bmjk,bi->bmi', rotation_matrices, rest_bones_t) + bone_transforms[:, :, 3, :]
    num_poses = bone_transforms.shape[1]
    num_bones = rest_bones_t.shape[0]
    # 初始化结果张量
    result = np.zeros((num_poses, num_bones, 3))

    # 对每个姿势和每个骨骼进行矩阵乘法
    for pose in range(num_poses):
        for bone in range(num_bones):
            # 取出当前姿势和骨骼的旋转矩阵
            rotation_matrix = bone_transforms[bone, pose,:3]
            # 对 rest_bones_t 进行矩阵乘法
            result[pose, bone] = np.dot(rotation_matrix, rest_bones_t[bone]) + bone_transforms[bone,pose,3]

    
    return result  # 形状为 |num_bones| x |num_poses| x 3


def apply_bone_transforms(rest_pose, rest_bone, W, bone_trans):
    num_frames = bone_trans.shape[1]
    transformed_points = []
    transformed_bones = []

    for t in range(num_frames):
        transformed_frame = np.zeros_like(rest_pose)
        transformed_bones_frame = []
        
        # for all pose : (w * (R @ p + T)) 
        for i in range(rest_bone.shape[0]):
            transform = bone_trans[i, t] # 获取每个bone的变换矩阵
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            
            # 计算每个骨骼变换后的位置
            bone_position = (rotation @ rest_bone.T).T + translation
            transformed_bones_frame.append(bone_position)

            # 将骨骼的变换应用到点云上
            transformed_frame += W[:, i:i+1] * ((rotation @ rest_pose.T).T + translation)
        transformed_points.append(transformed_frame)
        transformed_bones.append(transformed_bones_frame)

    return transformed_points, transformed_bones


def visualize_SSDR_result_pc():
    results = np.load("/home/wjx/research/code/DG-Mesh/outputs/ssdr/jumping/ssdr_result.npy",allow_pickle=True).item()
    rest_pose = results["rest_pose"]
    W = results["W"]
    rest_bone = results["rest_bone"]
    bone_trans = results["Bone_Trans"]

    print("W shape:", W.shape)
    print("Bone Transforms shape:", bone_trans.shape)
    print("Rest Bones shape:", rest_bone.shape)
    print("Rest Pose shape:", rest_pose.shape)

    # for i in range(bone_trans.shape[0]):
    #     bone_index = i  # 选择一个骨骼的索引
    #     weights = W[:, bone_index]  # 获取对应骨骼的weights

    #     mlab.figure(bgcolor=(1, 1, 1))
    #     points = rest_pose
    #     # 通过颜色显示skin weights
    #     mlab.points3d(points[:, 0], points[:, 1], points[:, 2], weights, scale_factor=0.01, colormap='viridis')
    #     mlab.title(f"Skinning Weights for Bone {bone_index}")
    #     mlab.show()
    # 计算变换后的点云
    # transformed_points,transformed_bones = apply_bone_transforms(rest_pose, rest_bone, W, bone_trans)
    transformed_points = reconstruct(rest_pose,bone_trans,rest_bone,W)
    transformed_bones = bone_trans[:,:,3,:].transpose(1,0,2)
    # transformed_bones = get_transformed_bones(bone_trans,rest_bone).transpose(1,0,2)

    # print(transformed_points.shape,transformed_bones.shape)

    # 配置绘制参数
    colors = [(1, 0, 0), (0, 0, 1)]  # 每个点云的颜色 (红色和绿色)
    opacity = [0.5,1]
    sizes = [0.01, 0.1]  # 每个点云的大小
    camera_params = {
        'azimuth': -90,
        'elevation': 75,
        'distance': 6,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小
    output_path = "outputs/ssdr/jumping/results_pc"
    check_path(output_path)

    for i in range(transformed_points.shape[0]):
        frame_points = transformed_points[i]
        frame_bones = transformed_bones[i]
        mlab.figure(bgcolor=(1, 1, 1), size=image_size)
        mlab.clf()
        

        mlab.options.offscreen = True
        mlab.points3d(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2], color=colors[0], scale_factor=sizes[0],opacity=opacity[0])
        frame_bones = np.array(frame_bones)
        # print(frame_bones.shape)
        mlab.points3d(frame_bones[:, 0], frame_bones[:, 1], frame_bones[:, 2], color=colors[1], scale_factor=sizes[1],opacity=opacity[1])
        mlab.view(azimuth=camera_params.get('azimuth', 90),
                  elevation=camera_params.get('elevation', 90),
                  distance=camera_params.get('distance', 10),
                  focalpoint=camera_params.get('focalpoint', (0, 0, 0)))
        
        mlab.savefig(output_path + f"/frame_{i}.png")
        print(f"image{i} has been saved in {output_path}")
        mlab.close()
        



def single_pc_exp():
    # 单点云可视化示例/文件夹点云可视化示例
    input_path = "/home/wjx/research/code/DG-Mesh/inputs/mesh_selected_inner_points.obj"  # 输入路径，支持文件夹或单个文件
    output_path = "/home/wjx/research/code/DG-Mesh/outputs/ssdr/render_pc"  # 输出路径，保存渲染结果
    camera_params = {
        'azimuth': 90,
        'elevation': 75,
        'distance': 10,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小

    visualize_and_save_point_cloud(input_path, output_path, camera_params, image_size)


def multi_pc_exp():
    # 多个点云可视化示例
    input_paths = ["/home/wjx/research/code/DG-Mesh/outputs/ssdr/pc/rest_pose.ply", "/home/wjx/research/code/DG-Mesh/outputs/ssdr/pc/rest_bone.ply"]  # 输入的点云文件路径列表
    output_path = "/home/wjx/research/code/DG-Mesh/outputs/ssdr/render_pc/rendered_2pcimage.png"  # 输出渲染图像的路径
    colors = [(1, 0, 0), (0, 0, 1)]  # 每个点云的颜色 (红色和绿色)
    opacity = [0.5,1]
    sizes = [0.01, 0.1]  # 每个点云的大小
    camera_params = {
        'azimuth': -90,
        'elevation': 75,
        'distance': 6,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小

    visualize_multiple_point_clouds(input_paths, output_path, colors, opacity, sizes, camera_params, image_size)

def visualize_B():
    results = np.load("/home/wjx/research/code/DG-Mesh/outputs/ssdr/ssdr_result.npy",allow_pickle=True).item()
    rest_pose = results["rest_pose"]
    W = results["W"]
    rest_bone = results["rest_bone"]
    bone_trans = results["Bone_Trans"]
    print("W shape:", W.shape)
    print("Bone Transforms shape:", bone_trans.shape)
    print("Rest Bones shape:", rest_bone.shape)
    print("Rest Pose shape:", rest_pose.shape)

    # for i in range(bone_trans.shape[0]):
    #     bone_index = i  # 选择一个骨骼的索引
    #     weights = W[:, bone_index]  # 获取对应骨骼的weights

    #     mlab.figure(bgcolor=(1, 1, 1))
    #     points = rest_pose
    #     # 通过颜色显示skin weights
    #     mlab.points3d(points[:, 0], points[:, 1], points[:, 2], weights, scale_factor=0.01, colormap='viridis')
    #     mlab.title(f"Skinning Weights for Bone {bone_index}")
    #     mlab.show()
    # 计算变换后的点云
    # transformed_points,transformed_bones = apply_bone_transforms(rest_pose, rest_bone, W, bone_trans)
    # transformed_points = reconstruct(rest_pose,bone_trans,rest_bone,W)
    transformed_bones = get_transformed_bones(bone_trans,rest_bone).transpose(1,0,2)

    # print(transformed_points.shape,transformed_bones.shape)

    # 配置绘制参数
    colors = [(1, 0, 0), (0, 0, 1)]  # 每个点云的颜色 (红色和绿色)
    opacity = [0.5,1]
    sizes = [0.01, 0.1]  # 每个点云的大小
    camera_params = {
        'azimuth': -90,
        'elevation': 75,
        'distance': 6,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小
    output_path = "outputs/ssdr/results_B"
    check_path(output_path)

    for i in range(transformed_bones.shape[0]):
        frame_bones = transformed_bones[i]
        mlab.figure(bgcolor=(1, 1, 1), size=image_size)
        mlab.clf()
        

        mlab.options.offscreen = True
        # mlab.points3d(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2], color=colors[0], scale_factor=sizes[0],opacity=opacity[0])
        frame_bones = np.array(frame_bones)
        # print(frame_bones.shape)
        mlab.points3d(frame_bones[:, 0], frame_bones[:, 1], frame_bones[:, 2], color=colors[1], scale_factor=sizes[1],opacity=opacity[1])
        mlab.view(azimuth=camera_params.get('azimuth', 90),
                  elevation=camera_params.get('elevation', 90),
                  distance=camera_params.get('distance', 10),
                  focalpoint=camera_params.get('focalpoint', (0, 0, 0)))
        
        mlab.savefig(output_path + f"/frame_{i}.png")
        print(f"image{i} has been saved in {output_path}")
        mlab.close()
    

visualize_SSDR_result_pc()
# visualize_B()
images_to_gif(path="/home/wjx/research/code/DG-Mesh/outputs/ssdr/jumping/results_pc",fps=10,output_name="jumping/output_trans.gif")