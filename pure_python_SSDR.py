# SSDR Implementation in Python
# Dalton Omens

# import maya.api.OpenMaya as om
# import pymel.core as pm
import numpy as np
from scipy.optimize import lsq_linear
from scipy.cluster.vq import vq, kmeans, whiten
import time
import os
import trimesh
import mayavi as mlab


def visualize_SSDR_result_pc(iter, W, rest_pose, rest_bone, bone_trans):
    # results = np.load("/home/wjx/research/code/DG-Mesh/outputs/ssdr/ssdr_result.npy", allow_pickle=True).item()
    # rest_pose = results["rest_pose"]
    # W = results["W"]
    # rest_bone = results["rest_bone"]
    # bone_trans = results["Bone_Trans"]
    # print("W shape:", W.shape)
    # print("Bone Transforms shape:", bone_trans.shape)
    # print("Rest Bones shape:", rest_bone.shape)
    # print("Rest Pose shape:", rest_pose.shape)

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
    transformed_points, transformed_bones = apply_bone_transforms(rest_pose, W, bone_trans)

    # 配置绘制参数
    colors = [(1, 0, 0), (0, 0, 1)]  # 每个点云的颜色 (红色和绿色)
    opacity = [0.5, 1]
    sizes = [0.01, 0.1]  # 每个点云的大小
    camera_params = {
        'azimuth': -90,
        'elevation': 75,
        'distance': 6,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小
    output_path = "outputs/ssdr/results_pc"

    for i, (frame_points, frame_bones) in enumerate(zip(transformed_points, transformed_bones)):
        mlab.figure(bgcolor=(1, 1, 1), size=image_size)
        mlab.clf()

        mlab.points3d(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2], color=colors[0],
                      scale_factor=sizes[0], opacity=opacity[0])
        frame_bones = np.array(frame_bones)
        print(frame_bones.shape)
        mlab.points3d(frame_bones[:, 0], frame_bones[:, 1], frame_bones[:, 2], color=colors[1], scale_factor=sizes[1],
                      opacity=opacity[1])
        mlab.view(azimuth=camera_params.get('azimuth', 90),
                  elevation=camera_params.get('elevation', 90),
                  distance=camera_params.get('distance', 10),
                  focalpoint=camera_params.get('focalpoint', (0, 0, 0)))

        mlab.savefig(output_path + f"/frame_{i}.png")
        mlab.close()

def kabsch(P, Q):
    """
    Computes the optimal translation and rotation matrices that minimize the 
    RMS deviation between two sets of points P and Q using Kabsch's algorithm.
    More here: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Inspiration: https://github.com/charnley/rmsd
    
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q
            
    return: A 4 x 3 matrix where the first 3 rows are the rotation and the last is translation
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P                       # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)                  # covariance matrix
    U, S, V = np.linalg.svd(H)                        # SVD
    R = U.dot(V).T                                    # calculate optimal rotation
    if np.linalg.det(R) < 0:                          # correct rotation matrix for             
        V[2,:] *= -1                                  #  right-hand coordinate system
        R = U.dot(V).T                          
    t = centroid_Q - R.dot(centroid_P)                # translation vector
    return np.vstack((R, t))


def initialize(poses, rest_pose, num_bones, iterations=5,initial_bone=None):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T            
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Use k-means to assign bones to vertices
    whitened = whiten(rest_pose)
    # codebook, _ = kmeans(whitened, num_bones)
    codebook = np.array(initial_bone)
    vert_assignments, _ = vq(whitened, codebook) # Bone assignment for each vertex (|num_verts| x 1) 将k-means分配的簇中的点的label分配
    
    # Compute initial random bone transformations
    for bone in range(num_bones):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0) #骨骼中心点
        rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone]) # 算出每个骨骼的刚性变换
    
    for it in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        constructed = np.empty((num_bones, num_poses, num_verts, 3)) # |num_bones| x |num_poses| x |num_verts| x 3
        for bone in range(num_bones):
            Rp = bone_transforms[bone,:,:3,:].dot((rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1)) # |num_poses| x |num_verts| x 3
            # R * p + T
            constructed[bone] = Rp + bone_transforms[bone, :, np.newaxis, 3, :]
        errs = np.linalg.norm(constructed - poses, axis=(1, 3)) # 对每个pose，计算每个骨骼带来的变换的误差的和，每个骨骼会对应一个RT变换，由此来计算误差的和，下次分配的时候，找出该点对应的误差最小的骨骼，进行二次分配
        vert_assignments = np.argmin(errs, axis=0)    
        
        ## Visualization of vertex assignments for bone 0 over iterations
        ## Make 5 copies of an example pose mesh and call them test0, test1...
        #for i in range(num_verts):
        #    if vert_assignments[i] == 0:
        #        pm.select('test{0}.vtx[{1}]'.format(it, i), add=True)
        #print(vert_assignments)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
            rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms, rest_bones_t

def Cluster(bones,points):
    whiten_pose = whiten(points)
    bones = np.array(bones)
    vert_assignments,_ = vq(whiten_pose,bones)

    return vert_assignments





def initialize_w_Bone(poses, rest_pose, num_bones, iterations=5,initial_bone=None):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T            
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Use k-means to assign bones to vertices
    # whitened = whiten(rest_pose)
    # # codebook, _ = kmeans(whitened, num_bones)
    # codebook = np.array(initial_bone)
    # vert_assignments, _ = vq(whitened, codebook) # Bone assignment for each vertex (|num_verts| x 1) 将k-means分配的簇中的点的label分配
    
    vert_assignments = Cluster(initial_bone, rest_pose)

    # Compute initial random bone transformations
    for bone in range(num_bones):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0) #骨骼中心点
        rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone]) # 算出每个骨骼的刚性变换
    
    for it in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        constructed = np.empty((num_bones, num_poses, num_verts, 3)) # |num_bones| x |num_poses| x |num_verts| x 3
        for bone in range(num_bones):
            Rp = bone_transforms[bone,:,:3,:].dot((rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1)) # |num_poses| x |num_verts| x 3
            # R * p + T
            constructed[bone] = Rp + bone_transforms[bone, :, np.newaxis, 3, :]
        errs = np.linalg.norm(constructed - poses, axis=(1, 3)) # 对每个pose，计算每个骨骼带来的变换的误差的和，每个骨骼会对应一个RT变换，由此来计算误差的和，下次分配的时候，找出该点对应的误差最小的骨骼，进行二次分配
        vert_assignments = np.argmin(errs, axis=0)    
        
        ## Visualization of vertex assignments for bone 0 over iterations
        ## Make 5 copies of an example pose mesh and call them test0, test1...
        #for i in range(num_verts):
        #    if vert_assignments[i] == 0:
        #        pm.select('test{0}.vtx[{1}]'.format(it, i), add=True)
        #print(vert_assignments)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
            rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms, rest_bones_t


def update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness):
    """
    Update the bone-vertex weight map W by fixing bone transformations and using a least squares
    solver subject to non-negativity constraint, affinity constraint, and sparseness constraint.

    inputs: bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            sparseness      Maximum number of bones allowed to influence a particular vertex

    return: A |num_verts| x |num_bones| weight map representing the influence of the jth bone on the ith vertex
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    num_bones = bone_transforms.shape[0]

    W = np.empty((num_verts, num_bones))

    for v in range(num_verts):
        # For every vertex, solve a least squares problem
        Rp = np.empty((num_bones, num_poses, 3))
        for bone in range(num_bones):
            Rp[bone] = bone_transforms[bone,:,:3,:].dot(rest_pose[v] - rest_bones_t[bone]) # |num_bones| x |num_poses| x 3
        # R * p + T
        Rp_T = Rp + bone_transforms[:, :, 3, :] # |num_bones| x |num_poses| x 3
        A = Rp_T.transpose((1, 2, 0)).reshape((3 * num_poses, num_bones)) # 3 * |num_poses| x |num_bones|
        b = poses[:, v, :].reshape(3 * num_poses) # 3 * |num_poses| x 1
        # A和b是最小二乘问题的两个变量

        # Bounds ensure non-negativity constraint and kind of affinity constraint
        # 直接调用函数解决有约束的最小二乘问题，并归一化权重
        w = lsq_linear(A, b, bounds=(0, 1), method='bvls').x  # |num_bones| x 1
        w /= np.sum(w) # Ensure that w sums to 1 (affinity constraint)

        # Remove |B| - |K| bone weights with the least "effect"·
        # 只获取前K个Bone weights
        effect = np.linalg.norm((A * w).reshape(num_poses, 3, num_bones), axis=1) # |num_poses| x |num_bones| # 这一行是算误差
        effect = np.sum(effect, axis=0) # |num_bones| x 1
        num_discarded = max(num_bones - sparseness, 0)
        effective = np.argpartition(effect, num_discarded)[num_discarded:] # |sparseness| x 1

        # Run least squares again, but only use the most effective bones
        # 再用最有用的bone再跑一次这个问题
        A_reduced = A[:, effective] # 3 * |num_oses| x |sparseness|
        w_reduced = lsq_linear(A_reduced, b, bounds=(0, 1), method='bvls').x # |sparseness| x 1
        w_reduced /= np.sum(w_reduced) # Ensure that w sums to 1 (affinity constraint)

        w_sparse = np.zeros(num_bones)
        w_sparse[effective] = w_reduced
        w_sparse /= np.sum(w_sparse) # Ensure that w_sparse sums to 1 (affinity constraint)

        W[v] = w_sparse

    return W


def update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose):
    """
    Updates the bone transformations by fixing the bone-vertex weight map and minimizing an
    objective function individually for each pose and each bone.
    
    inputs: W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.
            bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            
    return: |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
    """
    num_bones = W.shape[1]
    num_poses = poses.shape[0]
    num_verts = W.shape[0]
    
    for pose in range(num_poses):
        for bone in range(num_bones):
            # Represents the points in rest pose without this rest bone's translation
            p_corrected = rest_pose - rest_bones_t[bone] # |num_verts| x 3

            # Calculate q_i for all vertices by equation (6)
            constructed = np.empty((num_bones, num_verts, 3)) # |num_bones| x |num_verts| x 3
            for bone2 in range(num_bones):
                # can't use p_corrected before because we want to correct for every bone2 distinctly
                Rp = bone_transforms[bone2,pose,:3,:].dot((rest_pose - rest_bones_t[bone2]).T).T # |num_verts| x 3
                # R * p + T
                constructed[bone2] = Rp + bone_transforms[bone2, pose, 3, :]
            # w * (R * p + T)
            constructed = constructed.transpose((1, 0, 2)) * W[:, :, np.newaxis] # |num_verts| x |num_bones| x 3
            constructed = np.delete(constructed, bone, axis=1) # |num_verts| x |num_bones-1| x 3
            q = poses[pose] - np.sum(constructed, axis=1) # |num_verts| x 3

            # Calculate p_star, q_star, p_bar, and q_bar for all verts by equation (8)
            p_star = np.sum(np.square(W[:, bone, np.newaxis]) * p_corrected, axis=0) # |num_verts| x 3 => 3 x 1
            p_star /= np.sum(np.square(W[:, bone])) # 3 x 1
            
            q_star = np.sum(W[:, bone, np.newaxis] * q, axis=0) # |num_verts| x 3 => 3 x 1
            q_star /= np.sum(np.square(W[:, bone])) # 3 x 1
            p_bar = p_corrected - p_star # |num_verts| x 3
            q_bar = q - W[:, bone, np.newaxis] * q_star # |num_verts| x 3
            
            # Perform SVD by equation (9)
            P = (p_bar * W[:, bone, np.newaxis]).T # 3 x |num_verts|
            Q = q_bar.T # 3 x |num_verts|
            
            U, S, V = np.linalg.svd(np.matmul(P, Q.T))

            # Calculate rotation R and translation t by equation (10)
            R = U.dot(V).T # 3 x 3
            t = q_star - R.dot(p_star) # 3 x 1
            
            bone_transforms[bone, pose, :3, :] = R
            bone_transforms[bone, pose, 3, :] = t
    
    return bone_transforms


def SSDR(poses, rest_pose, num_bones, sparseness=4, max_iterations=20,initial_Bone=None):
    """
    Computes the Smooth Skinning Decomposition with Rigid bones
    
    inputs: poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones       number of bones to create
            sparseness      max number of bones influencing a single vertex
            
    return: An i x j matrix of bone-vertex weights, where i = # vertices and j = # bones
            A length-B list of (length-t lists of bone transformations [R_j | T_j] ), one list for each bone
            A list of bone translations for the bones at rest
    """
    start_time = time.time()
    print("SSDR: start initialize")
    bone_transforms, rest_bones_t = initialize(poses, rest_pose, num_bones,initial_bone=initial_Bone)
    print("SSDR: finish initialize, start reconstruction")
    for iter in range(max_iterations):
        W = update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness)
        bone_transforms = update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose)
        errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
        print("iter:",iter,"Reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))
    errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
    print("Avg reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))

    return W, bone_transforms, rest_bones_t


def SSDR_w_InitBone(poses, rest_pose, num_bones, sparseness=4, max_iterations=20,initial_Bone=None):
    """
    Computes the Smooth Skinning Decomposition with Rigid bones
    
    inputs: poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones       number of bones to create
            sparseness      max number of bones influencing a single vertex
            
    return: An i x j matrix of bone-vertex weights, where i = # vertices and j = # bones
            A length-B list of (length-t lists of bone transformations [R_j | T_j] ), one list for each bone
            A list of bone translations for the bones at rest
    """
    start_time = time.time()
    print("SSDR: start initialize")
    if initial_Bone is not None:
        num_bones = initial_Bone.shape[0]
    bone_transforms, rest_bones_t = initialize_w_Bone(poses, rest_pose, num_bones,initial_Bone)
    print("SSDR: finish initialize, start reconstruction")
    for iter in range(max_iterations):
        W = update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness)
        bone_transforms = update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose)
        errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
        print("iter:",iter,"Reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))
    errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
    print("Avg reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))

    return W, bone_transforms, rest_bones_t



def reconstruct(rest_pose, bone_transforms, rest_bones_t, W):
    """
    Computes the skinned vertex positions on some poses given bone transforms and weights.

    inputs : rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
             bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
             rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
             W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.

    return: |num_poses| x |num_verts| x 3 Vertex positions for all poses: sum{bones} (w * (R @ p + T)) 
    """
    # Points in rest pose without rest bone translations
    p_corrected = rest_pose[np.newaxis, :, :] - rest_bones_t[:, np.newaxis, :] # |num_bones| x |num_verts| x 3
    constructions = np.einsum('bijk,blk->bilj', bone_transforms[:, :, :3, :], p_corrected) # |num_bones| x |num_poses| x |num_verts| x 3
    constructions += bone_transforms[:, :, np.newaxis, 3, :] # |num_bones| x |num_poses| x |num_verts| x 3
    constructions *= (W.T)[:, np.newaxis, :, np.newaxis] # |num_bones| x |num_poses| x |num_verts| x 3
    return np.sum(constructions, axis=0)

def load_mesh_sequence_from_files(mesh_folder):
    # 获取文件夹中所有.obj文件，并按名称排序
    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.obj')])

    num_poses = len(mesh_files) - 1

    rest_pose = None
    poses = []

    for i, mesh_file in enumerate(mesh_files):
        # 构建完整文件路径
        mesh_path = os.path.join(mesh_folder, mesh_file)

        # 使用trimesh加载mesh文件
        mesh = trimesh.load(mesh_path)

        # 获取顶点信息
        points = mesh.vertices

        # 如果是最后一个mesh文件，设为rest_pose
        if i == num_poses:
            rest_pose = points
        else:
            poses.append(points)

    poses = np.array(poses)

    return rest_pose, poses


# Get numpy vertex arrays from selected objects. Rest pose is most recently selected.
# selectionLs = om.MGlobal.getActiveSelectionList()
# num_poses = selectionLs.length() - 1
# rest_pose = np.array(om.MFnMesh(selectionLs.getDagPath(num_poses)).getPoints(om.MSpace.kWorld))[:, :3]
# poses = np.array([om.MFnMesh(selectionLs.getDagPath(i)).getPoints(om.MSpace.kWorld) for i in range(num_poses)])[:, :, :3]


# mesh_folder = "/path/to/your/mesh/folder"
# rest_pose, poses = load_mesh_sequence_from_files(mesh_folder)
# W, bone_transforms, rest_bones_t = SSDR(poses, rest_pose, 2)


# you can also design a pre-define bone to run this code.