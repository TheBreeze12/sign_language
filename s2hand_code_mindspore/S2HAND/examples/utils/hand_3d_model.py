import os
import pickle
import numpy as np

import mindspore as ms
import mindspore.ops as ops


bases_num = 10
pose_num = 30
mesh_num = 778
keypoints_num = 16

_MANO_CACHE = None


def _to_tensor(x, dtype=ms.float32):
    return ms.Tensor(x, dtype=dtype)


def _load_mano_data():
    global _MANO_CACHE
    if _MANO_CACHE is not None:
        return _MANO_CACHE

    mano_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "MANO_RIGHT.pkl")
    dd = pickle.load(open(mano_file, "rb"), encoding="latin1")

    kintree_table = dd["kintree_table"]
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    mano = {
        "kintree_table": kintree_table,
        "parent": parent,
        "v_template": dd["v_template"].astype(np.float32),
        "shapedirs": dd["shapedirs"].astype(np.float32),
        "posedirs": dd["posedirs"].astype(np.float32),
        "J_regressor": dd["J_regressor"].todense().astype(np.float32),
        "weights": dd["weights"].astype(np.float32),
        "hands_components": np.vstack(dd["hands_components"][:pose_num]).astype(np.float32),
        "hands_mean": dd["hands_mean"].astype(np.float32),
        "f": dd["f"].astype(np.int32),
    }
    _MANO_CACHE = mano
    return mano


def rodrigues(r):
    eps = 1e-8
    theta = ops.sqrt(ops.sum(r * r, axis=1) + eps)

    def skew(n_):
        ns = ops.split(n_, axis=1, output_num=3)
        z = ops.zeros_like(ns[0])
        sn = ops.concat((z, -ns[2], ns[1], ns[2], z, -ns[0], -ns[1], ns[0], z), axis=1)
        return ops.reshape(sn, (-1, 3, 3))

    n = r / ops.expand_dims(theta, 1)
    sn = skew(n)

    b = r.shape[0]
    i3 = ops.tile(ops.expand_dims(ops.eye(3, 3, ms.float32), 0), (b, 1, 1))

    sin_t = ops.reshape(ops.sin(theta), (-1, 1, 1))
    cos_t = ops.reshape(ops.cos(theta), (-1, 1, 1))
    r1 = i3 + sin_t * sn + (1.0 - cos_t) * ops.matmul(sn, sn)

    sr = skew(r)
    theta2 = theta * theta
    t2 = ops.reshape(theta2, (-1, 1, 1))
    r2 = i3 + (1.0 - t2 / 6.0) * sr + (0.5 - t2 / 24.0) * ops.matmul(sr, sr)

    mask = ops.reshape(theta < 1e-30, (-1, 1, 1))
    r_out = ops.where(mask, r2, r1)
    return r_out, sn


def get_poseweights(poses, bsize):
    pose_matrix, _ = rodrigues(ops.reshape(poses[:, 1:, :], (-1, 3)))
    eye = np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0), bsize * (keypoints_num - 1), axis=0)
    pose_matrix = pose_matrix - _to_tensor(eye)
    pose_matrix = ops.reshape(pose_matrix, (bsize, -1))
    return pose_matrix


def rot_pose_beta_to_mesh(rots, poses, betas):
    mano = _load_mano_data()

    kintree_table = mano["kintree_table"]
    parent = mano["parent"]

    batch_size = rots.shape[0]

    mesh_mu = _to_tensor(np.expand_dims(mano["v_template"], 0))
    mesh_pca = _to_tensor(np.expand_dims(mano["shapedirs"], 0))
    posedirs = _to_tensor(np.expand_dims(mano["posedirs"], 0))
    j_regressor = _to_tensor(np.expand_dims(mano["J_regressor"], 0))
    weights = _to_tensor(np.expand_dims(mano["weights"], 0))
    hands_components = _to_tensor(np.expand_dims(mano["hands_components"], 0))
    hands_mean = _to_tensor(np.expand_dims(mano["hands_mean"], 0))
    root_rot = _to_tensor(np.array([[np.pi, 0.0, 0.0]], dtype=np.float32))

    mesh_face = _to_tensor(np.expand_dims(mano["f"], 0), dtype=ms.int32)
    mesh_face = ops.tile(mesh_face, (batch_size, 1, 1))

    poses = ops.reshape(hands_mean + ops.squeeze(ops.matmul(ops.expand_dims(poses, 1), hands_components), axis=1),
                        (batch_size, keypoints_num - 1, 3))
    poses = ops.concat((ops.reshape(ops.tile(root_rot, (batch_size, 1)), (batch_size, 1, 3)), poses), axis=1)

    shapedirs = ops.reshape(
        ops.transpose(ops.tile(mesh_pca, (batch_size, 1, 1, 1)), (0, 3, 1, 2)),
        (batch_size, bases_num, -1),
    )
    v_shaped = ops.reshape(
        ops.squeeze(ops.matmul(ops.expand_dims(betas, 1), shapedirs), axis=1)
        + ops.reshape(ops.tile(mesh_mu, (batch_size, 1, 1)), (batch_size, -1)),
        (batch_size, mesh_num, 3),
    )

    pose_weights = get_poseweights(poses, batch_size)
    v_posed = v_shaped + ops.squeeze(
        ops.matmul(
            ops.tile(posedirs, (batch_size, 1, 1, 1)),
            ops.tile(ops.reshape(pose_weights, (batch_size, 1, (keypoints_num - 1) * 9, 1)), (1, mesh_num, 1, 1)),
        ),
        axis=3,
    )

    j_posed = ops.matmul(
        ops.transpose(v_shaped, (0, 2, 1)),
        ops.transpose(ops.tile(j_regressor, (batch_size, 1, 1)), (0, 2, 1)),
    )
    j_posed = ops.transpose(j_posed, (0, 2, 1))
    j_posed_split = [ops.reshape(sp, (batch_size, 3)) for sp in ops.split(ops.transpose(j_posed, (1, 0, 2)), axis=0, output_num=j_posed.shape[1])]

    pose_split = ops.split(ops.transpose(poses, (1, 0, 2)), axis=0, output_num=poses.shape[1])

    angle_matrix = []
    for i in range(keypoints_num):
        out, _ = rodrigues(ops.reshape(pose_split[i], (-1, 3)))
        angle_matrix.append(out)

    homo_tail = _to_tensor(np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32))

    def with_zeros(x):
        return ops.concat((x, ops.tile(homo_tail, (batch_size, 1, 1))), axis=1)

    def pack(x):
        return ops.concat((ops.zeros((batch_size, 4, 3), ms.float32), x), axis=2)

    results = {}
    results[0] = with_zeros(ops.concat((angle_matrix[0], ops.reshape(j_posed_split[0], (batch_size, 3, 1))), axis=2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(
            ops.concat(
                (angle_matrix[i], ops.reshape(j_posed_split[i] - j_posed_split[parent[i]], (batch_size, 3, 1))),
                axis=2,
            )
        )
        results[i] = ops.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []
    for i in range(len(results)):
        vec = ops.reshape(ops.concat((j_posed_split[i], ops.zeros((batch_size, 1), ms.float32)), axis=1), (batch_size, 4, 1))
        results2.append(ops.expand_dims(results[i] - pack(ops.matmul(results[i], vec)), 0))

    results_cat = ops.concat(results2, axis=0)

    t = ops.matmul(
        ops.transpose(results_cat, (1, 2, 3, 0)),
        ops.tile(ops.expand_dims(ops.transpose(ops.tile(weights, (batch_size, 1, 1)), (0, 2, 1)), 1), (1, 4, 1, 1)),
    )
    ts = ops.split(t, axis=2, output_num=4)

    rest_shape_h = ops.concat((v_posed, ops.ones((batch_size, mesh_num, 1), ms.float32)), axis=2)
    rest_shape_hs = ops.split(rest_shape_h, axis=2, output_num=4)

    v = (
        ops.reshape(ts[0], (batch_size, 4, mesh_num)) * ops.reshape(rest_shape_hs[0], (-1, 1, mesh_num))
        + ops.reshape(ts[1], (batch_size, 4, mesh_num)) * ops.reshape(rest_shape_hs[1], (-1, 1, mesh_num))
        + ops.reshape(ts[2], (batch_size, 4, mesh_num)) * ops.reshape(rest_shape_hs[2], (-1, 1, mesh_num))
        + ops.reshape(ts[3], (batch_size, 4, mesh_num)) * ops.reshape(rest_shape_hs[3], (-1, 1, mesh_num))
    )

    rots_m = rodrigues(rots)[0]

    jtr = []
    for j_id in range(len(results_global)):
        jtr.append(results_global[j_id][:, :3, 3:4])

    jtr.insert(4, ops.expand_dims(v[:, :3, 320], 2))
    jtr.insert(8, ops.expand_dims(v[:, :3, 443], 2))
    jtr.insert(12, ops.expand_dims(v[:, :3, 672], 2))
    jtr.insert(16, ops.expand_dims(v[:, :3, 555], 2))
    jtr.insert(20, ops.expand_dims(v[:, :3, 744], 2))

    jtr = ops.concat(tuple(jtr), axis=2)

    v = ops.transpose(ops.matmul(rots_m, v[:, :3, :]), (0, 2, 1))
    jtr = ops.transpose(ops.matmul(rots_m, jtr), (0, 2, 1))

    return ops.concat((jtr, v), axis=1), mesh_face, poses
