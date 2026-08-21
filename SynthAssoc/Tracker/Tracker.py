import os
import time
import math
import copy
import itertools
import numpy as np # numpy backend
import pygmtools as pygm
import matplotlib.pyplot as plt  # for plotting
import scipy.io as sio  # for loading .mat file
import scipy.spatial as spa  # for Delaunay triangulation

from PIL import Image, ImageDraw
from matplotlib.patches import ConnectionPatch # for plotting matching result
import shutil

obj_resize = (256, 256)
n_outlier = 0
img_list = []
kpts_list = []
n_kpts_list = []
perm_list = []


def plot_image_with_graph(img, kpt, index, A=None):
    # 如果 img 不是 PIL.Image 对象，则转换（此处假设 img 为 np.array 格式时转换）
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    # 复制一份图像用于绘制，避免直接修改原图
    drawn_img = img.copy()
    draw = ImageDraw.Draw(drawn_img)
    
    # 绘制关键点：kpt 是一个 2×N 的 numpy 数组
    for x, y in zip(kpt[0], kpt[1]):
        x_int = int(round(x))
        y_int = int(round(y))
        r = 3  # 定义圆点半径
        # 绘制圆点（红色填充）
        draw.ellipse([x_int - r, y_int - r, x_int + r, y_int + r], fill=(255, 0, 0))
    
    # 如果提供邻接矩阵 A，则根据 A 中非零位置绘制关键点间的连线
    if A is not None:
        rows, cols = np.nonzero(A)
        for i, j in zip(rows, cols):
            x1 = int(round(kpt[0, i]))
            y1 = int(round(kpt[1, i]))
            x2 = int(round(kpt[0, j]))
            y2 = int(round(kpt[1, j]))
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=1)
    
    # 定义保存路径并保存绘制后的图像

    save_path = './PointsSample/' + str(index) + '_with_keypoints.jpg'
    drawn_img.save(save_path)

def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.transpose())
    A = np.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

def get_feature(n, points, adj):
    """
    :param n: points # of graph
    :param points: numpy.ndarray, (n, 2)
    :param adj: numpy.ndarray, (n, n)
    :return: edge feat, angle feat
    """
    points_1 = np.tile(points.reshape(n, 1, 2), (1, n, 1))
    points_2 = np.tile(points.reshape(1, n, 2), (n, 1, 1))
    edge_feat = np.sqrt(np.sum((points_1 - points_2) ** 2, axis=2))
    edge_feat = edge_feat / np.max(edge_feat)
    angle_feat = np.arctan((points_1[:, :, 1] - points_2[:, :, 1]) / (points_1[:, :, 0] - points_2[:, :, 0] + 1e-8))
    angle_feat = 2 * angle_feat / math.pi

    return edge_feat, angle_feat


def get_pair_affinity(edge_feat_1, angle_feat_1, edge_feat_2, angle_feat_2, adj1, adj2):
    n1, n2 = edge_feat_1.shape[0], edge_feat_2.shape[0]
    assert n1 == angle_feat_1.shape[0] and n2 == angle_feat_2.shape[0]

    left_adj = np.tile(adj1.reshape(n1, n1, 1, 1), (1, 1, n2, n2))
    right_adj = np.tile(adj2.reshape(1, 1, n2, n2), (n1, n1, 1, 1))
    adj = left_adj * right_adj

    left_edge_feat = np.tile(edge_feat_1.reshape(n1, n1, 1, 1, -1), (1, 1, n2, n2, 1))
    right_edge_feat = np.tile(edge_feat_2.reshape(1, 1, n2, n2, -1), (n1, n1, 1, 1, 1))
    edge_weight = np.sqrt(np.sum((left_edge_feat - right_edge_feat) ** 2, axis=-1))

    left_angle_feat = np.tile(angle_feat_1.reshape(n1, n1, 1, 1, -1), (1, 1, n2, n2, 1))
    right_angle_feat = np.tile(angle_feat_2.reshape(1, 1, n2, n2, -1), (n1, n1, 1, 1, 1))
    angle_weight = np.sqrt(np.sum((left_angle_feat - right_angle_feat) ** 2, axis=-1))

    affinity = edge_weight * 0.9 + angle_weight * 0.1
    affinity = np.exp(-affinity / 0.1) * adj
    affinity = affinity.transpose(0, 2, 1, 3)

    return affinity


def generate_affinity_matrix(n_points, points_list, adj_list):
    m = len(n_points)
    n_max = max(n_points)
    affinity = np.zeros((m, m, n_max, n_max, n_max, n_max))

    edge_feat_list = []
    angle_feat_list = []
    for n, points, adj in zip(n_points, points_list, adj_list):
        edge_feat, angle_feat = get_feature(n, points, adj)
        edge_feat_list.append(edge_feat)
        angle_feat_list.append(angle_feat)

    for i, j in itertools.product(range(m), range(m)):
        pair_affinity = get_pair_affinity(edge_feat_list[i],
                                          angle_feat_list[i],
                                          edge_feat_list[j],
                                          angle_feat_list[j],
                                          adj_list[i],
                                          adj_list[j])
        affinity[i, j] = pair_affinity

    affinity = affinity.transpose(0, 1, 3, 2, 5, 4).reshape(m, m, n_max * n_max, n_max * n_max)
    return affinity

def cal_accuracy(mat, gt_mat, n):
    m = mat.shape[0]
    acc = 0
    for i in range(m):
        for j in range(m):
            _mat, _gt_mat = mat[i, j], gt_mat[i, j]
            row_sum = np.sum(_gt_mat, axis=0)
            col_sum = np.sum(_gt_mat, axis=1)
            row_idx = [k for k in range(n) if row_sum[k] != 0]
            col_idx = [k for k in range(n) if col_sum[k] != 0]
            _mat = _mat[row_idx, :]
            _mat = _mat[:, col_idx]
            _gt_mat = _gt_mat[row_idx, :]
            _gt_mat = _gt_mat[:, col_idx]
            acc += 1 - np.sum(np.abs(_mat - _gt_mat)) / 2 / (n - n_outlier)
    return acc / (m * m)


def cal_consistency(mat, gt_mat, m, n):
    return np.mean(get_batch_pc_opt(mat))


def cal_affinity(X, X_gt, K, m, n):
    X_batch = X.reshape(-1, n, n)
    X_gt_batch = X_gt.reshape(-1, n, n)
    K_batch = K.reshape(-1, n * n, n * n)
    affinity = get_batch_affinity(X_batch, K_batch)
    affinity_gt = get_batch_affinity(X_gt_batch, K_batch)
    return np.mean(affinity / (affinity_gt + 1e-8))


def get_batch_affinity(X, K, norm=1):
    """
    calculate affinity score
    :param X: (b, n, n)
    :param K: (b, n*n, n*n)
    :param norm: normalization term
    :return: affinity_score (b, 1, 1)
    """
    b, n, _ = X.shape
    vx = X.transpose(0, 2, 1).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.transpose(0, 2, 1)  # (b, 1, n*n)
    affinity = np.matmul(np.matmul(vxt, K), vx) / norm
    return affinity


def get_single_affinity(X, K, norm=1):
    """
    calculate affinity score
    :param X: (n, n)
    :param K: (n*n, n*n)
    :param norm: normalization term
    :return: affinity_score scale
    """
    n, _ = X.shape
    vx = X.transpose(0, 1).reshape(-1, 1)
    vxt = vx.transpose(0, 1)
    affinity = np.matmul(np.matmul(vxt, K), vx) / norm
    return affinity


def get_single_pc(X, i, j, Xij=None):
    """
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :param Xij: (n, n) matching
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.shape
    if Xij is None:
        Xij = X[i, j]
    pair_con = 0
    for k in range(m):
        X_combo = np.matmul(X[i, k], X[k, j])
        pair_con += np.sum(np.abs(Xij - X_combo)) / (2 * n)
    return 1 - pair_con / m


def get_single_pc_opt(X, i, j, Xij=None):
    """
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.shape
    if Xij is None:
        Xij = X[i, j]
    X1 = X[i, :].reshape(-1, n, n)
    X2 = X[:, j].reshape(-1, n, n)
    X_combo = np.matmul(X1, X2)
    pair_con = 1 - np.sum(np.abs(Xij - X_combo)) / (2 * n * m)
    return pair_con


def get_batch_pc(X):
    """
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    pair_con = np.zeros(m, m)
    for i in range(m):
        for j in range(m):
            pair_con[i, j] = get_single_pc_opt(X, i, j)
    return pair_con


def get_batch_pc_opt(X):
    """
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m, _, n, _ = X.shape
    X1 = np.tile(X.reshape(m, 1, m, n, n), (1, m, 1, 1, 1)).reshape(-1, n, n)  # X1[i, j, k] = X[i, k]
    X2 = np.tile(X.reshape(1, m, m, n, n), (m, 1, 1, 1, 1)).transpose(0, 2, 1, 3, 4).reshape(-1, n, n)  # X2[i, j, k] = X[k, j]
    X_combo = np.matmul(X1, X2).reshape(m, m, m, n, n)
    X_ori = np.tile(X.reshape(m, m, 1, n, n), (1, 1, m, 1, 1))
    pair_con = 1 - np.sum(np.abs(X_combo - X_ori), axis=(2, 3, 4)) / (2 * n * m)
    return pair_con


def eval(mat, gt_mat, affinity, m, n):
    acc = cal_accuracy(mat, gt_mat, n)
    src = cal_affinity(mat, gt_mat, affinity, m, n)
    con = cal_consistency(mat, gt_mat, m, n)
    return acc, src, con

def PairMatch(indexi, indexj, rrwm_mat, output_dir):
    a = indexi
    b = indexj

    # 获取图像和对应的关键点（假设 img_list 中的图像均为 PIL.Image 对象，
    # kpts_list 中存储的关键点为形状 (2, n) 的 numpy 数组）
    img_a = img_list[a]
    img_b = img_list[b]

    # 获取图像尺寸
    width_a, height_a = img_a.size
    width_b, height_b = img_b.size

    # 创建一张新的图像，将两幅图横向拼接
    combined_width = width_a + width_b
    combined_height = max(height_a, height_b)
    combined_img = Image.new("RGB", (combined_width, combined_height))

    # 将图像分别粘贴到拼接图的左右位置
    combined_img.paste(img_a, (0, 0))
    combined_img.paste(img_b, (width_a, 0))

    # 创建一个 ImageDraw 对象用于绘制连接线和关键点标记
    draw = ImageDraw.Draw(combined_img)

    # 从匹配矩阵中提取第 a 张图与第 b 张图之间的匹配结果 X，X 的形状为 (n, n)
    X = rrwm_mat[a, b]

    # 对于图 a 中的每个关键点
    for i in range(X.shape[0]):
        # 找到匹配得分最大的索引 j，即图 a 的第 i 个关键点匹配到图 b 中的第 j 个关键点
        j = np.argmax(X[i]).item()
        # 从关键点数组中获取对应的坐标（x, y）
        x_a, y_a = kpts_list[a][0, i], kpts_list[a][1, i]
        x_b, y_b = kpts_list[b][0, j], kpts_list[b][1, j]
        # 由于图 b 在拼接图中位于右侧，因此 x 坐标需要加上图 a 的宽度
        x_b += width_a

        # 根据匹配情况选择颜色（如果 i 和 j 相同认为匹配正确，使用绿色，否则使用红色）
        color = "green" if i == j else "red"

        # 在拼接图上绘制一条连接两个关键点的直线
        draw.line([(x_a, y_a), (x_b, y_b)], fill=color, width=2)
        # 同时在两个关键点处绘制一个小圆点，便于观察
        r = 3  # 圆点半径
        draw.ellipse([x_a - r, y_a - r, x_a + r, y_a + r], fill=color)
        draw.ellipse([x_b - r, y_b - r, x_b + r, y_b + r], fill=color)

    # 保存最终拼接并标记匹配结果的图像
    save_path = output_dir + str(indexi) + '_' + str(indexj)  + '_with_match.jpg'
    combined_img.save(save_path)

def Vis(output_dir, n_images, rrwm_mat):
    if os.path.exists(output_dir):
        print(f"文件夹 '{output_dir}' 已存在，正在删除...")
        shutil.rmtree(output_dir)  # 删除整个文件夹及其内部所有文件和子文件夹
    os.makedirs(output_dir)
    print(f"文件夹 '{output_dir}' 已成功创建。")
    for i in range(n_images):
        for j in range(i+1, n_images):
            # print(str(i)+'_'+str(j))
            PairMatch(i, j, rrwm_mat, output_dir)


def _get_batch_pc_opt(X):
    """
    CAO/Floyd-fast helper function (compute consistency in batch)
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m = X.shape[0]
    n = X.shape[2]
    X1 = X.reshape(m, 1, m, n, n)
    X1 = np.tile(X1,(1, m, 1, 1, 1)).reshape(-1, n, n)  # X1[i, j, k] = X[i, k]
    X2 = X.reshape(1, m, m, n, n)
    X2 = np.tile(X2,(m, 1, 1, 1, 1)).swapaxes(1,2).reshape(-1, n, n)  # X2[i, j, k] = X[k, j]
    X_combo = np.matmul(X1, X2).reshape(m, m, m, n, n)
    X_ori = X.reshape(m, m, 1, n, n)
    X_ori = np.tile(X_ori,(1, 1, m, 1, 1))
    pair_con = 1 - np.sum(np.abs(X_combo - X_ori), axis=(2, 3, 4)) / (2 * n * m)
    return pair_con

def compute_affinity_score(X, K):
    """
    Numpy implementation of computing affinity score
    """
    b, n, _ = X.shape
    vx = X.swapaxes(1,2).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.swapaxes(1, 2)  # (b, 1, n*n)
    affinity = np.squeeze(np.squeeze(np.matmul(np.matmul(vxt, K), vx),axis=-1),axis=-1)
    return affinity

def mgm_floyd_fast_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node
    #K是相似性矩阵 #X是两张图之间的匹配矩阵
    def _comp_aff_score(X, K):
            b, n, _ = X.shape
            vx = X.swapaxes(1,2).reshape(b, -1, 1)  # (b, n*n, 1)
            vxt = vx.swapaxes(1, 2)  # (b, 1, n*n)
            conf = np.matmul(np.matmul(vxt, K), vx)
            affinity = np.squeeze(np.squeeze(conf, axis=-1),axis=-1)
            tempA = np.expand_dims(affinity,axis=-1)
            return np.expand_dims(tempA,axis=-1)

    mask1 = np.arange(m).reshape(m, 1).repeat(m,axis=1)
    mask2 = np.arange(m).reshape(1, m).repeat(m,axis=0)
    mask = (mask1 < mask2).astype(float)
    X_mask = mask.reshape(m, m, 1, 1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n*n, n*n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, torch.mean(pair_aff).item(), torch.mean(get_batch_pc_opt(X)).item()
        # ))

        X1 = X[:, k].reshape(m, 1, n, n)
        X1 = np.tile(X1,(1, m, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n)
        X2 = np.tile(X2,(m, 1, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = np.matmul(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        score_ori = aff_ori
        score_combo = aff_combo

        upt = (score_ori < score_combo).astype(float)
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.swapaxes(0,1).swapaxes(2, 3) * (1 - X_mask)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)

        X1 = X[:, k].reshape(m, 1, n, n)
        X1 = np.tile(X1,(1, m, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n)
        X2 = np.tile(X2,(m, 1, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = np.matmul(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        con_ori = np.sqrt(pair_con)
        con1 = pair_con[:, k].reshape(m, 1).repeat(m,axis=1)
        con2 = pair_con[k, :].reshape(1, m).repeat(m,axis=0)
        con_combo = np.sqrt(con1 * con2)

        score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        upt = (score_ori < score_combo).astype(float)
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.swapaxes(0,1).swapaxes(2, 3) * (1 - X_mask)
    return X


if __name__ == '__main__':
    
    bm = pygm.benchmark.Benchmark(name='WillowObject', 
                              sets='train', 
                              obj_resize=obj_resize,
                              problem= 'MGM')
    
    ids = ['060_0000', '060_0001', '060_0002', '060_0005', ]
    data_list, perm_mat_dict, ids_ret = bm.get_data(ids, test=False, shuffle=False)
    for data in data_list:
        img = Image.fromarray(data['img'])
        coords = sorted(data['kpts'], key=lambda x: x['labels'])
        kpts = np.array([[kpt['x'] for kpt in coords], 
                        [kpt['y'] for kpt in coords]])
        perm = np.eye(kpts.shape[1])
        img_list.append(img)
        kpts_list.append(kpts)
        n_kpts_list.append(kpts.shape[1])
        perm_list.append(perm)
    
    adj_list = []

    n_images = len(kpts_list)

    for i in range(n_images):
        A = delaunay_triangulation(kpts_list[i])
        adj_list.append(A)
    
    output_dir ='./PointsSample/'
    if os.path.exists(output_dir):
        print(f"文件夹 '{output_dir}' 已存在，正在删除...")
        shutil.rmtree(output_dir)  # 删除整个文件夹及其内部所有文件和子文件夹
    os.makedirs(output_dir)
    print(f"文件夹 '{output_dir}' 已成功创建。")
    for i in range(n_images):
        plot_image_with_graph(img_list[i], kpts_list[i], i, adj_list[i])

    #这里是亲和矩阵
    affinity_mat = generate_affinity_matrix(n_kpts_list, kpts_list, adj_list)

    m = len(kpts_list)
    n = int(np.max(np.array(n_kpts_list)))
    ns_src = np.ones(m * m, dtype=int) * n
    ns_tgt = np.ones(m * m, dtype=int) * n

    gt_mat = np.zeros((m, m, n, n))
    for i in range(m):
        for j in range(m):
            gt_mat[i, j] = np.matmul(perm_list[i].transpose(0, 1), perm_list[j])

    tic = time.time()
    # 两两匹配结果->构成超图
    rrwm_mat_0 = pygm.classic_solvers.rrwm(affinity_mat.reshape(-1, n * n, n * n), ns_src, ns_tgt)
    rrwm_mat = pygm.linear_solvers.hungarian(rrwm_mat_0)
    toc = time.time()
    rrwm_mat = rrwm_mat.reshape(m, m, n, n)
    rrwm_acc, rrwm_src, rrwm_con = eval(rrwm_mat, gt_mat, affinity_mat, m, n)
    rrwm_tim = toc - tic
    print(f"RRWM匹配结果：准确率 = {rrwm_acc:.4f}, 亲和度得分 = {rrwm_src:.4f}, 一致性 = {rrwm_con:.4f}, 耗时 = {rrwm_tim:.4f}秒")
    Vis(output_dir='./RRWM/', n_images = n_images, rrwm_mat = rrwm_mat)


    fn = mgm_floyd_fast_solver(affinity_mat, rrwm_mat, num_graph = 4, num_node = 10, param_lambda = 0.4)
    print(fn.shape)

    # import inspect
    # import pygmtools.numpy_backend as nb
    # print(inspect.getsource(nb.mgm_floyd_fast_solver))

    # mod = importlib.import_module(f'pygmtools.{backend}_backend')
    # fn = mod.compute_affinity_score
    # print(fn)
    


