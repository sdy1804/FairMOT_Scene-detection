import lap
import numpy as np
import scipy
import torch
import math
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from lib.tracking_utils import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    optimize_threshold = 0
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), \
               tuple(range(cost_matrix.shape[0])), \
               tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # print('x :', x.shape)
    # print('y :', y.shape)
    for ix, mx in enumerate(x):
        # print('mx :',mx)
        # if mx >= 0:
        if mx >= optimize_threshold:
            matches.append([ix, mx])
            # print(matches)

    # unmatched_a = np.where(x < 0)[0]
    unmatched_a = np.where(x < optimize_threshold)[0]
    # unmatched_b = np.where(y < 0)[0]
    unmatched_b = np.where(y < optimize_threshold)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs, predicted_class):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    # ious = bbox_ious(
    #     np.ascontiguousarray(atlbrs, dtype=np.float),
    #     np.ascontiguousarray(btlbrs, dtype=np.float)
    # )
    # ious = my_ciou(
    #     np.ascontiguousarray(atlbrs, dtype=np.float),
    #     np.ascontiguousarray(btlbrs, dtype=np.float)
    # )
    ious = my_ciou_det(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
        predicted_class
    )
    # ious = my_bbox_ious(
    #     np.ascontiguousarray(atlbrs, dtype=np.float),
    #     np.ascontiguousarray(btlbrs, dtype=np.float)
    # )
    # print("atlbrs :", atlbrs)
    # print("atlbrs.shape :",np.ascontiguousarray(atlbrs).shape)
    # print("btlbrs.shape :",np.ascontiguousarray(btlbrs).shape)
    # print("ious.shape :",ious.shape)
    return ious

def my_bbox_ious(boxes, query_boxes):
    
    n = boxes.shape[0]
    k = query_boxes.shape[0]

    overlaps = np.zeros((n, k))

    for K in range(k):
        query_box_w = query_boxes[K, 2] - query_boxes[K, 0] + 1
        query_box_h = query_boxes[K, 3] - query_boxes[K, 1] + 1

        query_box_area = query_box_h * query_box_w
        
        for N in range(n):
            box_w = boxes[N, 2] - boxes[N, 0] + 1
            box_h = boxes[N, 3] - boxes[N, 1] + 1

            box_area = box_h * box_w

            inter_w = min(boxes[N, 2], query_boxes[K, 2]) - \
                     max(boxes[N, 0], query_boxes[K, 0]) + 1
            if inter_w > 0:
                inter_h = min(boxes[N, 3], query_boxes[K, 3] - \
                         max(boxes[N, 1], query_boxes[K, 1])) + 1
                if inter_h > 0:
                    total_area = box_area + query_box_area - inter_w * inter_h

                    overlaps[N, K] = inter_w * inter_h / total_area
                    print("overlaps : ", overlaps)
    return overlaps

def my_ciou(boxes, query_boxes):

    n = boxes.shape[0]
    k = query_boxes.shape[0]

    overlaps = np.zeros((n, k))

    for K in range(k):
        query_box_w = query_boxes[K, 2] - query_boxes[K, 0] + 1
        query_box_h = query_boxes[K, 3] - query_boxes[K, 1] + 1

        query_box_area = query_box_h * query_box_w

        query_center_x = (query_boxes[K, 0] + query_boxes[K, 2]) / 2
        query_center_y = (query_boxes[K, 1] + query_boxes[K, 3]) / 2
        
        for N in range(n):
            box_w = boxes[N, 2] - boxes[N, 0] + 1
            box_h = boxes[N, 3] - boxes[N, 1] + 1

            box_area = box_h * box_w

            box_center_x = (boxes[N, 0] + boxes[N, 2]) / 2
            box_center_y = (boxes[N, 1] + boxes[N, 3]) / 2

            inter_l = max(box_center_x - box_w / 2, query_center_x - query_box_w / 2)
            inter_r = min(box_center_x + box_w / 2, query_center_x + query_box_w / 2)
            inter_t = max(box_center_y - box_h / 2, query_center_y - query_box_h / 2)
            inter_b = min(box_center_y + box_h / 2, query_center_y + query_box_h / 2)
            inter_area = np.clip((inter_r - inter_l), 0, None) * np.clip((inter_b - inter_t), 0, None)

            c_l = min(box_center_x - box_w / 2, query_center_x - query_box_w / 2)
            c_r = max(box_center_x + box_w / 2, query_center_x + query_box_w / 2)
            c_t = min(box_center_y - box_h / 2, query_center_y - query_box_h / 2)
            c_b = max(box_center_y + box_h / 2, query_center_y + query_box_h / 2)

            inter_diag = (query_center_x - box_center_x)**2 + (query_center_y - box_center_y)**2
            c_diag = np.clip((c_r - c_l),0, None)**2 + np.clip((c_b - c_t), 0, None)**2

            union = box_area + query_box_area - inter_area
            u = (inter_diag) / c_diag
            # print("root_inter_diag :", math.sqrt(inter_diag))
            # print("u :", u)
            iou = inter_area / union
            # print("iou : ", iou)
            v = (4 / (math.pi**2)) * pow((math.atan(query_box_w / query_box_h) - math.atan(box_w / box_h)), 2)            
            # print("v : ", v)
            S = (iou > 0.5).astype(np.float32)
            # print("S :", S)
            alpha = S * v / (1 - iou + v)
            # alpha = v / (1 - iou + v)
            # print("alpha : ", alpha)
            IoU_weight = 1.0
            u_weight = 1.0
            alpha_v_weight = 5000.0
            # print("alpha * v : ", alpha * v * alpha_v_weight)
            # overlaps[N, K] = IoU_weight * iou - u_weight * u - alpha_v_weight * alpha * v
            overlaps[N, K] = IoU_weight * iou - (1 - iou)**2 * u - alpha_v_weight * alpha * v
            # overlaps[N, K] = IoU_weight * iou - u_weight * math.sqrt(inter_diag) - alpha_v_weight * alpha * v
            # overlaps[N, K] = iou - u - alpha * v
            # print(my_sigmoid((1 - iou)**2 * my_sigmoid(math.sqrt(inter_diag))))
            overlaps[N, K] = np.clip(overlaps[N, K], 0.0, 1.0)
            print("overlaps : ", overlaps)
    return overlaps

def my_ciou_det(boxes, query_boxes, predicted_class):

    n = boxes.shape[0]
    k = query_boxes.shape[0]

    overlaps = np.zeros((n, k))

    for K in range(k):
        query_box_w = query_boxes[K, 2] - query_boxes[K, 0] + 1
        query_box_h = query_boxes[K, 3] - query_boxes[K, 1] + 1

        query_box_area = query_box_h * query_box_w

        query_center_x = (query_boxes[K, 0] + query_boxes[K, 2]) / 2
        query_center_y = (query_boxes[K, 1] + query_boxes[K, 3]) / 2
        
        for N in range(n):
            box_w = boxes[N, 2] - boxes[N, 0] + 1
            box_h = boxes[N, 3] - boxes[N, 1] + 1

            box_area = box_h * box_w

            box_center_x = (boxes[N, 0] + boxes[N, 2]) / 2
            box_center_y = (boxes[N, 1] + boxes[N, 3]) / 2

            inter_l = max(box_center_x - box_w / 2, query_center_x - query_box_w / 2)
            inter_r = min(box_center_x + box_w / 2, query_center_x + query_box_w / 2)
            inter_t = max(box_center_y - box_h / 2, query_center_y - query_box_h / 2)
            inter_b = min(box_center_y + box_h / 2, query_center_y + query_box_h / 2)
            inter_area = np.clip((inter_r - inter_l), 0, None) * np.clip((inter_b - inter_t), 0, None)

            c_l = min(box_center_x - box_w / 2, query_center_x - query_box_w / 2)
            c_r = max(box_center_x + box_w / 2, query_center_x + query_box_w / 2)
            c_t = min(box_center_y - box_h / 2, query_center_y - query_box_h / 2)
            c_b = max(box_center_y + box_h / 2, query_center_y + query_box_h / 2)

            inter_diag = (query_center_x - box_center_x)**2 + (query_center_y - box_center_y)**2
            c_diag = np.clip((c_r - c_l),0, None)**2 + np.clip((c_b - c_t), 0, None)**2

            union = box_area + query_box_area - inter_area
            u = (inter_diag) / c_diag
            # print("root_inter_diag :", math.sqrt(inter_diag))
            # print("u :", u)
            iou = inter_area / union
            # print("iou : ", iou)
            v = (4 / (math.pi**2)) * pow((math.atan(query_box_w / query_box_h) - math.atan(box_w / box_h)), 2)            
            # print("v : ", v)
            S = (iou > 0.5).astype(np.float32)
            # print("S :", S)
            alpha = S * v / (1 - iou + v)
            # alpha = v / (1 - iou + v)
            # print("alpha : ", alpha)
            IoU_weight = 1.0
            u_weight = 1.0
            alpha_v_weight = 1.0
            if predicted_class == 0: # Swimming
                alpha_v_weight = 1.0
            elif predicted_class == 1: # Turning
                alpha_v_weight = 1.0
            elif predicted_class == 2: # Diving
                alpha_v_weight = 41.0
            elif predicted_class == 3: # Finish
                predicted_class = 1.0
            elif predicted_class == 4: # On-Block
                alpha_v_weight = 41.0
            overlaps[N, K] = IoU_weight * iou - (1 - iou)**2 * u - alpha_v_weight * alpha * v     
            overlaps[N, K] = np.clip(overlaps[N, K], 0.0, 1.0)
            # print("overlaps : ", overlaps)
    return overlaps

def my_softmax(value):

    exp_value = np.exp(value)
    sum_exp_value = np.sum(exp_value)
    y = exp_value / sum_exp_value

    return y

def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou_distance(atracks, btracks, predicted_class):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    # print('atracks :', atracks)
    # print('btracks :', btracks)
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        # print('btlbrs :', btlbrs)
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        # print('btlbrs :', len(btlbrs))
        # print('atlbrs :', len(atlbrs))
    _ious = ious(atlbrs, btlbrs, predicted_class)
    # _ious = ciou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    # print(type(cost_matrix))
    # return np.array(cost_matrix)
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    # print("len(tracks):",len(tracks))
    # print("len(detections) : ", len(detections))
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    # print("cost_matrix.shape :", cost_matrix.shape)
    if cost_matrix.size == 0:
        return cost_matrix
    # for track in detections:
        # print("track.curr_feat :",track.curr_feat)
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # print("det_feature.shape : ", det_features.shape)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # print("track_feature.shape : ", track_features.shape)
    
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    # print("cost_matrix.shape :",cost_matrix.shape)
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean,
                                             track.covariance,
                                             measurements,
                                             only_position,
                                             metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix
