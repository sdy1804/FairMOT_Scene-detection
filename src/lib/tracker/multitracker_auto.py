from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from models import *
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker import matching
from lib.tracking_utils.kalman_filter_auto import KalmanFilter
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import *
from lib.utils.post_process import ctdet_post_process
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param buff_size:
        """

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  
        self.alpha = 0.9

    def update_features(self, feat):
        # print("feat.shape : ", feat.shape)
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        # print("self.curr_feat :", self.curr_feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat
            # print("SMOOTH_FEAT NONE")
        else:
            self.smooth_feat = self.alpha * \
                               self.smooth_feat + (1 - self.alpha) * feat
            # print("self.smooth_feat :", self.smooth_feat)
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, kalman_filter=None, std_weight_position=None, std_weight_velocity=None):
        if len(stracks) > 0:
            if kalman_filter is None:
                kalman_filter = KalmanFilter(std_weight_position=std_weight_position, std_weight_velocity=std_weight_velocity)
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            # multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                # multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
            # print("std_weight_position :",std_weight_position)
            # print("std_weight_velocity :",std_weight_velocity)

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30, predicted_class=None):
    # def __init__(self, opt, frame_rate=30, std_weight_position=None, std_weight_velocity=None):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model_FairMOT = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model_Scene = create_model(opt.arch, opt.heads, opt.head_conv)
        
        self.model_FairMOT = load_model(self.model_FairMOT, opt.load_model_FairMOT)
        self.model_Scene = load_model(self.model_Scene, opt.load_model_Scene)
        
        self.model_FairMOT = self.model_FairMOT.to(opt.device)
        self.model_Scene = self.model_Scene.to(opt.device)
        
        self.model_FairMOT.eval()
        self.model_Scene.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter(predicted_class=predicted_class)
        
        
    def post_process(self, dets, meta):
        """
        2D bbox检测结果后处理
        :param dets:
        :param meta:
        :return:
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6

        # 仿射变换到输出分辨率的坐标系
        dets = ctdet_post_process(dets.copy(),
                                  [meta['c']], [meta['s']],
                                  meta['out_height'],
                                  meta['out_width'],
                                  self.opt.num_classes)

        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        
        return dets[0]

    def merge_outputs(self, detections):
        """
        :param detections:
        :return:
        """
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections],
                                        axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        
        return results 
    
    def update(self, im_blob, img0):
        self.frame_id += 1
                
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c,
                's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():  
            output_FairMOT = self.model_FairMOT.forward(im_blob)[-1]
            output_Scene = self.model_Scene.forward(im_blob)[-1]
            output_Scene = output_Scene['scene']
            predicted_class = output_Scene.argmax(dim=-1, keepdim=True)
            self.kalman_filter = KalmanFilter(predicted_class)

            hm = output_FairMOT['hm'].sigmoid_()
            # print("hm shape ", hm.shape, "hm:\n", hm)

            wh = output_FairMOT['wh']
            # print("wh shape ", wh.shape, "wh:\n", wh)

            id_feature = output_FairMOT['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output_FairMOT['reg'] if self.opt.reg_offset else None
            # print("reg shape ", reg.shape, "reg:\n", reg)

            #  检测和分类结果解析
            dets, inds = mot_decode(heat=hm,  # heatmap
                                    wh=wh,
                                    reg=reg,
                                    cat_spec_wh=self.opt.cat_spec_wh,
                                    K=self.opt.K)
            # print(dets.size())
            # 组织用于Re-IDd的特征向量
            id_feature = _tranpose_and_gather_feat(id_feature, inds) # 1, 128, 128
            # print(id_feature.size())
            id_feature = id_feature.squeeze(0)  # K × FeatDim -> 128, 128
            # print(id_feature.size())
            id_feature = id_feature.cpu().numpy()

        # 检测结果后处理
        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        # 过滤掉score得分太低的dets
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds] # detected or tracked object num x 128
        # print(id_feature.shape) 

        # vis可视化bbox
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, 
                          (bbox[0], bbox[1]),  # left-top point
                          (bbox[2], bbox[3]),  # right-down point
                          (0, 255, 0), 
                          2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''
        
        if len(dets) > 0:
            '''Detections, tlbrs: top left bottom right score'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30)
                          for (tlbrs, feat) in zip(dets[:, :5], id_feature)]
            # for (tlbrs, feat) in zip(dets[:, :5], id_feature):
                # print("feat:",feat)
            # for (tlbrs, feat) in zip(dets[:, :5], id_feature):
            #     print("STrack :",STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30))
            # print("detections : ", detections)
            
        else:
            detections = []
        
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        # print("tracked_stracks :", len(self.tracked_stracks))
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # print('strack_pool : ',strack_pool)
        # print(type(strack_pool))
        # Predict the current location with KF
        # for strack in strack_pool:
        # strack.predict()
        # STrack.multi_predict(strack_pool)
        
        # STrack.multi_predict(strack_pool, kalman_filter=KalmanFilter(std_weight_position=scene_info[0, 1],
        #                                       std_weight_velocity=scene_info[0, 0]))
        STrack.multi_predict(strack_pool, self.kalman_filter)
        # print('strack_pool : ',strack_pool)
        # print('detections : ',detections)
        dists = matching.embedding_distance(strack_pool, detections)
        # print("dists.shape : ",dists.shape)
        # print("dists : ",dists)
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # print(dists.shape)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for i_tracked, i_det in matches:
            track = strack_pool[i_tracked]
            det = detections[i_det]
            if track.state == TrackState.Tracked:
                track.update(detections[i_det], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i]
                             for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections, predicted_class)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for i_tracked, i_det in matches:
            track = r_tracked_stracks[i_tracked]
            det = detections[i_det]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections, predicted_class)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for i_tracked, i_det in matches:
            unconfirmed[i_tracked].update(detections[i_det], self.frame_id)
            activated_starcks.append(unconfirmed[i_tracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for i_new in u_detection:
            track = detections[i_new]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks,
                                                                           self.lost_stracks, predicted_class)
        
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format(
            [track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format(
            [track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format(
            [track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format(
            [track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(t_list_a, t_list_b):
    """
    join two track lists
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, predicted_class):
    pdist = matching.iou_distance(stracksa, stracksb, predicted_class)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]

    return resa, resb
