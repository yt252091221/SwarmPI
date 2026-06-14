"""
SwarmPI 身份保持追踪管线 v6 —— Prototype 独立身份验证

相对 v5 的核心改进：
  Prototype 不再混入零水印项参与 Hungarian 匹配，
  而是作为独立的 **身份验证层**：
    1. Hungarian 匹配用 4 个局部特征（空间/HSV/运动/零水印）
    2. 匹配完成后，对每对匹配用 prototype 做全局身份一致性验证
    3. 若 prototype 距离过大（身份不一致），拒绝该匹配
    4. Prototype 更新使用 **匹配置信度加权**：
       高置信度匹配快速吸收，低置信度匹配弱吸收

这样 prototype 承担"守门员"角色，与局部特征互补而非并列。
"""

import os
import numpy as np
import cv2
import pywt
from collections import deque
from enum import Enum, auto
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ══════════════════════════════════════════════════════════════════
# 超参数
# ══════════════════════════════════════════════════════════════════

TAU_IOU   = 0.2
TAU_DEPTH = 0.10

# A/B/C/D = hist, motion, spatial, watermark.
COMPONENT_KEYS = ["hist", "motion", "spatial", "watermark"]
OMEGA_BASE = np.array([0.20, 0.25, 0.15, 0.40], dtype=np.float32)
EMA_ALPHA = 0.7

# 零水印参数
TERNARY_DELTA       = 0.15
TEMPORAL_WINDOW      = 8
STABILITY_THRESHOLD  = 0.75
IDENTITY_BIT_WEIGHT  = 2.0
ENTROPY_TOP_K        = 2

# Prototype 参数
PROTOTYPE_SUPPORT_SIZE = 15
PROTOTYPE_DIM          = 32

# 【v6】Prototype 独立验证层参数
PROTO_REJECT_THRESHOLD = 0.35    # prototype 余弦距离超过此值则拒绝匹配
                                 # 余弦距离 0=相同方向, 0.5=正交, 1=反向
                                 # 0.35 对应 cos_sim ≈ 0.3，即方向相关性较弱
PROTO_WARMUP_SAMPLES   = 5       # prototype 需积累至少 5 个样本才启用验证
PROTO_ABSORB_RATE_HIGH = 0.3
PROTO_ABSORB_RATE_LOW  = 0.05
PROTO_CONF_HIGH_THRESH = 0.7

# Gating 参数
GATING_LR          = 0.02
GATING_MOMENTUM    = 0.8
GATING_CONTEXT_DIM = 6
GATING_TEMPERATURE = 1.2
GATING_ENTROPY_REG = 0.01
GATING_MIN_WEIGHT  = 0.05


def _env_flag(name, default=True):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_float(name, default):
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_omega(name):
    v = os.getenv(name)
    if not v:
        return None
    try:
        vals = [float(x.strip()) for x in v.split(",")]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    arr = np.array(vals, dtype=np.float32)
    s = float(arr.sum())
    if s <= 1e-9:
        return None
    return arr / s


V6_USE_A = _env_flag("V6_USE_A", True)
V6_USE_B = _env_flag("V6_USE_B", True)
V6_USE_C = _env_flag("V6_USE_C", True)
V6_USE_D = _env_flag("V6_USE_D", True)
V6_USE_P = _env_flag("V6_USE_P", True)
V6_USE_G = _env_flag("V6_USE_G", True)
V6_OMEGA_OVERRIDE = _env_omega("V6_OMEGA")
V6_PROTO_THRESH = _env_float("V6_PROTO_THRESH", PROTO_REJECT_THRESHOLD)
V6_GATING_TEMP = _env_float("V6_GATING_TEMPERATURE", GATING_TEMPERATURE)


class TrackState(Enum):
    TRACKED    = auto()
    OVERLAP_FG = auto()
    OVERLAP_BG = auto()
    LOST       = auto()

def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-6)

def compute_bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def histogram_similarity(hist_a, hist_b):
    return float(np.sum(np.sqrt(np.maximum(hist_a * hist_b, 0.0))))

def motion_consistency(predicted_pos, observed_pos):
    dist = np.linalg.norm(np.array(predicted_pos) - np.array(observed_pos, dtype=float))
    return 1.0 / (1.0 + dist)

def safe_roi(frame, bbox):
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(frame.shape[1] - 1, int(bbox[2]))
    y2 = min(frame.shape[0] - 1, int(bbox[3]))
    if x1 >= x2 or y1 >= y2:
        return None
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None

def extract_histogram(roi):
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()
    except Exception:
        return None

def softmax(x, temperature=1.0):
    x_scaled = x / max(temperature, 1e-6)
    e = np.exp(x_scaled - np.max(x_scaled))
    return e / e.sum()


class MultiScaleWatermark:
    @staticmethod
    def extract(roi):
        try:
            r = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float64)
            coeffs = pywt.wavedec2(gray, 'db1', level=2)
            ll2 = coeffs[0]
            lh1 = coeffs[2][0]
            hl1 = coeffs[2][1]
            lh1_ds = cv2.resize(lh1, (16, 16), interpolation=cv2.INTER_AREA)
            hl1_ds = cv2.resize(hl1, (16, 16), interpolation=cv2.INTER_AREA)
            return {
                'LL2': MultiScaleWatermark._ternary_quantize(ll2).flatten(),
                'LH1': MultiScaleWatermark._ternary_quantize(lh1_ds).flatten(),
                'HL1': MultiScaleWatermark._ternary_quantize(hl1_ds).flatten(),
                'raw_LL2': ll2.flatten(),
                'raw_LH1': lh1_ds.flatten(),
                'raw_HL1': hl1_ds.flatten(),
            }
        except Exception:
            return None

    @staticmethod
    def _ternary_quantize(subband):
        mu = np.median(subband)
        sigma = np.std(subband)
        delta = TERNARY_DELTA * sigma
        result = np.zeros_like(subband, dtype=np.float32)
        result[subband > mu + delta] = 1.0
        result[subband < mu - delta] = -1.0
        return result

    @staticmethod
    def compute_similarity(desc_a, desc_b, stability_mask=None):
        valid = (desc_a != 0) & (desc_b != 0)
        n_valid = valid.sum()
        if n_valid == 0:
            return 0.5
        matches = (desc_a[valid] == desc_b[valid])
        if stability_mask is not None and len(stability_mask) == len(desc_a):
            weights = np.ones(n_valid, dtype=np.float32)
            weights[stability_mask[valid]] = IDENTITY_BIT_WEIGHT
            return float((matches.astype(np.float32) * weights).sum() / max(weights.sum(), 1e-6))
        return float(matches.sum() / n_valid)


class TemporalWatermarkFingerprint:
    def __init__(self, window_size=TEMPORAL_WINDOW):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.stability_mask = None
        self.identity_signature = None

    def update(self, descriptor):
        self.history.append(descriptor.copy())
        if len(self.history) >= 3:
            self._update_stability()

    def _update_stability(self):
        n = len(self.history)
        stacked = np.array(list(self.history))
        pos_count = (stacked > 0).sum(axis=0)
        neg_count = (stacked < 0).sum(axis=0)
        nonzero_count = pos_count + neg_count
        dominant_positive = pos_count >= neg_count
        dominant_count = np.where(dominant_positive, pos_count, neg_count)
        stability = np.zeros(stacked.shape[1], dtype=np.float32)
        valid = nonzero_count > 0
        stability[valid] = dominant_count[valid].astype(float) / nonzero_count[valid]
        min_obs = max(2, n // 2)
        self.stability_mask = (stability >= STABILITY_THRESHOLD) & (nonzero_count >= min_obs)
        self.identity_signature = np.zeros(stacked.shape[1], dtype=np.float32)
        self.identity_signature[self.stability_mask & dominant_positive] = 1.0
        self.identity_signature[self.stability_mask & ~dominant_positive] = -1.0

    def get_matching_info(self):
        if len(self.history) == 0:
            return None, None, None
        return self.history[-1], self.stability_mask, self.identity_signature

class ContextGating:
    def __init__(self):
        self.W = np.zeros((4, GATING_CONTEXT_DIM), dtype=np.float32)
        self.b = np.zeros(4, dtype=np.float32)
        self.W_momentum = np.zeros_like(self.W)
        self.b_momentum = np.zeros_like(self.b)
        self.last_context = None
        self.last_weights = None
        self.disc_ema = np.ones(4, dtype=np.float32) * 0.25
        self.temperature = V6_GATING_TEMP

    def compute_weights(self, context):
        logits = self.W @ context + self.b
        logits += self.disc_ema
        raw_weights = softmax(logits, temperature=self.temperature)
        weights = np.maximum(raw_weights, GATING_MIN_WEIGHT)
        weights = weights / weights.sum()
        self.last_context = context.copy()
        self.last_weights = weights.copy()
        return weights

    def update_with_discriminability(self, D_components, matched_pairs, n_tracks, n_dets):
        if len(matched_pairs) == 0 or n_tracks < 2 or n_dets < 2:
            return

        feature_keys = COMPONENT_KEYS
        discriminability = np.zeros(4, dtype=np.float32)

        for k, key in enumerate(feature_keys):
            D = D_components.get(key)
            if D is None:
                continue

            matched_costs = []
            for r, c in matched_pairs:
                if r < D.shape[0] and c < D.shape[1]:
                    matched_costs.append(D[r, c])

            if len(matched_costs) == 0:
                continue

            avg_matched = np.mean(matched_costs)

            non_matched_costs = []
            for r, c in matched_pairs:
                if r < D.shape[0] and c < D.shape[1]:
                    for j in range(D.shape[1]):
                        if j != c:
                            non_matched_costs.append(D[r, j])
                    for i in range(D.shape[0]):
                        if i != r:
                            non_matched_costs.append(D[i, c])

            if len(non_matched_costs) == 0:
                continue

            avg_non_matched = np.mean(non_matched_costs)
            discriminability[k] = max(avg_non_matched - avg_matched, 0.0)

        total_disc = discriminability.sum()
        if total_disc > 1e-6:
            disc_normalized = discriminability / total_disc
        else:
            disc_normalized = np.ones(4, dtype=np.float32) * 0.25

        self.disc_ema = 0.9 * self.disc_ema + 0.1 * disc_normalized

        if self.last_context is not None and self.last_weights is not None:
            target_diff = disc_normalized - self.last_weights
            grad_logits = target_diff * (self.last_weights * (1 - self.last_weights) + 1e-6)

            uniform = np.ones(4, dtype=np.float32) / 4.0
            entropy_grad = GATING_ENTROPY_REG * (uniform - self.last_weights)

            total_grad = grad_logits + entropy_grad
            grad_W = np.outer(total_grad, self.last_context)
            grad_b = total_grad

            self.W_momentum = GATING_MOMENTUM * self.W_momentum + GATING_LR * grad_W
            self.b_momentum = GATING_MOMENTUM * self.b_momentum + GATING_LR * grad_b

            self.W += self.W_momentum
            self.b += self.b_momentum

            self.W = np.clip(self.W, -1.5, 1.5)
            self.b = np.clip(self.b, -2.0, 2.0)

    def get_current_weights(self):
        if self.last_weights is not None:
            return self.last_weights
        return OMEGA_BASE.copy()

class PrototypeModule:
    """
    Prototype 独立维护每个目标的全局身份表达。

    设计要点：
    1. 不参与 Hungarian 匹配的代价矩阵
    2. 作为匹配后的身份验证层：拒绝身份不一致的匹配
    3. 更新速度由匹配置信度决定：
       - 高置信度匹配 → 快速吸收
       - 低置信度匹配 → 弱吸收
    """

    def __init__(self):
        self.support_set = deque(maxlen=PROTOTYPE_SUPPORT_SIZE)
        self.support_weights = deque(maxlen=PROTOTYPE_SUPPORT_SIZE)
        self.prototype = None

        rng = np.random.RandomState(42)
        self.proj_wm   = rng.randn(12, 256).astype(np.float32) / np.sqrt(256)
        self.proj_hist = rng.randn(12, 256).astype(np.float32) / np.sqrt(256)

    def build_embedding(self, wm_ll2, hist_features, velocity, bbox):
        parts = []
        if wm_ll2 is not None and len(wm_ll2) == 256:
            parts.append(self.proj_wm @ wm_ll2.astype(np.float32))
        else:
            parts.append(np.zeros(12, dtype=np.float32))

        if hist_features is not None and len(hist_features) == 256:
            parts.append(self.proj_hist @ hist_features.astype(np.float32))
        else:
            parts.append(np.zeros(12, dtype=np.float32))

        vx, vy = velocity if velocity else (0.0, 0.0)
        parts.append(np.array([vx, vy, 0.0, 0.0], dtype=np.float32))

        if bbox is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            aspect = w / max(h, 1)
            area = w * h
            parts.append(np.array([w / 100.0, h / 100.0, aspect, area / 10000.0],
                                  dtype=np.float32))
        else:
            parts.append(np.zeros(4, dtype=np.float32))

        return np.concatenate(parts)

    def add_to_support(self, embedding, confidence=1.0):
        """
        【v6】加入 support set，并记录匹配置信度作为权重。
        高置信度样本在 prototype 计算中占更大权重。
        """
        self.support_set.append(embedding.copy())
        self.support_weights.append(float(confidence))
        self._update_prototype()

    def _update_prototype(self):
        """【v6】加权均值：置信度高的样本贡献更大"""
        if len(self.support_set) == 0:
            self.prototype = None
            return
        embs = np.array(list(self.support_set))
        weights = np.array(list(self.support_weights), dtype=np.float32)
        weights = weights / max(weights.sum(), 1e-6)
        self.prototype = np.sum(embs * weights[:, None], axis=0)

    def is_warmed_up(self, min_samples=PROTO_WARMUP_SAMPLES):
        """【v6 修复】prototype 是否有足够样本用于可靠验证"""
        return len(self.support_set) >= min_samples

    def distance_to(self, query_embedding):
        """
        【v6 修复】改用余弦距离，值域稳定在 [0, 1]。
        0 = 方向完全一致
        1 = 方向正交或反向
        """
        if self.prototype is None:
            return 0.5
        # 归一化后计算 1 - cos_similarity
        p_norm = np.linalg.norm(self.prototype)
        q_norm = np.linalg.norm(query_embedding)
        if p_norm < 1e-6 or q_norm < 1e-6:
            return 0.5
        cos_sim = float(np.dot(self.prototype, query_embedding) / (p_norm * q_norm))
        # cos_sim ∈ [-1, 1] → distance ∈ [0, 1]
        return (1.0 - cos_sim) / 2.0

    def verify_identity(self, query_embedding, threshold=PROTO_REJECT_THRESHOLD):
        """
        【v6 修复】身份验证，仅在 prototype 充分 warm-up 后才启用。
        返回: (is_same_identity, distance)
        """
        # 冷启动保护：prototype 不成熟时不拒绝任何匹配
        if self.prototype is None or not self.is_warmed_up():
            return True, 0.5

        dist = self.distance_to(query_embedding)
        return dist < threshold, dist


class TrackableObject:
    _shared_prototype = None

    def __init__(self, objectID, centroid, bbox, frame=None):
        self.objectID = objectID
        self.centroids = deque([centroid], maxlen=5_000_000)
        self.bbox = bbox
        self.velocity = [0.0, 0.0]

        self.kalman = self._init_kalman()
        self._update_kalman(centroid)

        self.hist_features = None
        self.ms_watermark = None
        self.temporal_fp_LL2 = TemporalWatermarkFingerprint()
        self.temporal_fp_LH1 = TemporalWatermarkFingerprint()
        self.temporal_fp_HL1 = TemporalWatermarkFingerprint()
        self.active_subbands = ['LL2', 'LH1']
        self.feature_history = deque(maxlen=10)

        if TrackableObject._shared_prototype is None:
            TrackableObject._shared_prototype = PrototypeModule()
        self.proto = PrototypeModule()
        self.proto.proj_wm = TrackableObject._shared_prototype.proj_wm
        self.proto.proj_hist = TrackableObject._shared_prototype.proj_hist

        self.state = TrackState.TRACKED
        self.lost_frames = 0
        self.last_matched_bbox = None
        self.match_confidence = 0.0

        # 【v6】统计 prototype 拒绝次数（用于调试和监控）
        self.proto_reject_count = 0
        self.enable_proto = V6_USE_P

        if frame is not None:
            self._update_visual_features(frame, bbox)
            # 初始帧用最高置信度注入 prototype
            if self.enable_proto:
                self._update_prototype_with_confidence(frame, bbox, confidence=1.0)

    def _init_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        kf.R *= 10; kf.Q *= 0.1
        return kf

    def _update_kalman(self, centroid):
        self.kalman.predict()
        self.kalman.update(np.array([centroid[0], centroid[1]]))
        state = self.kalman.x
        self.velocity = [float(state[2][0]), float(state[3][0])]

    def kalman_predict_only(self):
        self.kalman.predict()
        state = self.kalman.x
        self.velocity = [float(state[2][0]), float(state[3][0])]
        return (float(state[0][0]), float(state[1][0]))

    def get_kalman_predicted_position(self):
        state = self.kalman.x
        return (float(state[0][0]), float(state[1][0]))

    def matched_update(self, centroid, bbox, frame, freeze_features=False,
                       confidence=0.0, update_proto=True):
        """
        【v6】匹配更新分两步：
        1. 更新位置和局部视觉特征（零水印、HSV）
        2. 根据置信度决定是否以及如何更新 prototype
        """
        self.centroids.append(centroid)
        self.bbox = bbox
        self.last_matched_bbox = list(bbox)
        self._update_kalman(centroid)
        self.lost_frames = 0
        self.match_confidence = confidence

        if not freeze_features and frame is not None:
            # 局部视觉特征总是更新（供下一帧 Hungarian 使用）
            self._update_visual_features(frame, bbox)

            # 【v6】Prototype 置信度加权更新
            if update_proto and self.enable_proto:
                self._update_prototype_with_confidence(frame, bbox, confidence)

    def _update_prototype_with_confidence(self, frame, bbox, confidence):
        if not self.enable_proto:
            return

        """
        【v6 新增】按置信度更新 prototype。

        - confidence > PROTO_CONF_HIGH_THRESH: 高置信度 → 完整权重加入
        - 否则: 低权重加入（或不加入，如果是 LOST 恢复的低置信度）
        """
        # LOST 状态下的低置信度匹配完全不加入（防止误匹配污染）
        if self.state == TrackState.LOST and confidence < 0.4:
            return

        # OVERLAP_BG 状态下不更新 prototype（防止前景污染）
        if self.state == TrackState.OVERLAP_BG:
            return

        roi = safe_roi(frame, bbox)
        if roi is None:
            return

        ms_desc = self.ms_watermark
        wm_ll2 = ms_desc['LL2'] if ms_desc else None

        embedding = self.proto.build_embedding(
            wm_ll2, self.hist_features, self.velocity, bbox)

        # 将匹配置信度作为 support set 的权重
        self.proto.add_to_support(embedding, confidence=confidence)

    def mark_lost(self):
        self.state = TrackState.LOST
        self.lost_frames += 1
        self.kalman_predict_only()

    def _update_visual_features(self, frame, bbox):
        roi = safe_roi(frame, bbox)
        if roi is None:
            return

        hist = extract_histogram(roi)
        if hist is not None:
            if self.hist_features is not None:
                self.hist_features = EMA_ALPHA * self.hist_features + (1 - EMA_ALPHA) * hist
            else:
                self.hist_features = hist

        ms_desc = MultiScaleWatermark.extract(roi)
        if ms_desc is not None:
            self.ms_watermark = ms_desc
            self.temporal_fp_LL2.update(ms_desc['LL2'])
            self.temporal_fp_LH1.update(ms_desc['LH1'])
            self.temporal_fp_HL1.update(ms_desc['HL1'])
            self._update_active_subbands(ms_desc)

        self.feature_history.append((hist, ms_desc))

    def _update_active_subbands(self, ms_desc):
        scores = {
            'LL2': float(np.var(ms_desc['raw_LL2'])),
            'LH1': float(np.var(ms_desc['raw_LH1'])),
            'HL1': float(np.var(ms_desc['raw_HL1'])),
        }
        sorted_bands = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        self.active_subbands = sorted_bands[:ENTROPY_TOP_K]

    def compute_watermark_similarity(self, det_ms_desc):
        if self.ms_watermark is None or det_ms_desc is None:
            return 0.5
        scores = []
        for band in self.active_subbands:
            tfp = {'LL2': self.temporal_fp_LL2, 'LH1': self.temporal_fp_LH1,
                   'HL1': self.temporal_fp_HL1}[band]
            _, stability_mask, _ = tfp.get_matching_info()
            sim = MultiScaleWatermark.compute_similarity(
                self.ms_watermark[band], det_ms_desc[band], stability_mask)
            scores.append(sim)
        return float(np.mean(scores)) if scores else 0.5


class RobustMultiObjectTracker:
    def __init__(self, maxDisappeared=300, maxDistance=100):
        self.nextObjectID   = 0
        self.objects         = {}
        self.disappeared     = {}
        self.maxDisappeared  = maxDisappeared
        self.maxDistance     = maxDistance
        self.occlusion_pairs = set()

        self.gating = ContextGating()
        self.current_weights = OMEGA_BASE.copy()
        self.enable_a = V6_USE_A
        self.enable_b = V6_USE_B
        self.enable_c = V6_USE_C
        self.enable_d = V6_USE_D
        self.enable_proto = V6_USE_P
        self.enable_gating = V6_USE_G
        self.proto_reject_threshold = V6_PROTO_THRESH
        self.fixed_weights = V6_OMEGA_OVERRIDE.copy() if V6_OMEGA_OVERRIDE is not None else OMEGA_BASE.copy()
        self.base_weights_used = self._apply_component_mask(self.fixed_weights.copy())
        self.gating_weights_used = self.base_weights_used.copy()

        self.prev_centroids = {}

        # 【v6】prototype 拒绝统计
        self.total_proto_rejects = 0

    def _apply_component_mask(self, weights):
        mask = np.array([
            1.0 if self.enable_a else 0.0,
            1.0 if self.enable_b else 0.0,
            1.0 if self.enable_c else 0.0,
            1.0 if self.enable_d else 0.0,
        ], dtype=np.float32)
        out = np.asarray(weights, dtype=np.float32) * mask
        s = float(out.sum())
        if s <= 1e-9:
            out = mask.copy()
            s = float(out.sum())
        if s <= 1e-9:
            out = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            s = 1.0
        return out / s

    def register(self, centroid, bbox, frame=None):
        obj = TrackableObject(self.nextObjectID, centroid, bbox, frame)
        self.objects[self.nextObjectID] = obj
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def _compute_depth_scores(self, bboxes):
        areas = np.array([compute_bbox_area(b) for b in bboxes], dtype=float)
        total = areas.sum()
        if total < 1e-6:
            return np.ones(len(bboxes)) / max(len(bboxes), 1)
        return areas / total

    def _compute_context(self, objectIDs, bboxes, n_dets):
        n_tracks = len(objectIDs)
        max_iou = 0.0
        track_bboxes = [self.objects[oid].bbox for oid in objectIDs if oid in self.objects]
        for i in range(len(track_bboxes)):
            for j in range(i + 1, len(track_bboxes)):
                iou = compute_iou(track_bboxes[i], track_bboxes[j])
                max_iou = max(max_iou, iou)

        if len(track_bboxes) >= 2:
            centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in track_bboxes])
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, 1e6)
            min_dist = dists.min() / max(self.maxDistance, 1)
        else:
            min_dist = 1.0

        n_dets_norm = min(n_dets / max(n_tracks, 1), 2.0) / 2.0

        n_lost = sum(1 for oid in objectIDs
                     if oid in self.objects and self.objects[oid].state == TrackState.LOST)
        n_lost_norm = n_lost / max(n_tracks, 1)

        avg_overlap = max_iou

        displacements = []
        for oid in objectIDs:
            if oid not in self.objects:
                continue
            obj = self.objects[oid]
            if obj.state == TrackState.LOST:
                continue
            current_cent = obj.centroids[-1]
            if oid in self.prev_centroids:
                prev = self.prev_centroids[oid]
                dx = current_cent[0] - prev[0]
                dy = current_cent[1] - prev[1]
                disp = np.sqrt(dx**2 + dy**2)
                displacements.append(disp)

        if len(displacements) > 0:
            avg_disp = np.mean(displacements)
            displacement_norm = min(avg_disp / max(self.maxDistance, 1), 1.0)
        else:
            displacement_norm = 0.0

        return np.array([max_iou, min_dist, n_dets_norm, n_lost_norm,
                         avg_overlap, displacement_norm],
                        dtype=np.float32)

    def _build_cost_matrix(self, frame, objectIDs, objectCentroids,
                           inputCentroids, bboxes, weights):

        n_tracks = len(objectIDs)
        n_dets   = len(inputCentroids)

        D_spatial = cdist(objectCentroids, inputCentroids)
        D_spatial_norm = D_spatial / max(D_spatial.max(), 1e-6)
        D_hist      = np.ones((n_tracks, n_dets), dtype=float)
        D_motion    = np.ones((n_tracks, n_dets), dtype=float)
        D_watermark = np.ones((n_tracks, n_dets), dtype=float)

        # 预提取检测特征
        det_ms_descs, det_hists = [], []
        for j in range(n_dets):
            roi = safe_roi(frame, bboxes[j])
            if roi is not None:
                det_ms_descs.append(MultiScaleWatermark.extract(roi))
                det_hists.append(extract_histogram(roi))
            else:
                det_ms_descs.append(None)
                det_hists.append(None)

        for i, oid in enumerate(objectIDs):
            obj = self.objects[oid]
            kalman_pred = obj.get_kalman_predicted_position()

            for j in range(n_dets):
                mcons = motion_consistency(kalman_pred, inputCentroids[j])
                D_motion[i, j] = 1.0 - mcons

                if obj.hist_features is not None and det_hists[j] is not None:
                    sim = histogram_similarity(obj.hist_features, det_hists[j])
                    D_hist[i, j] = 1.0 - min(sim, 1.0)

                if det_ms_descs[j] is not None:
                    # 【v6】零水印项只包含零水印相似度，不再融入 prototype
                    wm_sim = obj.compute_watermark_similarity(det_ms_descs[j])
                    D_watermark[i, j] = 1.0 - wm_sim

        cost_terms = [D_hist, D_motion, D_spatial_norm, D_watermark]
        D_combined = sum(float(w) * D for w, D in zip(weights, cost_terms))

        for i, oid in enumerate(objectIDs):
            obj = self.objects[oid]
            max_dist = self.maxDistance * 1.5 if obj.state == TrackState.LOST else self.maxDistance
            D_combined[i, D_spatial[i] > max_dist] = float('inf')

        return D_combined, {
            'hist': D_hist,
            'motion': D_motion,
            'spatial': D_spatial_norm,
            'watermark': D_watermark,
        }

    def _compute_det_embedding(self, frame, bbox):
        """为检测计算 prototype 嵌入向量"""
        roi = safe_roi(frame, bbox)
        if roi is None:
            return None

        ms_desc = MultiScaleWatermark.extract(roi)
        hist = extract_histogram(roi)

        proto_ref = TrackableObject._shared_prototype or PrototypeModule()
        wm_ll2 = ms_desc['LL2'] if ms_desc else None
        return proto_ref.build_embedding(wm_ll2, hist, (0.0, 0.0), bbox)

    def _update_overlap_states(self, matched_ids):
        ids = list(matched_ids)
        self.occlusion_pairs = set()
        valid_ids = [oid for oid in ids if oid in self.objects]
        if len(valid_ids) < 2:
            for oid in valid_ids:
                self.objects[oid].state = TrackState.TRACKED
            return

        bboxes = [self.objects[oid].bbox for oid in valid_ids]
        depth_scores = self._compute_depth_scores(bboxes)
        overlap_bg_set = set()

        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                iou = compute_iou(bboxes[i], bboxes[j])
                depth_diff = abs(depth_scores[i] - depth_scores[j])
                if iou > TAU_IOU and depth_diff > TAU_DEPTH:
                    self.occlusion_pairs.add((valid_ids[i], valid_ids[j]))
                    if depth_scores[i] > depth_scores[j]:
                        overlap_bg_set.add(valid_ids[j])
                        self.objects[valid_ids[i]].state = TrackState.OVERLAP_FG
                    else:
                        overlap_bg_set.add(valid_ids[i])
                        self.objects[valid_ids[j]].state = TrackState.OVERLAP_FG

        for oid in valid_ids:
            if oid in overlap_bg_set:
                self.objects[oid].state = TrackState.OVERLAP_BG
            elif self.objects[oid].state != TrackState.OVERLAP_FG:
                self.objects[oid].state = TrackState.TRACKED

    def update(self, frame, detections):
        bboxes = [list(d[:4]) for d in detections]

        if len(bboxes) == 0:
            for oid in list(self.objects.keys()):
                self.objects[oid].mark_lost()
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = np.array([
            (int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in bboxes
        ])

        if len(self.objects) == 0:
            for i in range(len(bboxes)):
                self.register(inputCentroids[i], bboxes[i], frame)
            return self.objects

        allIDs = list(self.objects.keys())

        snapshot_centroids = {}
        for oid in allIDs:
            obj = self.objects[oid]
            if obj.state != TrackState.LOST:
                snapshot_centroids[oid] = obj.centroids[-1]

        allCentroids = []
        for oid in allIDs:
            obj = self.objects[oid]
            if obj.state == TrackState.LOST:
                allCentroids.append(obj.get_kalman_predicted_position())
            else:
                allCentroids.append(obj.centroids[-1])
        allCentroids = np.array(allCentroids, dtype=float)

        context = self._compute_context(allIDs, bboxes, len(bboxes))
        self.base_weights_used = self._apply_component_mask(self.fixed_weights.copy())
        if self.enable_gating:
            self.gating_weights_used = self._apply_component_mask(self.gating.compute_weights(context))
            self.current_weights = self.gating_weights_used.copy()
        else:
            self.gating_weights_used = self.base_weights_used.copy()
            self.current_weights = self.base_weights_used.copy()
        self.prev_centroids = snapshot_centroids

        # Step 1: Hungarian 匹配（只用 4 个局部特征）
        D_combined, D_components = self._build_cost_matrix(
            frame, allIDs, allCentroids, inputCentroids, bboxes,
            self.current_weights)

        D_for_assign = np.where(np.isinf(D_combined), 1e9, D_combined)
        rows, cols = linear_sum_assignment(D_for_assign)

        # Step 2: 【v6 修订】Prototype 不再在主匹配路径拒绝
        # 主匹配完全信任 Hungarian 结果；prototype 距离仅作为置信度信号
        matched_track_ids = set()
        usedCols = set()
        matched_pairs = []

        for r, c in zip(rows, cols):
            if D_combined[r, c] == float('inf'):
                continue

            oid = allIDs[r]
            obj = self.objects[oid]

            match_conf = 1.0 - min(D_combined[r, c], 1.0)

            # Prototype 距离只用于置信度调整，不做拒绝
            det_embedding = self._compute_det_embedding(frame, bboxes[c])
            if (self.enable_proto and det_embedding is not None and
                obj.proto.prototype is not None and
                obj.proto.is_warmed_up()):

                proto_dist = obj.proto.distance_to(det_embedding)

                # 只有当 prototype 距离特别大时才"软惩罚"置信度
                # 但不拒绝匹配
                if proto_dist > self.proto_reject_threshold:
                    obj.proto_reject_count += 1
                    self.total_proto_rejects += 1
                    # 降低 prototype 更新速度，但仍接受位置更新
                    match_conf *= 0.3
                else:
                    # 验证通过时强化置信度
                    proto_boost = 1.0 - proto_dist
                    match_conf = 0.6 * match_conf + 0.4 * proto_boost

            obj.matched_update(
                inputCentroids[c], bboxes[c], frame,
                freeze_features=False, confidence=match_conf,
                update_proto=self.enable_proto)
            self.disappeared[oid] = 0
            matched_track_ids.add(oid)
            usedCols.add(c)
            matched_pairs.append((r, c))

        # Step 3: 遮挡状态
        self._update_overlap_states(matched_track_ids)

        # OVERLAP_BG 特征回退
        for oid in matched_track_ids:
            obj = self.objects[oid]
            if obj.state == TrackState.OVERLAP_BG and len(obj.feature_history) >= 2:
                prev_hist, prev_ms = obj.feature_history[-2]
                if prev_hist is not None:
                    obj.hist_features = prev_hist
                if prev_ms is not None:
                    obj.ms_watermark = prev_ms

        # Step 4: 未匹配目标 → LOST
        for oid in allIDs:
            if oid not in matched_track_ids and oid in self.objects:
                self.objects[oid].mark_lost()
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

        # Step 5: 未匹配新检测 → 尝试 LOST Re-ID 或注册
        for c in set(range(len(bboxes))) - usedCols:
            matched_lost = self._try_match_lost(frame, inputCentroids[c], bboxes[c])
            if matched_lost is not None:
                # LOST 恢复时也要经过 prototype 验证（已在 _try_match_lost 内部做了）
                self.objects[matched_lost].matched_update(
                    inputCentroids[c], bboxes[c], frame,
                    freeze_features=False, confidence=0.5,
                    update_proto=self.enable_proto)
                self.objects[matched_lost].state = TrackState.TRACKED
                self.disappeared[matched_lost] = 0
            else:
                self.register(inputCentroids[c], bboxes[c], frame)

        # 在线更新 gating
        if self.enable_gating:
            self.gating.update_with_discriminability(
                D_components, matched_pairs, len(allIDs), len(bboxes))

        return self.objects

    def _try_match_lost(self, frame, det_centroid, det_bbox):
        """
        【v6 修订】LOST 目标恢复：
        - prototype 已 warm-up 的目标：用 prototype 做强验证
        - prototype 未 warm-up（新目标很快丢失）：用传统特征匹配
        """
        det_embedding = self._compute_det_embedding(frame, det_bbox)

        roi = safe_roi(frame, det_bbox)
        if roi is None:
            return None

        det_ms = MultiScaleWatermark.extract(roi)
        det_hist = extract_histogram(roi)

        best_oid = None
        best_score = -1.0

        for oid, obj in self.objects.items():
            if obj.state != TrackState.LOST:
                continue

            pred_pos = obj.get_kalman_predicted_position()
            spatial_dist = np.linalg.norm(
                np.array(pred_pos) - np.array(det_centroid, dtype=float))
            if spatial_dist > self.maxDistance * 2.0:
                continue

            # Prototype 验证：只对 warm-up 完成的目标启用强验证
            proto_score = 0.5
            if self.enable_proto and det_embedding is not None and obj.proto.is_warmed_up():
                is_same, proto_dist = obj.proto.verify_identity(
                    det_embedding,
                    threshold=self.proto_reject_threshold * 1.2
                )
                if not is_same:
                    continue
                proto_score = 1.0 - proto_dist

            score = 0.0
            checks = 0

            if self.enable_proto and obj.proto.is_warmed_up():
                score += proto_score
                checks += 1

            if self.enable_d and det_ms is not None:
                score += obj.compute_watermark_similarity(det_ms)
                checks += 1

            if self.enable_a and det_hist is not None and obj.hist_features is not None:
                score += min(histogram_similarity(obj.hist_features, det_hist), 1.0)
                checks += 1

            if self.enable_b:
                score += motion_consistency(pred_pos, det_centroid)
                checks += 1

            # 空间距离的倒数也作为信号（LOST 恢复时近距离更可信）
            if self.enable_c:
                spatial_score = 1.0 / (1.0 + spatial_dist / self.maxDistance)
                score += spatial_score
                checks += 1

            if checks == 0:
                continue

            avg = score / checks
            if avg > best_score and avg > 0.35:  # 稍高的阈值防止误匹配
                best_score = avg
                best_oid = oid

        return best_oid

        return best_oid


class IdentityPreservationPipeline:
    def __init__(self, yolo_model_path):
        self.detector = YOLO(yolo_model_path)
        # maxDisappeared 从 300 降到 30：遮挡超过1秒(30帧)的目标真的丢了
        # maxDistance 调高：适应更大的帧间位移
        self.tracker  = RobustMultiObjectTracker(maxDisappeared=30, maxDistance=150)
        self.trajectory_history = {}
        self.color_palette = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        self.detection_size = (1280, 720)
        self.original_size  = None
        self.current_id_positions = {}

    def get_color(self, idx):
        return self.color_palette[idx % len(self.color_palette)]

    def scale_coordinates(self, coords, from_size, to_size):
        sx = to_size[0] / from_size[0]
        sy = to_size[1] / from_size[1]
        if len(coords) == 4:
            return [int(coords[0]*sx), int(coords[1]*sy),
                    int(coords[2]*sx), int(coords[3]*sy)]
        return [int(coords[0]*sx), int(coords[1]*sy)]

    def draw_dashed_rect(self, img, pt1, pt2, color, thickness=2, dash=8):
        x1, y1 = pt1; x2, y2 = pt2
        for seg in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                     ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
            self._draw_dashed_line(img, seg[0], seg[1], color, thickness, dash)

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash=8):
        dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
        n = int(dist/dash)
        if n == 0: cv2.line(img, pt1, pt2, color, thickness); return
        for i in range(n):
            s = i/n; e = (i+0.5)/n
            p1 = (int(pt1[0]*(1-s)+pt2[0]*s), int(pt1[1]*(1-s)+pt2[1]*s))
            p2 = (int(pt1[0]*(1-e)+pt2[0]*e), int(pt1[1]*(1-e)+pt2[1]*e))
            cv2.line(img, p1, p2, color, thickness)

    def optimize_id_position(self, frame, bbox, objectID):
        x1,y1,x2,y2 = map(int, bbox)
        for pos in [(x1,y1-10),(x1,y2+20),(x1-100,y1),(x2+10,y1),(x1,y1+(y2-y1)//2)]:
            if 0<=pos[0]<frame.shape[1] and 0<=pos[1]<frame.shape[0]:
                if not any(abs(pos[0]-op[0])<50 and abs(pos[1]-op[1])<20
                           for op in self.current_id_positions.values()):
                    self.current_id_positions[objectID] = pos
                    return pos
        d = (x1, y1-10); self.current_id_positions[objectID] = d; return d

    def process_frame(self, frame):
        oh, ow = frame.shape[:2]
        self.original_size = (ow, oh)
        self.current_id_positions = {}

        det_frame = cv2.resize(frame, self.detection_size)
        results = self.detector.predict(det_frame, conf=0.6, verbose=False)[0]
        scaled_dets = []
        for box in results.boxes:
            bd = box.data[0]
            x1,y1,x2,y2 = map(int, bd[:4])
            scaled_dets.append(self.scale_coordinates(
                [x1,y1,x2,y2], self.detection_size, self.original_size))

        objects = self.tracker.update(frame, scaled_dets)

        for oid, obj in objects.items():
            if oid not in self.trajectory_history:
                self.trajectory_history[oid] = deque(maxlen=100000)
            if obj.state != TrackState.LOST:
                self.trajectory_history[oid].append(obj.centroids[-1])
        for oid in list(self.trajectory_history.keys()):
            if oid not in objects: del self.trajectory_history[oid]

        output = frame.copy()
        for oid, obj in objects.items():
            color = self.get_color(oid)
            if obj.state == TrackState.LOST:
                pp = obj.get_kalman_predicted_position()
                if obj.last_matched_bbox:
                    fw=obj.last_matched_bbox[2]-obj.last_matched_bbox[0]
                    fh=obj.last_matched_bbox[3]-obj.last_matched_bbox[1]
                    self.draw_dashed_rect(output,
                        (int(pp[0]-fw/2),int(pp[1]-fh/2)),
                        (int(pp[0]+fw/2),int(pp[1]+fh/2)), color, 2)
                    cv2.putText(output, f"UAV {oid}|LOST({obj.lost_frames})",
                        (int(pp[0]-fw/2),int(pp[1]-fh/2)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue

            x1,y1,x2,y2 = map(int, obj.bbox)
            if oid in self.trajectory_history and len(self.trajectory_history[oid])>1:
                cv2.polylines(output, [np.array(self.trajectory_history[oid],
                              dtype=np.int32)], False, color, 2)
            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)

            state_lbl = {TrackState.OVERLAP_FG:"FG",TrackState.OVERLAP_BG:"BG"}.get(obj.state,"")
            n_support = len(obj.proto.support_set)
            label = f"UAV {oid}"
            if state_lbl: label += f"|{state_lbl}"
            label += f" P:{n_support}"
            if obj.proto_reject_count > 0:
                label += f" R:{obj.proto_reject_count}"

            tp = self.optimize_id_position(output, (x1,y1,x2,y2), oid)
            cv2.putText(output, label, tp, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 状态栏
        w = self.tracker.current_weights
        n_t = len(objects)
        n_l = sum(1 for o in objects.values() if o.state==TrackState.LOST)
        n_rej = self.tracker.total_proto_rejects
        cv2.putText(output,
            f"T:{n_t} L:{n_l} w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}] ProtoRej:{n_rej}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return output


def process_video(video_path, yolo_model_path, output_path):
    pipeline = IdentityPreservationPipeline(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: {video_path}"); return

    W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5); total=int(cap.get(7))
    print(f"视频: {W}x{H} @ {fps:.1f}fps, {total} 帧")
    print(f"基础权重: {OMEGA_BASE}")
    print(f"Prototype: support={PROTOTYPE_SUPPORT_SIZE}, dim={PROTOTYPE_DIM}")
    print(f"Proto 验证: reject_thresh={PROTO_REJECT_THRESHOLD}")
    print(f"Gating: lr={GATING_LR}, momentum={GATING_MOMENTUM}\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    if not out.isOpened():
        out = cv2.VideoWriter(output_path.replace('.mp4','.avi'),
                              cv2.VideoWriter_fourcc(*'XVID'), fps, (W,H))

    count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            output = pipeline.process_frame(frame)
            if output.shape[:2] != (H,W): output = cv2.resize(output, (W,H))
            out.write(output); count += 1

            cv2.imshow("SwarmPI v6", cv2.resize(output, (1280, 720)))
            if count % 30 == 0:
                tr = pipeline.tracker
                w = tr.current_weights
                disc = tr.gating.disc_ema
                ctx = tr.gating.last_context
                disp_val = ctx[5] * tr.maxDistance if ctx is not None else 0.0
                print(f"  帧 {count:>5}/{total}  |  "
                      f"目标:{len(tr.objects)} 总ID:{tr.nextObjectID} "
                      f"ProtoRej:{tr.total_proto_rejects}  "
                      f"ω=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]  "
                      f"disc=[{disc[0]:.2f},{disc[1]:.2f},{disc[2]:.2f},{disc[3]:.2f}]  "
                      f"disp:{disp_val:.1f}px")
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); out.release(); cv2.destroyAllWindows()
        print(f"\n完成: {count} 帧 → {output_path}")
        print(f"总注册 ID: {pipeline.tracker.nextObjectID}")
        print(f"Prototype 累计拒绝匹配: {pipeline.tracker.total_proto_rejects} 次")
        w = pipeline.tracker.gating.get_current_weights()
        d = pipeline.tracker.gating.disc_ema
        print(f"最终 gating 权重:  ω=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}]")
        print(f"最终判别力分布: disc=[{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, {d[3]:.3f}]")
        print(f"  ω₁={w[0]:.3f} 空间距离  |  ω₂={w[1]:.3f} HSV直方图")
        print(f"  ω₃={w[2]:.3f} 运动一致性 |  ω₄={w[3]:.3f} 零水印")


def main():
    video_path = '../DJI_2.MP4'
    yolo_model_path = '../runs/detect/train7/weights/best.pt'
    output_path = 'DJI_2_tracking_v6.mp4'
    for p, n in [(video_path,"视频"),(yolo_model_path,"YOLO")]:
        if not os.path.exists(p): print(f"错误: {n}不存在: {p}"); return
    process_video(video_path, yolo_model_path, output_path)


if __name__ == '__main__':
    main()
