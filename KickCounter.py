# 필요한 전문가들 불러오기
from ultralytics import YOLO
import ctypes
import cv2
import numpy as np
from collections import defaultdict
import sys
import time
import math # 거리 계산을 위해 추가
import os

# [수정] 한글 폰트(PIL) 지원 라이브러리 추가
from PIL import ImageFont, ImageDraw, Image

# --- PyInstaller 환경 리소스 경로 함수 ---
def resource_path(relative_path):
    """ PyInstaller 환경에서 실행될 때 파일의 절대 경로를 리턴 """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- SORT 모듈 임포트 (PyInstaller 호환) ---
try:
    sort_path = resource_path('sort')
    if sort_path not in sys.path:
        sys.path.insert(0, sort_path)
    from sort import Sort 
except ImportError as e:
    print("="*60); 
    print(f"🚨 치명적 오류: 'sort' 모듈을 찾을 수 없습니다."); 
    print(f"ImportError: {e}");
    print("PyInstaller 빌드 시 '--add-data \"sort:sort\"' 옵션을 사용했는지 확인하세요.");
    print("="*60);
    sys.exit(1)
# ------------------------------------------

# === 유틸리티 함수 ===
def calculate_distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

# [★★★ 16:9 비율 고정 (레터박스) 1/6 ★★★]
def letterbox(img, new_shape=(1080, 1920), color=(0, 0, 0)):
    """
    원본 이미지를 비율을 유지한 채 new_shape 크기로 리사이즈하고,
    남는 공간을 color로 채웁니다(레터박스).
    """
    shape = img.shape[:2]  # 현재 높이, 너비
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 스케일 비율 (new_shape 높이 기준)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 비율을 유지한 리사이즈 크기 계산
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 패딩 계산 (너비, 높이)

    dw /= 2  # 좌우 패딩 분할
    dh /= 2  # 상하 패딩 분hal

    if shape[::-1] != new_unpad:  # 리사이즈가 필요하면
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 레터박스(테두리) 추가
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh) # 리사이즈된 이미지, 비율, 패딩값 반환


# === [버그 수정] 누락되었던 overlay_transparent 함수 추가 ===
def overlay_transparent(background, overlay, x, y):
    """
    background 이미지에 overlay 이미지를 (x, y) 위치에 투명도를 고려하여 합성합니다.
    overlay 이미지는 4채널(BGRA)이어야 합니다.
    """
    try:
        bg_h, bg_w = background.shape[:2]
        ol_h, ol_w = overlay.shape[:2]

        if x < -ol_w or x > bg_w or y < -ol_h or y > bg_h:
            return

        # 오버레이가 배경을 벗어나는 부분 클리핑
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(bg_w, x + ol_w)
        y_end = min(bg_h, y + ol_h)

        # 오버레이에서 가져올 부분 계산
        ol_x_start = max(0, -x)
        ol_y_start = max(0, -y)
        ol_x_end = ol_x_start + (x_end - x_start)
        ol_y_end = ol_y_start + (y_end - y_start)

        # 실제 ROI(Region of Interest)
        roi = background[y_start:y_end, x_start:x_end]
        ol_roi = overlay[ol_y_start:ol_y_end, ol_x_start:ol_x_end]

        if ol_roi.shape[2] != 4:
            # 오버레이가 4채널이 아니면 그냥 덮어쓰기 (오류 방지)
            roi[:] = ol_roi[:, :, :3]
            return

        # 알파 채널 분리 및 정규화
        alpha = ol_roi[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha

        # 채널별 합성
        for c in range(3):
            roi[:, :, c] = (alpha * ol_roi[:, :, c] +
                            alpha_inv * roi[:, :, c])
            
    except Exception as e:
        print(f"Error in overlay_transparent: {e}")
        # 오류 발생 시 해당 프레임은 합성을 건너뜀
        pass


# === 상태 정의 ===
MENU_START = 0
MENU_PLAYER_SELECT = 1
GAME_RUNNING = 2
GAME_COUNTDOWN = 3
GAME_TIMER_RUNNING = 4
GAME_OVER = 5

# === [신규] 캘리브레이션 박스 생성 함수 ===
def get_calibration_boxes(max_players, W, H, bottom_margin=20):
    boxes = []
    box_width = 300
    box_height = 100
    
    positions = []
    num_sections = max_players + 1
    for i in range(1, num_sections):
        positions.append((W // num_sections) * i)
                        
    for cx in positions:
        x1 = cx - (box_width // 2)
        y1 = H - bottom_margin - box_height
        x2 = cx + (box_width // 2)
        y2 = H - bottom_margin
        boxes.append((x1, y1, x2, y2))
        
    return boxes

# === [신규] 점이 박스 안에 있는지 확인하는 함수 ===
def is_point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

# === [신규] UI 그리기 함수 (PIL 사용) ===
def draw_menu_ui(state, frame_dims=(1920, 1080), fonts=None, mouse_pos=(0,0), player_select_img=None, splash_img=None):
    H, W = frame_dims[1], frame_dims[0] # (1080, 1920)
    
    PLAYER_SELECT_ZONES = [
        (98, 303, 402, 902),     # 1인 구역
        (509, 303, 813, 902),    # 2인 구역
        (920, 303, 1224, 902),   # 3인 구역
        (1331, 303, 1635, 902)   # 4인 구역
    ]
    active_hover_zone = -1 

    if state == MENU_START:
        # 1. 스플래시 스크린
        if splash_img is not None:
            frame = splash_img.copy()
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        return frame, active_hover_zone 

    elif state == MENU_PLAYER_SELECT:
        # 2. 인원 선택 화면
        if player_select_img is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8) 
            text = "플레이할 인원을 선택해 주세요. (1~4 키)"
            cv2.putText(frame, text, (W//2 - 300, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return frame, active_hover_zone
        
        frame = player_select_img.copy()

        # 마우스 호버(Hover) 감지
        for i, zone in enumerate(PLAYER_SELECT_ZONES):
            x1, y1, x2, y2 = zone
            if (x1 <= mouse_pos[0] <= x2) and (y1 <= mouse_pos[1] <= y2):
                active_hover_zone = i
                break
        
        # [수정] 호버 박스 그리기 삭제

        return frame, active_hover_zone

# === [신규] PIL Draw 객체에 직접 텍스트를 그리는 헬퍼 함수 ===
def draw_pil_text_on_image(draw, text, pos_top_left, font, text_color_rgb, bg_color_rgb=None, align="left"):
    """ 
    PIL 'draw' 객체에 직접 텍스트와 배경을 그립니다. 
    pos는 좌상단(top-left) 기준입니다.
    align: 'left' 또는 'center'
    """
    try:
        # 텍스트 크기 계산
        if hasattr(font, 'getbbox'): # Pillow 10+
            bbox = draw.textbbox((0,0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else: # Pillow < 10
            text_width, text_height = draw.textsize(text, font=font)

        # 정렬에 따라 x 좌표 조정
        draw_pos_x = pos_top_left[0]
        if align == "center":
            draw_pos_x -= text_width // 2
        
        # 실제 그릴 위치
        draw_pos = (draw_pos_x, pos_top_left[1])
        
        if bg_color_rgb is not None:
            # 그릴 위치 기준으로 bbox 다시 계산
            if hasattr(font, 'getbbox'):
                bbox = draw.textbbox(draw_pos, text, font=font)
            else:
                bbox = (draw_pos[0], draw_pos[1], draw_pos[0] + text_width, draw_pos[1] + text_height)
            
            # 배경 그리기 (계산된 bbox 기준)
            bg_tl = (bbox[0] - 2, bbox[1] - 2)
            bg_br = (bbox[2] + 2, bbox[3] + 2)
            
            draw.rectangle([bg_tl, bg_br], fill=bg_color_rgb)
        
        # 텍스트 그리기
        draw.text(draw_pos, text, font=font, fill=text_color_rgb)
    except Exception as e:
        pass # 오류 시 그리기를 건너뜀


# === [리팩토링] 게임 상태 변수 초기화 함수 ===
def create_game_state():
    """ 모든 게임 관련 상태 변수를 포함하는 딕셔너리를 생성하여 반환합니다. """
    return {
        'base_data': defaultdict(dict),
        'final_scores': {},
        'kick_counters': defaultdict(int),
        'l_kick_state': defaultdict(int),
        'r_kick_state': defaultdict(int),
        'l_reset_counter': defaultdict(int),
        'r_reset_counter': defaultdict(int),
        'person_kick_timer': defaultdict(int),
        'player_id_counter': 1,
        'track_id_to_player_id': {},
        'floor_timers': defaultdict(lambda: None),
        'floor_y_history': defaultdict(list),
        'calibrated_box_indices': set(),
        'active_calib_boxes': {}, # 현재 캘리브레이션 중인 박스 UI 추적용
        
        # [★★★ 1초 유예 기간 추가 ★★★]
        'lost_id_timers': {}, # 1초 유예 타이머 딕셔너리
        
        # 설정값
        'RESET_FRAME_COUNT': 3,
        'KICK_COOLDOWN_FRAMES': 5,
        'CALIBRATION_TIME': 2.0,
        'STABILITY_THRESH': 40,
        'KICK_THRESH_PIXELS_Y': 30,
        'KICK_THRESH_PIXELS_X': 1,
        'KICK_THRESH_RATIO_Z': 0.10,
        'JOINT_CONF_THRESH': 0.1,
        'CALIB_BOX_BOTTOM_MARGIN': 50
    }


# === [리팩토링] 핵심 로직 함수 (YOLO, SORT, 캘리브레이션, 킥 판정) ===
def process_frame_logic(frame, letterboxed_frame, ratio_x, ratio_y, pad_w, pad_h,
                        model, tracker, max_players, game_state, fonts, temp_draw,
                        calibration_boxes=None, 
                        is_game_mode=False): # [★★★ 게임 모드 제한 추가 ★★★]
    """
    원본 프레임을 받아 YOLO/SORT 분석, 캘리브레이션, 킥 판정을 수행하고
    그릴 텍스트 목록(pil_draw_list)을 반환합니다.
    game_state 딕셔너리를 직접 수정합니다.
    """

    # [★★★ UnboundLocalError 버그 수정 ★★★]
    pil_draw_list = []
    
    # === [★★★ 버그 수정 ★★★] ===
    results = model(frame, conf=0.6, verbose=False) 
    
    scale_x = ratio_x
    scale_y = ratio_y
    pad_x_val = pad_w
    pad_y_val = pad_h
    # === [버그 수정 끝] ===
    
    dets, keypoints_list = [], []
    for r in results:
        if r.keypoints is not None:
            xy_data, conf_data = getattr(r.keypoints, "xy", []), getattr(r.keypoints, "conf", [])
            if len(xy_data) != len(conf_data): continue
            for i in range(len(xy_data)):
                
                # --- [★★★ 버그 수정 ★★★] ---
                person_kp = xy_data[i].cpu().numpy()
                person_conf = conf_data[i].cpu().numpy()

                person_kp[:, 0] = (person_kp[:, 0] * scale_x) + pad_x_val
                person_kp[:, 1] = (person_kp[:, 1] * scale_y) + pad_y_val
                # --- [좌표 변환 끝] ---
                
                valid_kps = person_kp[person_kp[:, 1] > 10]
                if len(valid_kps) == 0: continue
                min_x, max_x = np.min(valid_kps[:,0]), np.max(valid_kps[:,0])
                min_y, max_y = np.min(valid_kps[:,1]), np.max(valid_kps[:,1])
                dets.append([min_x, min_y, max_x, max_y, 1.0])
                keypoints_list.append((person_kp, person_conf))
                          
    dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
    
    # [★★★ 치명적 버그 수정 ★★★]
    dets_to_track = dets
    keypoints_list_to_track = keypoints_list
    # [★★★ 버그 수정 끝 ★★★]
            
    tracks = tracker.update(dets_to_track) 

    # --- ID별 로직 처리 ---
    active_track_ids = set()
    matched = set()
    
    base_data = game_state['base_data']
    track_id_to_player_id = game_state['track_id_to_player_id']
    floor_timers = game_state['floor_timers']
    floor_y_history = game_state['floor_y_history']
    final_scores = game_state['final_scores']
    kick_counters = game_state['kick_counters']
    l_kick_state = game_state['l_kick_state']
    r_kick_state = game_state['r_kick_state']
    l_reset_counter = game_state['l_reset_counter']
    r_reset_counter = game_state['r_reset_counter']
    person_kick_timer = game_state['person_kick_timer']

    JOINT_CONF_THRESH = game_state['JOINT_CONF_THRESH']
    CALIBRATION_TIME = game_state['CALIBRATION_TIME']
    STABILITY_THRESH = game_state['STABILITY_THRESH']

    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            active_track_ids.add(track_id)
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            best_idx, best_dist = -1, 1e9
            for idx, (person_kp_loop, _) in enumerate(keypoints_list_to_track):
                if idx in matched: continue 
                center = np.array([person_kp_loop[0][0], person_kp_loop[0][1]])
                dist = np.linalg.norm(bbox_center - center)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx == -1: continue
            matched.add(best_idx)

            person_xy, person_conf = keypoints_list_to_track[best_idx]
            head_x, head_y = person_xy[0][0], person_xy[0][1]
            
            foot_y = max(person_xy[15][1], person_xy[16][1])
            min_x, min_y, max_x, max_y = int(x1), int(y1), int(x2), int(y2)
            current_bbox_height, current_bbox_width = max_y - min_y, max_x - min_x
            head_conf = person_conf[0]
            is_head_visible = head_conf >= JOINT_CONF_THRESH
            major_joints_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
            is_full_body_visible = all(person_conf[idx] >= JOINT_CONF_THRESH for idx in major_joints_indices)

            # 5. 캘리브레이션 단계
            if track_id not in base_data:
                
                if is_game_mode:
                    continue

                l_ankle_xy, r_ankle_xy = person_xy[15][:2], person_xy[16][:2]
                is_in_any_box = False
                current_box_index = -1
                calibrated_box_indices = game_state['calibrated_box_indices']

                if calibration_boxes is not None:
                    for i, box in enumerate(calibration_boxes):
                        if is_point_in_box(l_ankle_xy, box) or is_point_in_box(r_ankle_xy, box):
                            if i in calibrated_box_indices: continue
                            is_in_any_box = True
                            current_box_index = i
                            game_state['active_calib_boxes'][i] = track_id
                            break
                
                base_height = foot_y - head_y
                
                is_standing_aspect_ratio = current_bbox_height > (current_bbox_width * 1.3) 
                
                l_shoulder_x, r_shoulder_x = person_xy[5][0], person_xy[6][0]
                l_hip_x, r_hip_x = person_xy[11][0], person_xy[12][0]
                shoulder_width = abs(l_shoulder_x - r_shoulder_x)
                hip_width = abs(l_hip_x - r_hip_x)
                FRONT_FACING_WIDTH_RATIO = 0.1 
                is_facing_front = (shoulder_width > base_height * FRONT_FACING_WIDTH_RATIO) and \
                                  (hip_width > base_height * (FRONT_FACING_WIDTH_RATIO - 0.02))
                
                l_hip_xy, r_hip_xy = person_xy[11][:2], person_xy[12][:2]
                l_ankle_xy, r_ankle_xy = person_xy[15][:2], person_xy[16][:2]
                l_leg_dist_2d = calculate_distance(l_hip_xy, l_ankle_xy)
                r_leg_dist_2d = calculate_distance(r_hip_xy, r_ankle_xy)
                l_leg_dist_y = abs(l_ankle_xy[1] - l_hip_xy[1])
                r_leg_dist_y = abs(r_ankle_xy[1] - r_hip_xy[1])
                LEG_STRAIGHTNESS_RATIO = 0.80
                is_l_leg_straight = (l_leg_dist_2d > 10) and ((l_leg_dist_y / l_leg_dist_2d) > LEG_STRAIGHTNESS_RATIO)
                is_r_leg_straight = (r_leg_dist_2d > 10) and ((r_leg_dist_y / r_leg_dist_2d) > LEG_STRAIGHTNESS_RATIO)
                is_standing = is_l_leg_straight or is_r_leg_straight
                
                print(f"--- ID {track_id} 캘리브레이션 조건 확인 ---")
                print(f"  1. 박스 안에 있나?: {is_in_any_box}")
                print(f"  2. 전신이 보이나?: {is_full_body_visible}")
                print(f"  3. 키가 적당한가?: {base_height > 100} (키: {base_height:.0f})")
                print(f"  4. 서 있는 자세?:  {is_standing}")
                print(f"  5. 정면을 보나?:    {is_facing_front}")
                print(f"  6. 비율이 맞나?:    {is_standing_aspect_ratio}")
                print("-------------------------------------")

                if is_in_any_box and is_full_body_visible and base_height > 100 and is_standing and is_facing_front and is_standing_aspect_ratio:
                    current_time = time.time()
                    if floor_timers[track_id] is None:
                        floor_timers[track_id] = current_time
                        floor_y_history[track_id] = [foot_y]
                    else:
                        floor_y_history[track_id].append(foot_y)
                        elapsed = current_time - floor_timers[track_id]
                        
                        if elapsed > CALIBRATION_TIME:
                            history = floor_y_history[track_id]
                            y_movement = np.max(history) - np.min(history)
                            
                            if y_movement < STABILITY_THRESH:
                                box_index_to_calibrate = current_box_index
                                if box_index_to_calibrate != -1:
                                    calibrated_box_indices.add(box_index_to_calibrate)

                                base_l_ankle_x, base_l_ankle_y = person_xy[15][0], person_xy[15][1]
                                base_r_ankle_x, base_r_ankle_y = person_xy[16][0], person_xy[16][1]
                                l_hip_xy = person_xy[11][:2]
                                r_hip_xy = person_xy[12][:2]
                                base_l_hip_ankle_dist = calculate_distance(l_hip_xy, (base_l_ankle_x, base_l_ankle_y))
                                base_r_hip_ankle_dist = calculate_distance(r_hip_xy, (base_r_ankle_x, base_r_ankle_y))

                                Y_MID = game_state['KICK_THRESH_PIXELS_Y']; X_MID = game_state['KICK_THRESH_PIXELS_X'];
                                Z_MID_EST_L = base_l_hip_ankle_dist * game_state['KICK_THRESH_RATIO_Z'] 
                                Z_MID_EST_R = base_r_hip_ankle_dist * game_state['KICK_THRESH_RATIO_Z']
                                Y_HIGH = Y_MID * 2.0; Z_HIGH_EST_L = Z_MID_EST_L * 1.5; Z_HIGH_EST_R = Z_MID_EST_R * 1.5 
                                Y_RST = Y_MID * 0.75; Z_RST_EST_L = Z_MID_EST_L * 0.4; Z_RST_EST_R = Z_MID_EST_R * 0.4

                                base_data[track_id] = {
                                    "base_height": base_height, "base_bbox_height": current_bbox_height, "base_bbox_width": current_bbox_width,
                                    "base_l_ankle_x": base_l_ankle_x, "base_l_ankle_y": base_l_ankle_y, "base_l_hip_ankle_dist": base_l_hip_ankle_dist,
                                    "base_r_ankle_x": base_r_ankle_x, "base_r_ankle_y": base_r_ankle_y, "base_r_hip_ankle_dist": base_r_hip_ankle_dist,
                                    "Y_MID": Y_MID, "X_MID": X_MID, "Z_MID_EST_L": Z_MID_EST_L, "Z_MID_EST_R": Z_MID_EST_R,
                                    "Y_HIGH": Y_HIGH, "Z_HIGH_EST_L": Z_HIGH_EST_L, "Z_HIGH_EST_R": Z_HIGH_EST_R,
                                    "Y_RST": Y_RST, "Z_RST_EST_L": Z_RST_EST_L, "Z_RST_EST_R": Z_RST_EST_R,
                                    "box_index": box_index_to_calibrate
                                }
                                
                                if track_id not in track_id_to_player_id:
                                    new_player_id = game_state['player_id_counter']
                                    track_id_to_player_id[track_id] = new_player_id
                                    game_state['player_id_counter'] += 1
                                    print(f"--- ID {track_id} 캘리브레이션 완료 (Player {new_player_id}) ---")
                                else:
                                    print(f"--- ID {track_id} (Player {track_id_to_player_id[track_id]}) 재캘리브레이션 완료 ---")
                                
                                kick_counters[track_id] = 0; l_kick_state[track_id] = 0; r_kick_state[track_id] = 0;
                                l_reset_counter[track_id] = 0; r_reset_counter[track_id] = 0; person_kick_timer[track_id] = 0;
                                
                                floor_timers[track_id] = None
                                floor_y_history[track_id] = []
                            
                            else:
                                print(f"--- ID {track_id} 캘리브레이션 실패: 2초간 Y축 흔들림 {y_movement:.1f}px (기준: {STABILITY_THRESH}px) ---")
                                floor_timers[track_id], floor_y_history[track_id] = None, []
                                if track_id in base_data: del base_data[track_id] 
                else:
                    if floor_timers[track_id] is not None:
                        reason = "박스 이탈" if not is_in_any_box else "자세 이탈"
                        print(f"--- ID {track_id} 캘리브레이션 중단: {reason} ---")
                    
                    floor_timers[track_id], floor_y_history[track_id] = None, []
                    if track_id in base_data: del base_data[track_id] 
            
            if is_head_visible and floor_timers[track_id] is not None and fonts:
                ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                elapsed = time.time() - floor_timers[track_id]
                angle = min((elapsed / CALIBRATION_TIME) * 360, 360)
                
                cv2.ellipse(letterboxed_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, (100,100,100), 2)
                cv2.ellipse(letterboxed_frame, (ui_center_x, ui_center_y), (radius, radius), 0, -90, int(angle-90), (0,255,255), 4)
                
                text_percent = f'{min(int(angle/3.6), 100)}%'
                
                if hasattr(temp_draw, 'textbbox'):
                    bbox = temp_draw.textbbox((0,0), text_percent, font=fonts['ui_percent'])
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    text_w, text_h = temp_draw.textsize(text_percent, font=fonts['ui_percent'])
                
                text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2) 
                pil_draw_list.append( (text_percent, text_pos, fonts['ui_percent'], (255,255,255), None) ) 

            # 6. 킥 카운트 단계
            elif is_head_visible and fonts and track_id in base_data:
                if "base_height" not in base_data[track_id]:
                    continue
                                
                pd = base_data[track_id] 
                
                is_too_close = (current_bbox_height > pd['base_bbox_height'] * 2.0) or (current_bbox_width > pd['base_bbox_width'] * 2.0) 

                if is_too_close:
                    player_id = track_id_to_player_id.get(track_id, 0)
                    if player_id != 0:
                        final_scores[player_id] = kick_counters[track_id]
                    
                    # --- [★★★ 버그 수정 ★★★] ---
                    if "box_index" in pd: 
                        box_idx = pd.get("box_index", -1)
                        if box_idx != -1 and box_idx in game_state['calibrated_box_indices']:
                            game_state['calibrated_box_indices'].remove(box_idx)
                            print(f"-> 박스 {box_idx} 반납됨 (너무 가까움).")
                    # --- [수정 끝] ---
                    
                    if track_id in base_data: del base_data[track_id]
                    print(f"ID {track_id} (Player {player_id}) 너무 가까움. 리셋.")
                
                else:
                    current_l_ankle_x, current_l_ankle_y = person_xy[15][0], person_xy[15][1]
                    current_r_ankle_x, current_r_ankle_y = person_xy[16][0], person_xy[16][1]
                    l_ankle_conf, r_ankle_conf = person_conf[15], person_conf[16]
                    l_hip_conf, r_hip_conf = person_conf[11], person_conf[12]
                    
                    is_l_ankle_visible = (l_ankle_conf >= JOINT_CONF_THRESH)
                    is_r_ankle_visible = (r_ankle_conf >= JOINT_CONF_THRESH)
                    is_l_hip_visible = (l_hip_conf >= JOINT_CONF_THRESH)
                    is_r_hip_visible = (r_hip_conf >= JOINT_CONF_THRESH)
                    
                    l_y_diff = pd['base_l_ankle_y'] - current_l_ankle_y
                    l_x_diff = abs(pd['base_l_ankle_x'] - current_l_ankle_x)
                    r_y_diff = pd['base_r_ankle_y'] - current_r_ankle_y
                    r_x_diff = abs(pd['base_r_ankle_x'] - current_r_ankle_x)
                    l_z_diff_est = 0.0
                    if is_l_hip_visible and is_l_ankle_visible and pd['base_l_hip_ankle_dist'] > 1:
                        current_l_hip_ankle_dist = calculate_distance(person_xy[11][:2], (current_l_ankle_x, current_l_ankle_y))
                        l_z_diff_est = abs(current_l_hip_ankle_dist - pd['base_l_hip_ankle_dist'])
                    r_z_diff_est = 0.0
                    if is_r_hip_visible and is_r_ankle_visible and pd['base_r_hip_ankle_dist'] > 1:
                        current_r_hip_ankle_dist = calculate_distance(person_xy[12][:2], (current_r_ankle_x, current_r_ankle_y))
                        r_z_diff_est = abs(current_r_hip_ankle_dist - pd['base_r_hip_ankle_dist'])
                    Y_MID, X_MID = pd['Y_MID'], pd['X_MID']
                    Y_HIGH = pd['Y_HIGH']; Y_RST = pd['Y_RST']
                    Z_MID_EST_L, Z_MID_EST_R = pd['Z_MID_EST_L'], pd['Z_MID_EST_R']
                    Z_HIGH_EST_L, Z_HIGH_EST_R = pd['Z_HIGH_EST_L'], pd['Z_HIGH_EST_R']
                    Z_RST_EST_L, Z_RST_EST_R = pd['Z_RST_EST_L'], pd['Z_RST_EST_R']
                    player_id = track_id_to_player_id.get(track_id, '?')
                    kick_detected_this_frame = False
                    if person_kick_timer[track_id] > 0: person_kick_timer[track_id] -= 1
                    
                    KICK_COOLDOWN_FRAMES = game_state['KICK_COOLDOWN_FRAMES']
                    RESET_FRAME_COUNT = game_state['RESET_FRAME_COUNT']

                    if person_kick_timer[track_id] == 0:
                        if l_kick_state[track_id] == 0:
                            is_vis = is_l_ankle_visible and is_l_hip_visible
                            if is_vis and l_y_diff > Y_MID: 
                                l_mid = (l_x_diff > X_MID and l_z_diff_est > Z_MID_EST_L)
                                l_high = (l_y_diff > Y_HIGH or l_z_diff_est > Z_HIGH_EST_L)
                                if l_mid or l_high:
                                    kick_counters[track_id] += 1; l_kick_state[track_id] = 1; l_reset_counter[track_id] = 0
                                    person_kick_timer[track_id] = KICK_COOLDOWN_FRAMES; kick_detected_this_frame = True 
                                    reason = "Mid" if l_mid else ("High(Y)" if l_y_diff > Y_HIGH else "High(Z)")
                                    print(f"=== Player {player_id} L-Kick!({reason})(Y:{l_y_diff:.1f}, X:{l_x_diff:.1f}, Z:{l_z_diff_est:.1f})(Tot:{kick_counters[track_id]}) ===")
                        elif l_kick_state[track_id] == 1:
                            l_base_cond = (is_l_ankle_visible and l_y_diff < Y_RST and l_z_diff_est < Z_RST_EST_L)
                            if l_base_cond: l_reset_counter[track_id] += 1
                            else: l_reset_counter[track_id] = 0 
                            if l_reset_counter[track_id] >= RESET_FRAME_COUNT: l_kick_state[track_id] = 0; print(f"Player {player_id} L-Kick RESET.")
                    
                    if not kick_detected_this_frame and person_kick_timer[track_id] == 0:
                        if r_kick_state[track_id] == 0:
                            is_vis = is_r_ankle_visible and is_r_hip_visible
                            if is_vis and r_y_diff > Y_MID:
                                r_mid = (r_x_diff > X_MID and r_z_diff_est > Z_MID_EST_R)
                                r_high = (r_y_diff > Y_HIGH or r_z_diff_est > Z_HIGH_EST_R)
                                if r_mid or r_high:
                                    kick_counters[track_id] += 1; r_kick_state[track_id] = 1; r_reset_counter[track_id] = 0
                                    person_kick_timer[track_id] = KICK_COOLDOWN_FRAMES 
                                    reason = "Mid" if r_mid else ("High(Y)" if r_y_diff > Y_HIGH else "High(Z)")
                                    print(f"=== Player {player_id} R-Kick! ({reason}) (Y:{r_y_diff:.1f}, X:{r_x_diff:.1f}, Z:{r_z_diff_est:.1f})(Tot:{kick_counters[track_id]}) ===")
                        elif r_kick_state[track_id] == 1:
                            r_base_cond = (is_r_ankle_visible and r_y_diff < Y_RST and r_z_diff_est < Z_RST_EST_R)
                            if r_base_cond: r_reset_counter[track_id] += 1
                            else: r_reset_counter[track_id] = 0
                            if r_reset_counter[track_id] >= RESET_FRAME_COUNT: r_kick_state[track_id] = 0; print(f"Player {player_id} R-Kick RESET.")
                    
                    ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                    player_id = track_id_to_player_id.get(track_id, "?")
                    text_count = f'{player_id}'
                    color_bgr = (0, 100, 255) if person_kick_timer[track_id] > 0 else (0, 255, 0) 
                    cv2.ellipse(letterboxed_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, color_bgr, -1) 
                    
                    if hasattr(temp_draw, 'textbbox'):
                        bbox = temp_draw.textbbox((0,0), text_count, font=fonts['ui_player'])
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        text_w, text_h = temp_draw.textsize(text_count, font=fonts['ui_player'])
                    text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2)
                    pil_draw_list.append( (text_count, text_pos, fonts['ui_player'], (0,0,0), None) ) 
                    
                    count_str = f"K: {kick_counters[track_id]}"
                    if hasattr(temp_draw, 'textbbox'):
                        bbox_kick = temp_draw.textbbox((0,0), count_str, font=fonts['ui_kick'])
                        text_w_k, text_h_k = bbox_kick[2] - bbox_kick[0], bbox_kick[3] - bbox_kick[1]
                    else:
                        text_w_k, text_h_k = temp_draw.textsize(count_str, font=fonts['ui_kick'])
                    pos_kick = (ui_center_x + radius + 5, ui_center_y + (radius//2) - (text_h_k//2) - 2)
                    pil_draw_list.append( (count_str, pos_kick, fonts['ui_kick'], (255, 255, 255), (0,0,0)) ) 
    
    # --- [★★★ 1초 유예 기간 로직 수정 ★★★] ---
    lost_id_timers = game_state['lost_id_timers']
    current_time = time.time()
    
    tracked_ids = set(base_data.keys()) | set(k for k,v in floor_timers.items() if v is not None)
    
    newly_lost_ids = tracked_ids - active_track_ids
    for lost_track_id in newly_lost_ids:
        if lost_track_id not in lost_id_timers:
            lost_id_timers[lost_track_id] = current_time
            player_id_str = track_id_to_player_id.get(lost_track_id, '?')
            print(f"ID {lost_track_id} (Player {player_id_str}) 추적 임시 손실. 1초 유예 시작...")

    GRACE_PERIOD = 1.0
    for track_id in list(lost_id_timers.keys()):
        if track_id in active_track_ids:
            del lost_id_timers[track_id]
            player_id_str = track_id_to_player_id.get(track_id, '?')
            print(f"ID {track_id} (Player {player_id_str}) 재추적 성공. 유예 취소.")
        
        elif (current_time - lost_id_timers[track_id]) > GRACE_PERIOD:
            player_id = track_id_to_player_id.get(track_id, 0)
            print(f"ID {track_id} (Player {player_id}) 1초간 미발견. 캘리브레이션 영구 삭제.")

            if track_id in base_data and "box_index" in base_data[track_id]:
                box_idx = base_data[track_id].get("box_index", -1)
                if box_idx != -1 and box_idx in game_state['calibrated_box_indices']:
                    game_state['calibrated_box_indices'].remove(box_idx)
                    print(f"-> 박스 {box_idx} 반납됨.")
            
            if track_id in base_data:
                if player_id != 0:
                    final_scores[player_id] = kick_counters[track_id]
                    print(f"-> 최종 점수 {kick_counters[track_id]}점 저장.")
            
            for d in [base_data, kick_counters, floor_timers, floor_y_history, track_id_to_player_id, 
                        l_kick_state, r_kick_state, l_reset_counter, r_reset_counter, person_kick_timer]:
                if track_id in d: del d[track_id]
            
            del lost_id_timers[track_id]

    return pil_draw_list


# === 메인 프로그램 ===
def main():
    # --- 1. 초기 설정 ---
    
    model_path = resource_path(os.path.join('image', 'yolov8n-pose.pt'))
    
    try:
        model = YOLO(model_path)
        print("모델 로드 성공.")
    except Exception as e:
        print(f"YOLOv8 모델 로드 오류: {e}\n경로: {model_path}")
        return 
        
    tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.3)
    
    current_state = MENU_START 
    max_players = 0 

    game_state = create_game_state()
    
    show_debug_ui = False

    game_mode_start_time = 0.0
    game_mode_stage = 0
    GAME_DURATION_SECONDS = 30.0
    
    # --- 2. 카메라/화면 설정 ---
    try:
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except Exception:
        screen_width, screen_height = 1920, 1080
        
    print("모니터 해상도:", screen_width, screen_height)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열 수 없음.")
        return 

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 실제 해상도: {frame_width} x {frame_height}")

    TARGET_W, TARGET_H = 1920, 1080
    
    mouse_scale_x, mouse_scale_y = 1.0, 1.0
    mouse_pad_x, mouse_pad_y = 0, 0
    
    mouse_clicked = False
    mouse_pos = (0, 0)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_clicked, mouse_pos
        
        scaled_x = int((x - mouse_pad_x) / mouse_scale_x)
        scaled_y = int((y - mouse_pad_y) / mouse_scale_y)
        mouse_pos = (scaled_x, scaled_y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True

    WIN_NAME = 'Kick Counter - Multi Person Tracking'
    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)

    # --- [수정] 폰트 로드 ---
    try:
        FONT_PATH = resource_path(os.path.join('image', 'malgun.ttf'))
        fonts = {
            'title': ImageFont.truetype(FONT_PATH, 100),
            'subtitle': ImageFont.truetype(FONT_PATH, 40),
            'menu': ImageFont.truetype(FONT_PATH, 50),
            'ui_main': ImageFont.truetype(FONT_PATH, 24),
            'ui_kick': ImageFont.truetype(FONT_PATH, 20),
            'ui_player': ImageFont.truetype(FONT_PATH, 22),
            'ui_percent': ImageFont.truetype(FONT_PATH, 20),
            'game_button': ImageFont.truetype(FONT_PATH, 30)
        }
    except IOError:
        print(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
        fonts = None

    splash_image_resized = None
    player_select_image_resized = None
    countdown_imgs = {}
    game_instructions_img = None
    instructions_normal_img = None 
    timer_bg_img = None
    timer_fg_img = None
    
    try:
        splash_image_path = resource_path(os.path.join('image', 'splash.png'))
        splash_image = cv2.imread(splash_image_path)
        if splash_image is None: raise FileNotFoundError("image/splash.png")
        splash_image_resized = cv2.resize(splash_image, (TARGET_W, TARGET_H))
        print("스플래시 이미지 로드 성공.")
        
        player_select_image_path = resource_path(os.path.join('image', 'player_select.png'))
        player_select_image = cv2.imread(player_select_image_path)
        if player_select_image is None: raise FileNotFoundError("image/player_select.png")
        player_select_image_resized = cv2.resize(player_select_image, (TARGET_W, TARGET_H))
        print("인원 선택 이미지 로드 성공.")
        
        for i in [1, 2, 3]:
            img_path = resource_path(os.path.join('image', f'countdown_{i}.png'))
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: raise FileNotFoundError(f"image/countdown_{i}.png")
            countdown_imgs[i] = img
        print("카운트다운 이미지 로드 성공.")

        img_path = resource_path(os.path.join('image', 'game_instructions.png'))
        game_instructions_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if game_instructions_img is None: raise FileNotFoundError("image/game_instructions.png")
        print("게임 설명 이미지 로드 성공.")

        img_path = resource_path(os.path.join('image', 'timer_bg.png'))
        timer_bg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if timer_bg_img is None: raise FileNotFoundError("image/timer_bg.png")
        
        img_path = resource_path(os.path.join('image', 'timer_fg.png'))
        timer_fg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if timer_fg_img is None: raise FileNotFoundError("image/timer_fg.png")
        print("타이머 바 이미지 로드 성공.")
        
        img_path = resource_path(os.path.join('image', 'instructions_normal.png'))
        instructions_normal_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if instructions_normal_img is None: raise FileNotFoundError("image/instructions_normal.png")
        print("일반 모드 설명 이미지 로드 성공.")

    except Exception as e:
        print(f"경고: 필수 이미지 로드 실패. {e}")
        if splash_image_resized is None:
            splash_image_resized = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        if player_select_image_resized is None:
            player_select_image_resized = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)


    # === 3. 메인 루프 (프로그램 실행) ===
    tracks = [] 
    active_hover_zone = -1 
    
    has_shown_normal_instructions = False
    
    # --- 버튼 영역 상수 ---
    radius = 35 
    margin = 30
    center_x_esc = TARGET_W - radius - margin
    center_y_esc = radius + margin
    ESC_BUTTON_ZONE = (center_x_esc - radius, center_y_esc - radius, center_x_esc + radius, center_y_esc + radius)
    
    center_x_v = radius + margin
    center_y_v = radius + margin
    V_BUTTON_ZONE = (center_x_v - radius, center_y_v - radius, center_x_v + radius, center_y_v + radius)
    
    center_x_game = TARGET_W - radius - margin
    center_y_game = center_y_esc + radius + radius + 10 # ESC 버튼 아래
    GAME_BUTTON_ZONE = (center_x_game - radius, center_y_game - radius, center_x_game + radius, center_y_game + radius)
    
    # [★★★ 리셋 버튼 추가 1/6 ★★★]
    center_x_reset = TARGET_W - radius - margin # X좌표 동일
    center_y_reset = center_y_game + radius + radius + 10 # Y좌표는 GAME 버튼 아래
    RESET_BUTTON_ZONE_UI = (center_x_reset - radius, center_y_reset - radius, center_x_reset + radius, center_y_reset + radius)

    btn_w, btn_h = 300, 80
    btn_y = TARGET_H // 2 + 200
    btn_restart_x1 = (TARGET_W // 2) - btn_w - 20
    btn_restart_y1 = btn_y
    RESTART_BUTTON_ZONE = (btn_restart_x1, btn_restart_y1, btn_restart_x1 + btn_w, btn_restart_y1 + btn_h)
    
    btn_normal_x1 = (TARGET_W // 2) + 20
    btn_normal_y1 = btn_y
    NORMAL_MODE_BUTTON_ZONE = (btn_normal_x1, btn_normal_y1, btn_normal_x1 + btn_w, btn_normal_y1 + btn_h)

    temp_pil_img = Image.new("RGB", (1,1))
    temp_draw = ImageDraw.Draw(temp_pil_img)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        letterboxed_frame, (ratio_x, ratio_y), (pad_w, pad_h) = letterbox(frame, new_shape=(TARGET_H, TARGET_W))
        
        mouse_scale_x, mouse_scale_y = ratio_x, ratio_y
        mouse_pad_x, mouse_pad_y = pad_w, pad_h
        
        display_frame = letterboxed_frame.copy()

        key = cv2.waitKeyEx(1)
        
        # -----------------------------------------------------
        # A. UI 및 상태 전환 로직
        # -----------------------------------------------------

        if current_state == MENU_START:
            display_frame, _ = draw_menu_ui(MENU_START, 
                                            frame_dims=(TARGET_W, TARGET_H), 
                                            splash_img=splash_image_resized)
            
            if key == ord('q'):
                break 
            
            elif (key != -1 and key != 255) or mouse_clicked:
                current_state = MENU_PLAYER_SELECT
                mouse_clicked = False 
        
        elif current_state == MENU_PLAYER_SELECT:
            display_frame, active_hover_zone = draw_menu_ui(MENU_PLAYER_SELECT, 
                                                            frame_dims=(TARGET_W, TARGET_H), 
                                                            fonts=fonts, 
                                                            mouse_pos=mouse_pos, 
                                                            player_select_img=player_select_image_resized)
            
            has_shown_normal_instructions = False
            
            if key == ord('q'):
                break 

            elif key == ord('1'): max_players = 1; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드 (키보드)")
            elif key == ord('2'): max_players = 2; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드 (키보드)")
            elif key == ord('3'): max_players = 3; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드 (키보드)")
            elif key == ord('4'): max_players = 4; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드 (키보드)")
            
            if mouse_clicked:
                if active_hover_zone != -1: 
                    max_players = active_hover_zone + 1 
                    current_state = GAME_RUNNING
                    print(f"게임 시작: {max_players}인 모드 (마우스 클릭)")
                
                mouse_clicked = False 
        
        # -----------------------------------------------------
        # B. 게임 실행 로직
        # -----------------------------------------------------

        elif current_state == GAME_RUNNING:
            
            # --- 키보드/마우스 입력 처리 (상태 변경) ---
            if key == ord('q'): break
            
            # [★★★ V/ESC/R 수정 1/3 ★★★]
            # --- 모든 트리거를 'if/elif' 체인 *시작 전*에 계산 ---
            esc_triggered = (key == 27) or \
                            (mouse_clicked and is_point_in_box(mouse_pos, ESC_BUTTON_ZONE))
            
            game_triggered = mouse_clicked and is_point_in_box(mouse_pos, GAME_BUTTON_ZONE)

            reset_triggered = (key == ord('r')) or \
                              (mouse_clicked and is_point_in_box(mouse_pos, RESET_BUTTON_ZONE_UI))
            
            v_triggered = (key == ord('v')) or \
                          (mouse_clicked and is_point_in_box(mouse_pos, V_BUTTON_ZONE))
            # --- 트리거 계산 끝 ---


            # [★★★ V/ESC/R 수정 2/3 ★★★]
            # --- 'if/elif' 체인 시작 ---
            
            # 1. ESC (종료)
            if esc_triggered:
                current_state = MENU_PLAYER_SELECT 
                game_state = create_game_state() 
                print("인원 선택 화면으로 복귀. 모든 데이터 초기화.")
                mouse_clicked = False 
                continue 
            
            # 2. GAME (게임 모드 시작)
            elif game_triggered:
                current_state = GAME_COUNTDOWN
                game_mode_start_time = 0.0 
                game_mode_stage = 0 
                game_state['kick_counters'].clear() 
                game_state['final_scores'].clear() 
                mouse_clicked = False
                print("게임 모드 진입. 스페이스바 대기 중...")
                continue 

            # 3. RESET (일반 모드 리셋)
            # [★★★ 리셋 버튼 추가 3/6 ~ 6/6 ★★★]
            elif reset_triggered:
                print("--- [RESET] --- 인원 수 유지, 모든 상태 초기화.")
                game_state = create_game_state() 
                tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.3) 
                mouse_clicked = False 
                continue 
            
            # --- 키보드 입력 처리 (값 변경) ---
            elif key == 2490368: game_state['KICK_THRESH_PIXELS_Y'] += 1
            elif key == 2621440: game_state['KICK_THRESH_PIXELS_Y'] = max(1, game_state['KICK_THRESH_PIXELS_Y'] - 1)
            elif key == 2424832: game_state['KICK_THRESH_PIXELS_X'] = max(1, game_state['KICK_THRESH_PIXELS_X'] - 1)
            elif key == 2555904: game_state['KICK_THRESH_PIXELS_X'] += 1
            elif key == 2162688: game_state['KICK_THRESH_RATIO_Z'] += 0.01
            elif key == 2228224: game_state['KICK_THRESH_RATIO_Z'] = max(0.01, game_state['KICK_THRESH_RATIO_Z'] - 0.01)
            elif key == 2359296: game_state['KICK_COOLDOWN_FRAMES'] += 1
            elif key == 2293760: game_state['KICK_COOLDOWN_FRAMES'] = max(1, game_state['KICK_COOLDOWN_FRAMES'] - 1)
            elif key == ord('8'): game_state['CALIB_BOX_BOTTOM_MARGIN'] += 5
            elif key == ord('2'): game_state['CALIB_BOX_BOTTOM_MARGIN'] = max(5, game_state['CALIB_BOX_BOTTOM_MARGIN'] - 5)
            
            # [★★★ V/ESC/R 수정 3/3 ★★★]
            # 4. V (디버그 UI 토글)
            elif v_triggered: 
                show_debug_ui = not show_debug_ui
                if mouse_clicked: # 마우스로 클릭했으면
                    mouse_clicked = False # 클릭 이벤트 소모


            # [★★★ 수정] 캘리브레이션 박스 계산
            calibration_boxes = get_calibration_boxes(max_players, TARGET_W, TARGET_H, game_state['CALIB_BOX_BOTTOM_MARGIN'])
            game_state['active_calib_boxes'].clear() 

            pil_draw_list = process_frame_logic(
                frame, display_frame, ratio_x, ratio_y, pad_w, pad_h,
                model, tracker, max_players, game_state, fonts, temp_draw,
                calibration_boxes=calibration_boxes,
                is_game_mode=False
            )

            # --- [★★★ 추가] 캘리브레이션 박스 UI 그리기 ---
            if fonts:
                for i, box in enumerate(calibration_boxes):
                    x1, y1, x2, y2 = box
                    
                    is_being_calibrated = (i in game_state['active_calib_boxes'])
                    is_calibrated = (i in game_state['calibrated_box_indices'])

                    if is_being_calibrated:
                        color = (0, 255, 0) # Green
                    elif is_calibrated:
                        color = (0, 0, 255) # Red
                    else:
                        color = (255, 0, 0) # Blue
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if not is_being_calibrated and not is_calibrated:
                        text = "여기 서주세요"
                        font = fonts['subtitle']
                        
                        text_w, text_h = 0, 0
                        if hasattr(temp_draw, 'textbbox'):
                            bbox = temp_draw.textbbox((0,0), text, font=font)
                            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        else:
                            text_w, text_h = temp_draw.textsize(text, font=font)
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        text_x = x1 + (box_width - text_w) // 2
                        text_y = y1 + (box_height - text_h) // 2
                        
                        pil_draw_list.append( (text, (text_x, text_y), font, (255, 0, 0), None) )

            num_calibrated = len(game_state['calibrated_box_indices'])
            
            if (not has_shown_normal_instructions or num_calibrated < max_players) and instructions_normal_img is not None:
                h, w = instructions_normal_img.shape[:2]
                x_pos = (TARGET_W - w) // 2
                y_pos = 50
                
                cv2.rectangle(display_frame, (x_pos - 5, y_pos - 5), (x_pos + w + 5, y_pos + h + 5), (0,0,0), -1)
                
                overlay_transparent(display_frame, instructions_normal_img, x_pos, y_pos)
                
                if num_calibrated > 0:
                        has_shown_normal_instructions = True


            # --- 9. 공통 UI 그리기 (텍스트) ---
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                dark_blue_rgb = (0, 0, 205)
                white_rgb = (255, 255, 255) 
                
                # ESC UI
                esc_text = "ESC"
                esc_font = fonts['ui_main'] 
                radius = 35 
                margin = 30
                center_x = TARGET_W - radius - margin
                center_y = radius + margin
                draw_final.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox = draw_final.textbbox((0,0), esc_text, font=esc_font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    text_w, text_h = draw_final.textsize(esc_text, font=esc_font)
                text_pos = (center_x - text_w // 2, center_y - text_h // 2 - 2)
                draw_final.text(text_pos, esc_text, font=esc_font, fill=white_rgb)

                # "GAME" 버튼
                game_text = "GAME"
                game_font = fonts['ui_player']
                radius_g = 35
                center_x_g = TARGET_W - radius_g - margin
                center_y_g = center_y + radius + radius_g + 10
                draw_final.ellipse([(center_x_g - radius_g, center_y_g - radius_g), 
                                    (center_x_g + radius_g, center_y_g + radius_g)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox_g = draw_final.textbbox((0,0), game_text, font=game_font)
                    text_w_g, text_h_g = bbox_g[2] - bbox_g[0], bbox_g[3] - bbox_g[1]
                else:
                    text_w_g, text_h_g = draw_final.textsize(game_text, font=game_font)
                text_pos_g = (center_x_g - text_w_g // 2, center_y_g - text_h_g // 2 - 2)
                draw_final.text(text_pos_g, game_text, font=game_font, fill=white_rgb)

                # [★★★ 리셋 버튼 추가 2/6 ★★★]
                # "RESET" 버튼 그리기
                reset_text = "RESET" # [수정] "R" -> "RESET"
                reset_font = fonts['ui_kick'] # [수정] ui_player(22pt) -> ui_kick(20pt)
                radius_r = 35
                center_x_r = center_x_reset 
                center_y_r = center_y_reset
                draw_final.ellipse([(center_x_r - radius_r, center_y_r - radius_r), 
                                    (center_x_r + radius_r, center_y_r + radius_r)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox_r = draw_final.textbbox((0,0), reset_text, font=reset_font)
                    text_w_r, text_h_r = bbox_r[2] - bbox_r[0], bbox_r[3] - bbox_r[1]
                else:
                    text_w_r, text_h_r = draw_final.textsize(reset_text, font=reset_font)
                text_pos_r = (center_x_r - text_w_r // 2, center_y_r - text_h_r // 2 - 2)
                draw_final.text(text_pos_r, reset_text, font=reset_font, fill=white_rgb)

                # V-Key UI 토글
                if show_debug_ui:
                    draw_pil_text_on_image(draw_final, f"Ankle Height (Y): {game_state['KICK_THRESH_PIXELS_Y']}px (Up/Down)", (10, 30), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Ankle Dist (X): {game_state['KICK_THRESH_PIXELS_X']}px (Left/Right)", (10, 60), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Z-Est Ratio: {game_state['KICK_THRESH_RATIO_Z']*100:.0f}% (PgUp/PgDn)", (10, 90), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Cooldown: {game_state['KICK_COOLDOWN_FRAMES']}f (Home/End)", (10, 120), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Box Margin (Y): {game_state['CALIB_BOX_BOTTOM_MARGIN']}px (8/2 keys)", (10, 150), fonts['ui_main'], (255, 0, 0), (0,0,0))
                else:
                    v_text = "V"
                    v_font = fonts['ui_main'] # 24pt 폰트
                    radius_v = 35
                    margin_v = 30
                    center_x_v = radius_v + margin_v
                    center_y_v = radius_v + margin_v
                    draw_final.ellipse([(center_x_v - radius_v, center_y_v - radius_v), 
                                        (center_x_v + radius_v, center_y_v + radius_v)], fill=dark_blue_rgb)
                    
                    if hasattr(draw_final, 'textbbox'):
                        bbox_v = draw_final.textbbox((0,0), v_text, font=v_font)
                        text_w_v, text_h_v = bbox_v[2] - bbox_v[0], bbox_v[3] - bbox_v[1]
                    else:
                        text_w_v, text_h_v = draw_final.textsize(v_text, font=v_font)
                    text_pos_v = (center_x_v - text_w_v // 2, center_y_v - text_h_v // 2 - 2)
                    draw_final.text(text_pos_v, v_text, font=v_font, fill=white_rgb)

                
                # 저장된 텍스트 그리기
                for (text, pos, font, txt_col, bg_col, *align) in pil_draw_list:
                    alignment = align[0] if align else "left"
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col, align=alignment)
                
                # 최종 점수판
                final_scores = game_state['final_scores']
                if len(final_scores) > 0:
                    x_pos = TARGET_W - 250 
                    score_text = "== Final Scores =="
                    
                    if hasattr(draw_final, 'textbbox'):
                        bbox_score = draw_final.textbbox((0,0), score_text, font=fonts['ui_kick'])
                        text_h_score = bbox_score[3] - bbox_score[1]
                    else:
                        _, text_h_score = draw_final.textsize(score_text, font=fonts['ui_kick'])
                    
                    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

                    total_scores_height = text_h_score + (30 * len(sorted_scores))
                    current_y_top = (TARGET_H // 2) - (total_scores_height // 2)
                    
                    draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))
                    
                    for id_num, count in sorted_scores:
                        current_y_top += 30 
                        text = f"Player {id_num} : {count}"
                        draw_pil_text_on_image(draw_final, text, (x_pos + 20, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))

                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
            
        # -----------------------------------------------------
        # C. 게임 카운트다운 상태
        # -----------------------------------------------------
        elif current_state == GAME_COUNTDOWN:
            display_frame = letterboxed_frame.copy()
            
            if game_mode_stage == 0:
                if fonts:
                    img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    draw_final = ImageDraw.Draw(img_pil_final)
                    text = "스페이스바를 눌러 게임을 시작하세요"
                    draw_pil_text_on_image(draw_final, text, (TARGET_W // 2, 50), fonts['subtitle'], (255, 255, 0), (0,0,0), align="center")
                    display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
                
                if key == ord(' '): # 스페이스바
                    game_mode_stage = 1
                    game_mode_start_time = time.time()
                    print("게임 카운트다운 시작...")
            
            else:
                elapsed = time.time() - game_mode_start_time
                
                if game_instructions_img is not None:
                    h, w = game_instructions_img.shape[:2]
                    x_pos = (TARGET_W - w) // 2
                    y_pos = 50
                    
                    cv2.rectangle(display_frame, (x_pos - 5, y_pos - 5), (x_pos + w + 5, y_pos + h + 5), (0,0,0), -1)
                    
                    overlay_transparent(display_frame, game_instructions_img, x_pos, y_pos)

                number_to_show = 0
                if elapsed < 1.0:
                    number_to_show = 3
                elif elapsed < 2.0:
                    number_to_show = 2
                elif elapsed < 3.0:
                    number_to_show = 1
                elif elapsed >= 3.0:
                    current_state = GAME_TIMER_RUNNING
                    game_mode_start_time = time.time()
                    game_state['kick_counters'].clear()
                    game_state['final_scores'].clear() 
                    print("게임 시작!")
                    continue
                
                if number_to_show > 0 and countdown_imgs.get(number_to_show) is not None:
                    img_num = countdown_imgs[number_to_show]
                    h, w = img_num.shape[:2]
                    x_pos = (TARGET_W - w) // 2
                    y_pos = (TARGET_H - h) // 2
                    overlay_transparent(display_frame, img_num, x_pos, y_pos)

            if key == ord('q'): break
            if key == 27:
                current_state = MENU_PLAYER_SELECT
                game_state = create_game_state()
                print("인원 선택 화면으로 복귀.")
                continue
                
        # -----------------------------------------------------
        # D. 게임 타이머 실행 상태
        # -----------------------------------------------------
        elif current_state == GAME_TIMER_RUNNING:
            
            if key == ord('q'): break
            if key == 27:
                current_state = MENU_PLAYER_SELECT
                game_state = create_game_state()
                print("인원 선택 화면으로 복귀.")
                continue

            pil_draw_list = process_frame_logic(
                frame, display_frame, ratio_x, ratio_y, pad_w, pad_h,
                model, tracker, max_players, game_state, fonts, temp_draw,
                calibration_boxes=None,
                is_game_mode=True
            )

            # --- 2. 타이머 UI 그리기 ---
            elapsed = time.time() - game_mode_start_time
            time_left = GAME_DURATION_SECONDS - elapsed
            progress = elapsed / GAME_DURATION_SECONDS

            if time_left <= 0:
                print("게임 종료!")
                game_state['final_scores'].clear()
                for track_id, player_id in game_state['track_id_to_player_id'].items():
                    if track_id in game_state['kick_counters']:
                        game_state['final_scores'][player_id] = game_state['kick_counters'][track_id]
                
                current_state = GAME_OVER 
                mouse_clicked = False 
                continue

            if timer_bg_img is not None and timer_fg_img is not None:
                h_bg, w_bg = timer_bg_img.shape[:2]
                h_fg, w_fg = timer_fg_img.shape[:2]
                
                x_pos = (TARGET_W - w_fg) // 2
                y_pos = 50
                overlay_transparent(display_frame, timer_bg_img, x_pos, y_pos)

                clock_width = h_bg
                total_bar_width = w_fg - clock_width 
                
                # [★★★ 타이머 로직 수정 (줄어들기) ★★★]
                # progress는 (0.0 -> 1.0)으로 증가함
                # (1.0 - progress)는 (1.0 -> 0.0)으로 감소함
                remaining_width = int(total_bar_width * (1.0 - progress))
                # [★★★ 수정 끝 ★★★]

                if remaining_width > 0:
                    # [수정] remaining_width 사용
                    bar_crop = timer_fg_img[:, clock_width : clock_width + remaining_width]
                    
                    overlay_transparent(display_frame, bar_crop, x_pos + clock_width, y_pos)


            # --- 3. PIL 텍스트 그리기 (카운터) ---
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)
                
                for (text, pos, font, txt_col, bg_col, *align) in pil_draw_list:
                    alignment = align[0] if align else "left"
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col, align=alignment)
                
                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)

        # -----------------------------------------------------
        # E. 게임 종료 (결과) 상태
        # -----------------------------------------------------
        elif current_state == GAME_OVER:
            display_frame = letterboxed_frame.copy()
            
            if key == ord('q'): break
            if key == 27: # ESC
                current_state = MENU_PLAYER_SELECT
                game_state = create_game_state()
                print("인원 선택 화면으로 복귀.")
                continue
            
            if mouse_clicked:
                if is_point_in_box(mouse_pos, RESTART_BUTTON_ZONE):
                    current_state = GAME_COUNTDOWN
                    game_mode_start_time = 0.0
                    game_mode_stage = 0
                    game_state['kick_counters'].clear()
                    game_state['final_scores'].clear() 
                    mouse_clicked = False
                    print("게임 모드 재시작. 스페이스바 대기 중...")
                    continue
                elif is_point_in_box(mouse_pos, NORMAL_MODE_BUTTON_ZONE):
                    current_state = GAME_RUNNING
                    game_state['kick_counters'].clear()
                    game_state['final_scores'].clear() 
                    mouse_clicked = False
                    print("일반 모드로 복귀.")
                    continue
            
            mouse_clicked = False

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (TARGET_W, TARGET_H), (0,0,0), -1)
            display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
            
            cv2.rectangle(display_frame, (RESTART_BUTTON_ZONE[0], RESTART_BUTTON_ZONE[1]), (RESTART_BUTTON_ZONE[2], RESTART_BUTTON_ZONE[3]), (0, 200, 0), -1)
            cv2.rectangle(display_frame, (NORMAL_MODE_BUTTON_ZONE[0], NORMAL_MODE_BUTTON_ZONE[1]), (NORMAL_MODE_BUTTON_ZONE[2], NORMAL_MODE_BUTTON_ZONE[3]), (205, 0, 0), -1)
            
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                white_rgb = (255, 255, 255)
                yellow_rgb = (0, 255, 255)
                
                x_pos = TARGET_W // 2
                score_text = "== Final Scores =="
                font_title = fonts['subtitle']
                font_score = fonts['ui_main']
                
                if hasattr(draw_final, 'textbbox'):
                    bbox_score = draw_final.textbbox((0,0), score_text, font=font_title)
                    text_h_score = bbox_score[3] - bbox_score[1]
                else:
                    _, text_h_score = draw_final.textsize(score_text, font=font_title)
                
                final_scores = game_state['final_scores']
                sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
                
                total_scores_height = text_h_score + (40 * len(sorted_scores))
                current_y_top = (TARGET_H // 2) - (total_scores_height // 2) - 50
                
                draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), font_title, yellow_rgb, (0,0,0), align="center")
                
                current_y_top += 60
                
                for rank, (id_num, count) in enumerate(sorted_scores):
                    text = f"{rank + 1}위 - Player {id_num} : {count} 회"
                    draw_pil_text_on_image(draw_final, text, (x_pos, current_y_top), font_score, white_rgb, None, align="center")
                    current_y_top += 40 
                
                font_btn = fonts['game_button']
                btn_text_restart = "RESTART"
                btn_text_normal = "NORMAL MODE"
                
                if hasattr(draw_final, 'textbbox'):
                    bbox_r = draw_final.textbbox((0,0), btn_text_restart, font=font_btn)
                    w_r, h_r = bbox_r[2] - bbox_r[0], bbox_r[3] - bbox_r[1]
                    bbox_n = draw_final.textbbox((0,0), btn_text_normal, font=font_btn)
                    w_n, h_n = bbox_n[2] - bbox_n[0], bbox_n[3] - bbox_n[1]
                else:
                    w_r, h_r = draw_final.textsize(btn_text_restart, font=font_btn)
                    w_n, h_n = draw_final.textsize(btn_text_normal, font=font_btn)

                pos_r_x = RESTART_BUTTON_ZONE[0] + (btn_w - w_r) // 2
                pos_r_y = RESTART_BUTTON_ZONE[1] + (btn_h - h_r) // 2
                draw_pil_text_on_image(draw_final, btn_text_restart, (pos_r_x, pos_r_y), font_btn, white_rgb, None)
                
                pos_n_x = NORMAL_MODE_BUTTON_ZONE[0] + (btn_w - w_n) // 2
                pos_n_y = NORMAL_MODE_BUTTON_ZONE[1] + (btn_h - h_n) // 2
                draw_pil_text_on_image(draw_final, btn_text_normal, (pos_n_x, pos_n_y), font_btn, white_rgb, None)

                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)


        # --- 최종 화면 표시 ---
        final_display = cv2.resize(display_frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN_NAME, final_display)

    # === 메인 루프 끝 ===
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()