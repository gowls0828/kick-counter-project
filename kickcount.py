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

try:
    if getattr(sys, 'frozen', False):
        # 1. PyInstaller 환경 (exe) 감지
        base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.executable)
        sort_path = os.path.join(base_path, 'sort')
        
        # 2. sort 폴더 경로를 sys.path에 추가하여 임포트 가능하게 함
        if sort_path not in sys.path:
            sys.path.insert(0, sort_path)
    else:
        # 3. 개발 환경 (.py)에서는 기존 방식 유지
        sys.path.append('./sort') 
        
    from sort import Sort 
    
except ImportError as e:
    # 최종 오류 발생 시 메시지 출력
    print("="*60); 
    print(f"🚨 치명적 오류: 'sort' 모듈을 찾을 수 없습니다."); 
    print(f"ImportError: {e}");
    print("PyInstaller 빌드 시 '--add-data \"sort;sort\"' 옵션을 사용했는지 확인하세요.");
    print("="*60);
    sys.exit(1)

# === 두 점 사이의 거리 계산 함수 ===
def calculate_distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

# 배경 있는 텍스트 그리기 함수
def draw_text_with_bg(img, text, pos, scale, thickness, txt_col, bg_col):
    try: 
        (w,h),bl = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
        tl=(max(0,pos[0]),max(0,pos[1]-h-bl))
        br=(min(img.shape[1]-1,pos[0]+w),min(img.shape[0]-1,pos[1]+bl))
        cv2.rectangle(img,tl,br,bg_col,-1)
        cv2.putText(img,text,(pos[0],pos[1]),cv2.FONT_HERSHEY_SIMPLEX,scale,txt_col,thickness,cv2.LINE_AA)
    except Exception as e: 
        pass

# === 메인 프로그램 시작 ===
def main():
    # --- 1. 초기 설정 ---
    model = YOLO('yolov8n-pose.pt')
    tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.3)

    base_data = {}
    final_scores = {}
    kick_counters = defaultdict(int)
    
    # 3중 상태 변수
    l_kick_state = defaultdict(int)
    r_kick_state = defaultdict(int)
    l_reset_counter = defaultdict(int)
    r_reset_counter = defaultdict(int)
    person_kick_timer = defaultdict(int) 
    
    RESET_FRAME_COUNT = 3  # 리셋에 필요한 연속 프레임
    
    # ▼▼▼ (수정) 쿨다운 기본값 설정 (이제 이 변수가 키로 조작됨) ▼▼▼
    KICK_COOLDOWN_FRAMES = 5 
    # ▲▲▲ ▲▲▲ ▲▲▲
    
    player_id_counter = 1
    track_id_to_player_id = {} 

    floor_timers = defaultdict(lambda: None)
    floor_y_history = defaultdict(list)
    CALIBRATION_TIME = 2.0
    STABILITY_THRESH = 20

    KICK_THRESH_PIXELS_Y = 30 
    KICK_THRESH_PIXELS_X = 20 
    KICK_THRESH_RATIO_Z = 0.10 # 10%

    JOINT_CONF_THRESH = 0.1 # 10%

    # --- 2. 카메라 및 화면 설정 ---
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print("모니터 해상도:", screen_width, screen_height)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"요청 해상도: 1920x1080, 카메라 실제 해상도: {frame_width} x {frame_height}")

    cv2.namedWindow('Kick Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Kick Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0

    # === 3. 메인 루프 (프로그램 실행) ===
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        display_frame = frame 

        # --- 화면 그리기 ---
        threshold_text_y = f"Ankle Height (Y): {KICK_THRESH_PIXELS_Y}px (Up/Down)"
        draw_text_with_bg(display_frame, threshold_text_y, (10, 30), 0.8, 2, (255, 0, 0), (0,0,0))
        threshold_text_x = f"Ankle Dist (X): {KICK_THRESH_PIXELS_X}px (Left/Right)"
        draw_text_with_bg(display_frame, threshold_text_x, (10, 60), 0.8, 2, (255, 0, 0), (0,0,0))
        threshold_text_z = f"Z-Est Ratio: {KICK_THRESH_RATIO_Z*100:.0f}% (PgUp/PgDn)"
        draw_text_with_bg(display_frame, threshold_text_z, (10, 90), 0.8, 2, (255, 0, 0), (0,0,0))
        
        # ▼▼▼ (수정) 쿨다운 프레임 UI 표시 ▼▼▼
        cooldown_text = f"Cooldown: {KICK_COOLDOWN_FRAMES}f (Home/End)"
        draw_text_with_bg(display_frame, cooldown_text, (10, 120), 0.8, 2, (255, 0, 0), (0,0,0))
        # ▲▲▲ ▲▲▲ ▲▲▲

        # --- 사람 감지 및 ID 추적 ---
        results = model(frame, conf=0.6, verbose=False)
        dets, keypoints_list = [], []
        for r in results:
            if r.keypoints is not None:
                xy_data, conf_data = getattr(r.keypoints, "xy", []), getattr(r.keypoints, "conf", [])
                if len(xy_data) != len(conf_data): continue
                for i in range(len(xy_data)):
                    person_kp, person_conf = xy_data[i].cpu().numpy(), conf_data[i].cpu().numpy()
                    valid_kps = person_kp[person_kp[:, 1] > 10]
                    if len(valid_kps) == 0: continue
                    min_x, max_x = np.min(valid_kps[:,0]), np.max(valid_kps[:,0])
                    min_y, max_y = np.min(valid_kps[:,1]), np.max(valid_kps[:,1])
                    dets.append([min_x, min_y, max_x, max_y, 1.0])
                    keypoints_list.append((person_kp, person_conf))
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
        tracks = tracker.update(dets)

        # --- 키보드 입력 처리 ---
        key = cv2.waitKeyEx(1)
        if key == ord('q'): break
        elif key == 2490368: KICK_THRESH_PIXELS_Y += 1
        elif key == 2621440: KICK_THRESH_PIXELS_Y = max(1, KICK_THRESH_PIXELS_Y - 1)
        elif key == 2424832: KICK_THRESH_PIXELS_X = max(1, KICK_THRESH_PIXELS_X - 1)
        elif key == 2555904: KICK_THRESH_PIXELS_X += 1
        elif key == 2162688: KICK_THRESH_RATIO_Z += 0.01
        elif key == 2228224: KICK_THRESH_RATIO_Z = max(0.01, KICK_THRESH_RATIO_Z - 0.01)
        # ▼▼▼ (수정) Home/End 키로 쿨다운 조절 ▼▼▼
        elif key == 2359296: # Home
            KICK_COOLDOWN_FRAMES += 1
        elif key == 2293760: # End
            KICK_COOLDOWN_FRAMES = max(1, KICK_COOLDOWN_FRAMES - 1)
        # ▲▲▲ ▲▲▲ ▲▲▲

        # --- ID별 로직 처리 ---
        active_track_ids = set()
        matched = set()
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            active_track_ids.add(track_id)
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            best_idx, best_dist = -1, 1e9
            for idx, (person_kp_loop, _) in enumerate(keypoints_list):
                if idx in matched: continue 
                center = np.array([person_kp_loop[0][0], person_kp_loop[0][1]])
                dist = np.linalg.norm(bbox_center - center)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx == -1: continue
            matched.add(best_idx)

            person_xy, person_conf = keypoints_list[best_idx]
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
                if is_full_body_visible: 
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
                    
                    if base_height > 100 and is_standing and is_facing_front and is_standing_aspect_ratio:
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
                                    base_l_ankle_x, base_l_ankle_y = person_xy[15][0], person_xy[15][1]
                                    base_r_ankle_x, base_r_ankle_y = person_xy[16][0], person_xy[16][1]
                                    l_hip_xy = person_xy[11][:2]
                                    r_hip_xy = person_xy[12][:2]
                                    base_l_hip_ankle_dist = calculate_distance(l_hip_xy, (base_l_ankle_x, base_l_ankle_y))
                                    base_r_hip_ankle_dist = calculate_distance(r_hip_xy, (base_r_ankle_x, base_r_ankle_y))

                                    Y_MID = KICK_THRESH_PIXELS_Y
                                    X_MID = KICK_THRESH_PIXELS_X
                                    Z_MID_EST_L = base_l_hip_ankle_dist * KICK_THRESH_RATIO_Z 
                                    Z_MID_EST_R = base_r_hip_ankle_dist * KICK_THRESH_RATIO_Z
                                    
                                    Y_HIGH = Y_MID * 2.0  
                                    Z_HIGH_EST_L = Z_MID_EST_L * 1.5 
                                    Z_HIGH_EST_R = Z_MID_EST_R * 1.5 
                                    
                                    Y_RST = Y_MID * 0.75 
                                    Z_RST_EST_L = Z_MID_EST_L * 0.4 
                                    Z_RST_EST_R = Z_MID_EST_R * 0.4

                                    base_data[track_id] = {
                                        "base_height": base_height, "base_bbox_height": current_bbox_height, "base_bbox_width": current_bbox_width,
                                        "base_l_ankle_x": base_l_ankle_x, "base_l_ankle_y": base_l_ankle_y, "base_l_hip_ankle_dist": base_l_hip_ankle_dist,
                                        "base_r_ankle_x": base_r_ankle_x, "base_r_ankle_y": base_r_ankle_y, "base_r_hip_ankle_dist": base_r_hip_ankle_dist,
                                        "Y_MID": Y_MID, "X_MID": X_MID, 
                                        "Z_MID_EST_L": Z_MID_EST_L, "Z_MID_EST_R": Z_MID_EST_R,
                                        "Y_HIGH": Y_HIGH, 
                                        "Z_HIGH_EST_L": Z_HIGH_EST_L, "Z_HIGH_EST_R": Z_HIGH_EST_R,
                                        "Y_RST": Y_RST, 
                                        "Z_RST_EST_L": Z_RST_EST_L, "Z_RST_EST_R": Z_RST_EST_R
                                    }
                                    
                                    if track_id not in track_id_to_player_id:
                                        new_player_id = player_id_counter
                                        track_id_to_player_id[track_id] = new_player_id
                                        player_id_counter += 1
                                        print(f"ID {track_id} 캘리브레이션 완료 (Player {new_player_id})")
                                    else:
                                        print(f"ID {track_id} (Player {track_id_to_player_id[track_id]}) 재캘리브레이션 완료")
                                    
                                    kick_counters[track_id] = 0
                                    l_kick_state[track_id] = 0
                                    r_kick_state[track_id] = 0
                                    l_reset_counter[track_id] = 0
                                    r_reset_counter[track_id] = 0
                                    person_kick_timer[track_id] = 0
                                else:
                                    print(f"ID {track_id} 캘리브레이션 실패: Y좌표 흔들림 {y_movement}px")
                                floor_timers[track_id], floor_y_history[track_id] = None, []
                    else:
                        floor_timers[track_id], floor_y_history[track_id] = None, []
                else:
                    floor_timers[track_id], floor_y_history[track_id] = None, []

            # 6. 킥 카운트 단계
            else:
                pd = base_data[track_id] 
                
                is_too_close = (current_bbox_height > pd['base_bbox_height'] * 2.0) or (current_bbox_width > pd['base_bbox_width'] * 2.0) 

                if is_too_close:
                    final_scores[track_id] = kick_counters[track_id]; del base_data[track_id]; print(f"ID {track_id} (Player {track_id_to_player_id.get(track_id, '?')}) 너무 가까움. 리셋.")
                else:
                    # --- 1. 현재 값 계산 ---
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

                    # --- 2. 기준값 가져오기 ---
                    Y_MID, X_MID = pd['Y_MID'], pd['X_MID']
                    Y_HIGH = pd['Y_HIGH']
                    Y_RST = pd['Y_RST']
                    Z_MID_EST_L, Z_MID_EST_R = pd['Z_MID_EST_L'], pd['Z_MID_EST_R']
                    Z_HIGH_EST_L, Z_HIGH_EST_R = pd['Z_HIGH_EST_L'], pd['Z_HIGH_EST_R']
                    Z_RST_EST_L, Z_RST_EST_R = pd['Z_RST_EST_L'], pd['Z_RST_EST_R']
                    
                    player_id = track_id_to_player_id.get(track_id, '?')
                    kick_detected_this_frame = False

                    # --- 3. '사람' 쿨다운 타이머 감소 ---
                    if person_kick_timer[track_id] > 0: 
                        person_kick_timer[track_id] -= 1

                    # --- 4. 왼발 킥 로직 ---
                    if person_kick_timer[track_id] == 0:
                        if l_kick_state[track_id] == 0:
                            
                            is_vis = is_l_ankle_visible and is_l_hip_visible
                            
                            if is_vis and l_y_diff > Y_MID: 
                                l_mid = (l_x_diff > X_MID and l_z_diff_est > Z_MID_EST_L)
                                l_high = (l_y_diff > Y_HIGH or l_z_diff_est > Z_HIGH_EST_L)

                                if l_mid or l_high:
                                    kick_counters[track_id] += 1
                                    l_kick_state[track_id] = 1 
                                    l_reset_counter[track_id] = 0
                                    person_kick_timer[track_id] = KICK_COOLDOWN_FRAMES 
                                    kick_detected_this_frame = True 
                                    reason = "Mid" if l_mid else ("High(Y)" if l_y_diff > Y_HIGH else "High(Z)")
                                    print(f"=== ID {player_id} L-Kick!({reason})(Y:{l_y_diff:.1f}, X:{l_x_diff:.1f}, Z:{l_z_diff_est:.1f})(Tot:{kick_counters[track_id]}) ===")

                        elif l_kick_state[track_id] == 1:
                            l_base_cond = (is_l_ankle_visible and
                                           l_y_diff < Y_RST and
                                           l_z_diff_est < Z_RST_EST_L)
                            
                            if l_base_cond:
                                l_reset_counter[track_id] += 1
                            else:
                                l_reset_counter[track_id] = 0 
                            
                            if l_reset_counter[track_id] >= RESET_FRAME_COUNT:
                                l_kick_state[track_id] = 0 
                                print(f"ID {player_id} L-Kick RESET.")

                    # --- 5. 오른발 킥 로직 ---
                    if not kick_detected_this_frame and person_kick_timer[track_id] == 0:
                        if r_kick_state[track_id] == 0:
                            
                            is_vis = is_r_ankle_visible and is_r_hip_visible

                            if is_vis and r_y_diff > Y_MID:
                                r_mid = (r_x_diff > X_MID and r_z_diff_est > Z_MID_EST_R)
                                r_high = (r_y_diff > Y_HIGH or r_z_diff_est > Z_HIGH_EST_R)

                                if r_mid or r_high:
                                    kick_counters[track_id] += 1
                                    r_kick_state[track_id] = 1
                                    r_reset_counter[track_id] = 0
                                    person_kick_timer[track_id] = KICK_COOLDOWN_FRAMES 
                                    reason = "Mid" if r_mid else ("High(Y)" if r_y_diff > Y_HIGH else "High(Z)")
                                    print(f"=== ID {player_id} R-Kick!({reason})(Y:{r_y_diff:.1f}, X:{r_x_diff:.1f}, Z:{r_z_diff_est:.1f})(Tot:{kick_counters[track_id]}) ===")

                        elif r_kick_state[track_id] == 1:
                            r_base_cond = (is_r_ankle_visible and
                                           r_y_diff < Y_RST and
                                           r_z_diff_est < Z_RST_EST_R)
                            
                            if r_base_cond:
                                r_reset_counter[track_id] += 1
                            else:
                                r_reset_counter[track_id] = 0
                            
                            if r_reset_counter[track_id] >= RESET_FRAME_COUNT:
                                r_kick_state[track_id] = 0
                                print(f"ID {player_id} R-Kick RESET.")
            
            if is_head_visible:
                ui_center_x, ui_center_y, radius, font_scale = int(head_x), int(head_y) - 60, 30, 1.0
                
                if track_id in base_data: 
                    player_id = track_id_to_player_id.get(track_id, "?")
                    text_count = f'{player_id}'
                    color = (0, 100, 255) if person_kick_timer[track_id] > 0 else (0, 255, 0) 
                    
                    cv2.circle(display_frame, (ui_center_x, ui_center_y), radius, color, -1)
                    draw_text_with_bg(display_frame, text_count, (ui_center_x - int(radius*0.3), ui_center_y + int(radius*0.3)), font_scale*0.8, 2, (0,0,0), color)
                    
                    count_str = f"K: {kick_counters[track_id]}"
                    draw_text_with_bg(display_frame, count_str, (ui_center_x + radius + 5, ui_center_y + 10), font_scale * 0.7, 2, (255, 255, 255), (0, 0, 0))

                elif floor_timers[track_id] is not None and is_full_body_visible:
                    if person_conf[11] >= JOINT_CONF_THRESH and person_conf[12] >= JOINT_CONF_THRESH and \
                       person_conf[13] >= JOINT_CONF_THRESH and person_conf[14] >= JOINT_CONF_THRESH:
                        
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
                        is_standing_aspect_ratio = current_bbox_height > (current_bbox_width * 1.3)
                        
                        if is_standing and is_facing_front and is_standing_aspect_ratio:
                            elapsed = time.time() - floor_timers[track_id]
                            angle = min((elapsed / CALIBRATION_TIME) * 360, 360)
                            
                            cv2.circle(display_frame, (ui_center_x, ui_center_y), radius, (100,100,100), 2)
                            cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius,radius), -90, 0, angle, (0,255,255), 4)
                            
                            text_percent = f'{min(int(angle/3.6), 100)}%'
                            draw_text_with_bg(display_frame, text_percent, (ui_center_x - 18, ui_center_y + 10), font_scale*0.7, 2, (255,255,255), (0,0,0))

        # === ID별 순회 루프 끝 ===

        # --- 사라진 ID 처리 ---
        active_ids_set = set(t[4] for t in tracks)
        tracked_ids = set(base_data.keys()) | set(k for k,v in floor_timers.items() if v is not None)
        lost_ids = tracked_ids - active_ids_set
        
        for lost_track_id in lost_ids:
            if lost_track_id in base_data:
                player_id = track_id_to_player_id.get(lost_track_id, 0)
                if player_id != 0:
                    print(f"ID {int(lost_track_id)} (Player {player_id}) 화면 이탈. 점수 저장.")
                    final_scores[player_id] = kick_counters[lost_track_id]
                else:
                    print(f"ID {int(lost_track_id)} (Player ?) 화면 이탈. 리셋.")
            elif lost_track_id in floor_timers and floor_timers[lost_track_id] is not None:
                print(f"ID {int(lost_track_id)} 캘리브 중 이탈. 리셋.")
            
            for d in [base_data, kick_counters, floor_timers, floor_y_history, track_id_to_player_id, 
                      l_kick_state, r_kick_state, l_reset_counter, r_reset_counter, person_kick_timer]:
                if lost_track_id in d: del d[lost_track_id]

        # --- 최종 점수판 그리기 ---
        y_pos, x_pos = frame_height - 100, frame_width - 250
        draw_text_with_bg(display_frame, "== Final Scores ==", (x_pos, y_pos), 0.7, 2, (0, 255, 255), (0,0,0))
        sorted_scores = sorted([(int(id_num), count) for id_num, count in final_scores.items()])
        for id_num, count in sorted_scores:
            y_pos += 30
            text = f"ID {id_num} : {count}"
            cv2.putText(display_frame, text, (x_pos + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- 최종 화면 표시 ---
        cv2.imshow('Kick Counter - Multi Person Tracking', display_frame)

    # === 메인 루프 끝 ===
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()