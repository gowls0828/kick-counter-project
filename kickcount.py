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

# === [수정] 텍스트 그리기 함수 (PIL 사용) ===
def draw_text_pil(img_bgr, text, pos, font, txt_color_bgr=(255,255,255), bg_color_bgr=(0,0,0)):
    """ 
    OpenCV BGR 이미지를 받아 PIL로 텍스트와 배경을 그립니다.
    pos는 OpenCV의 putText와 유사하게 (x, y) 좌측 하단 기준입니다.
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        txt_col = (txt_color_bgr[2], txt_color_bgr[1], txt_color_bgr[0]) # BGR -> RGB
        bg_col = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0]) # BGR -> RGB

        # 텍스트 바운딩 박스 계산 (Pillow 10.x 호환)
        if hasattr(font, 'getbbox'):
            bbox = draw.textbbox((0,0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height_from_baseline = bbox[3] # 텍스트의 총 높이 (베이스라인 기준)
        else:
            # 이전 버전 호환
            text_width, text_height_from_baseline = draw.textsize(text, font=font)

        # OpenCV (x, y)는 좌하단 기준 -> PIL (x, y)는 좌상단 기준
        pil_pos = (pos[0], pos[1] - text_height_from_baseline - int(font.size * 0.15)) # 베이스라인 근사치 조정
        
        # 배경 계산
        tl = (pil_pos[0] - 2, pil_pos[1] - 2)
        br = (pil_pos[0] + text_width + 2, pil_pos[1] + text_height_from_baseline + 2)
        
        draw.rectangle([tl, br], fill=bg_col)
        draw.text(pil_pos, text, font=font, fill=txt_col)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        return img_bgr # 오류 시 원본 반환

# === 상태 정의 ===
MENU_START = 0
MENU_PLAYER_SELECT = 1
GAME_RUNNING = 2

# === [신규] 캘리브레이션 박스 생성 함수 ===
# [수정] 자동 간격 계산 및 높이 조절(bottom_margin) 기능 적용
def get_calibration_boxes(max_players, W, H, bottom_margin=20):
    boxes = []
    # [수정] 박스 크기 2배로 증가 (150->300, 50->100)
    box_width = 300
    box_height = 100
    
    # [수정] max_players 값에 따라 화면을 n+1 등분하여 중앙에 배치 (균등 간격)
    positions = []
    num_sections = max_players + 1
    for i in range(1, num_sections):
        positions.append((W // num_sections) * i)
                         
    for cx in positions:
        x1 = cx - (box_width // 2)
        # [수정] 파라미터로 받은 bottom_margin 적용
        y1 = H - bottom_margin - box_height
        x2 = cx + (box_width // 2)
        y2 = H - bottom_margin
        boxes.append((x1, y1, x2, y2))
        
    return boxes

# === [신규] 점이 박스 안에 있는지 확인하는 함수 ===
def is_point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    # [수정] 경계선을 포함하도록 <= 로 변경
    return (x1 <= x <= x2) and (y1 <= y <= y2)

# === [신규] UI 그리기 함수 (PIL 사용) ===
def draw_menu_ui(state, frame_dims=(1920, 1080), fonts=None):
    H, W = frame_dims[1], frame_dims[0]
    
    # 1. 검은색 배경 생성
    frame = np.zeros((H, W, 3), dtype=np.uint8) 
    
    # BGR -> RGB 변환 (PIL 작업용)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    if state == MENU_START:
        # 2. 메인 타이틀: 스피드 발차기
        text = "스피드 발차기"
        bbox = draw.textbbox((0,0), text, font=fonts['title'])
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = ( (W - text_w) // 2, (H // 2) - 150 )
        draw.text(pos, text, font=fonts['title'], fill=(255, 255, 0)) # 노란색
        
        # 3. 시작 안내 텍스트
        text = "시작하려면 아무 키나 눌러 주세요."
        bbox = draw.textbbox((0,0), text, font=fonts['subtitle'])
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = ( (W - text_w) // 2, (H // 2) + 50 )
        draw.text(pos, text, font=fonts['subtitle'], fill=(255, 255, 255)) # 흰색

    elif state == MENU_PLAYER_SELECT:
        # 2. 플레이할 인원 선택 안내
        text = "플레이할 인원을 선택해 주세요. (1~4 키)"
        bbox = draw.textbbox((0,0), text, font=fonts['subtitle'])
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = ( (W - text_w) // 2, (H // 2) - 200 )
        draw.text(pos, text, font=fonts['subtitle'], fill=(255, 255, 255))
        
        button_width = 250
        button_height = 250
        gap = 50
        start_x = (W - (4 * button_width + 3 * gap)) // 2
        
        # 3. 인원 버튼 배치
        for i in range(1, 5):
            x1 = start_x + (i - 1) * (button_width + gap)
            y1 = H // 2 - 100
            x2 = x1 + button_width
            y2 = y1 + button_height
            
            # 버튼 그리기 (BGR: (200, 20, 20) -> RGB: (20, 20, 200))
            draw.rectangle([(x1, y1), (x2, y2)], fill=(20, 20, 200), outline=(255,255,255), width=2)
            
            # 텍스트 그리기
            text = f"{i}인"
            bbox = draw.textbbox((0,0), text, font=fonts['menu'])
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = x1 + (button_width - text_w) // 2
            text_y = y1 + (button_height - text_h) // 2
            draw.text((text_x, text_y), text, font=fonts['menu'], fill=(255, 255, 255))

    # PIL -> BGR 변환하여 반환
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# === [신규] PIL Draw 객체에 직접 텍스트를 그리는 헬퍼 함수 ===
def draw_pil_text_on_image(draw, text, pos_top_left, font, text_color_rgb, bg_color_rgb=None):
    """ 
    PIL 'draw' 객체에 직접 텍스트와 배경을 그립니다. 
    pos는 좌상단(top-left) 기준입니다.
    bg_color_rgb가 None이 아니면 배경 사각형을 그립니다.
    """
    try:
        if bg_color_rgb is not None:
            if hasattr(font, 'getbbox'):
                # Pillow 10+
                bbox = draw.textbbox((0,0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1] 
            else:
                # Pillow < 10
                text_width, text_height = draw.textsize(text, font=font)
            
            # 배경 그리기 (pos_top_left 기준)
            bg_tl = (pos_top_left[0] - 2, pos_top_left[1] - 2)
            bg_br = (pos_top_left[0] + text_width + 2, pos_top_left[1] + text_height + 2)
            
            # [★★★ 버그 수정 ★★★]
            # 'br'이 아니라 'bg_br' 변수를 사용해야 합니다.
            draw.rectangle([bg_tl, bg_br], fill=bg_color_rgb)
        
        # 텍스트 그리기
        draw.text(pos_top_left, text, font=font, fill=text_color_rgb)
    except Exception as e:
        # print(f"Error drawing text: {e}") # 디버깅용
        pass # 오류 시 그리기를 건너뜀

# === 메인 프로그램 ===
def main():
    # --- 1. 초기 설정 ---
    
    # [수정] PyInstaller 호환 경로 사용
    model_path = resource_path('yolov8n-pose.pt')
    
    try:
        model = YOLO(model_path)
        print("모델 로드 성공.")
    except Exception as e:
        print(f"YOLOv8 모델 로드 오류: {e}\n경로: {model_path}")
        return 
        
    tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.3)
    
    # --- 상태 변수 (UI 통합) ---
    current_state = MENU_START 
    max_players = 0 # 인원 선택 시 이 값 변경

    # --- 킥 카운트 변수 (고객님 코드 유지) ---
    base_data = {}; final_scores = {}; kick_counters = defaultdict(int); 
    l_kick_state = defaultdict(int); r_kick_state = defaultdict(int); 
    l_reset_counter = defaultdict(int); r_reset_counter = defaultdict(int); 
    person_kick_timer = defaultdict(int); 
    
    RESET_FRAME_COUNT = 3
    KICK_COOLDOWN_FRAMES = 5 
    
    player_id_counter = 1
    track_id_to_player_id = {} 

    floor_timers = defaultdict(lambda: None)
    floor_y_history = defaultdict(list)
    CALIBRATION_TIME = 2.0
    STABILITY_THRESH = 20

    KICK_THRESH_PIXELS_Y = 30 
    # [★★★ 사용자 요청 수정 ★★★] 
    # X축(좌우) 민감도를 1로 최소화하여 앞발차기 인식이 잘 되도록 함
    KICK_THRESH_PIXELS_X = 1 
    KICK_THRESH_RATIO_Z = 0.10 # 10%

    JOINT_CONF_THRESH = 0.1 # 10%
    
    # [수정] 캘리브레이션 박스 화면 하단 여백 (기본 30 -> 50 으로 더 올림)
    CALIB_BOX_BOTTOM_MARGIN = 50 
    
    # [신규] 캘리브레이션 박스 활성화 상태
    active_calib_boxes = {} # {box_index: track_id}

    # [★★★ 사용자 요청 수정 1/5 ★★★]
    # V키로 UI를 토글하기 위한 상태 변수
    show_debug_ui = True 

    # --- 2. 카메라/화면 설정 ---
    try:
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except Exception:
        screen_width, screen_height = 1920, 1080
        
    print("모니터 해상도:", screen_width, screen_height)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("웹캠 열 수 없음."); return

    # [최종 수정] 해상도를 1920x1080 (FHD)로 복구
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 실제 해상도: {frame_width} x {frame_height}")

    cv2.namedWindow('Kick Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Kick Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- [수정] 폰트 로드 ---
    try:
        FONT_PATH = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕
        fonts = {
            'title': ImageFont.truetype(FONT_PATH, 100),
            'subtitle': ImageFont.truetype(FONT_PATH, 40),
            'menu': ImageFont.truetype(FONT_PATH, 50),
            'ui_main': ImageFont.truetype(FONT_PATH, 24),
            'ui_kick': ImageFont.truetype(FONT_PATH, 20),
            'ui_player': ImageFont.truetype(FONT_PATH, 22),
            'ui_percent': ImageFont.truetype(FONT_PATH, 20)
        }
    except IOError:
        print(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
        fonts = None # 폰트 로드 실패

    # === 3. 메인 루프 (프로그램 실행) ===
    tracks = [] # [수정] tracks 변수 초기화

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        # [최종 수정] 카메라 원본 프레임 복사
        display_frame = frame.copy()

        key = cv2.waitKeyEx(1)
        
        # -----------------------------------------------------
        # A. UI 및 상태 전환 로직
        # -----------------------------------------------------

        if current_state == MENU_START:
            if fonts:
                display_frame = draw_menu_ui(MENU_START, frame_dims=(frame_width, frame_height), fonts=fonts)
            if key != -1 and key != 255: # 아무 키나 누르면
                current_state = MENU_PLAYER_SELECT
        
        elif current_state == MENU_PLAYER_SELECT:
            if fonts:
                display_frame = draw_menu_ui(MENU_PLAYER_SELECT, frame_dims=(frame_width, frame_height), fonts=fonts)
            
            if key == ord('1'): max_players = 1; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드")
            elif key == ord('2'): max_players = 2; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드")
            elif key == ord('3'): max_players = 3; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드")
            elif key == ord('4'): max_players = 4; current_state = GAME_RUNNING; print(f"게임 시작: {max_players}인 모드")
        
        # -----------------------------------------------------
        # B. 게임 실행 로직 (고객님 코드 유지)
        # -----------------------------------------------------

        elif current_state == GAME_RUNNING:
            
            # [최적화 2/10] PIL로 그릴 텍스트 정보를 담을 리스트 초기화
            pil_draw_list = [] 

            # [★★★ 사용자 요청 수정 ★★★]
            # 텍스트 크기 계산을 위한 임시 Draw 객체 (한 번만 생성)
            temp_pil_img = Image.new("RGB", (1,1))
            temp_draw = ImageDraw.Draw(temp_pil_img)

            # [최적화 3/10] OpenCV로 캘리브레이션 박스를 먼저 그림 (빠름)
            calibration_boxes = get_calibration_boxes(max_players, frame_width, frame_height, CALIB_BOX_BOTTOM_MARGIN)
            
            for i, box in enumerate(calibration_boxes):
                x1, y1, x2, y2 = box
                
                if i in active_calib_boxes:
                    # 활성 (초록)
                    color = (0, 255, 0) 
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                else:
                    # 비활성 (빨강)
                    color = (0, 0, 255) 
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # [★★★ 사용자 요청 수정 ★★★]
                    # 비활성 박스에 "여기 서주세요" 텍스트 추가 (크게, 파란색으로)
                    text = "여기 서주세요"
                    # [수정] 폰트를 'subtitle'(40pt)로 변경
                    font = fonts['subtitle'] 
                    
                    # 텍스트 크기 계산 (미리 만든 temp_draw 사용)
                    if hasattr(temp_draw, 'textbbox'):
                        bbox = temp_draw.textbbox((0,0), text, font=font)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        text_w, text_h = temp_draw.textsize(text, font=font)
                    
                    # 텍스트 위치 계산 (박스 중앙)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    text_x = x1 + (box_width - text_w) // 2
                    text_y = y1 + (box_height - text_h) // 2
                    
                    # [수정] PIL 그리기 리스트에 추가 (글자색을 노란색 -> 파란색으로 변경)
                    pil_draw_list.append( (text, (text_x, text_y), font, (0, 0, 255), None) ) # 파란색

            # --- 사람 감지 및 ID 추적 ---
            results = model(frame, conf=0.6, verbose=False) # 원본 BGR 프레임 사용
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
            
            # [수정] max_players 명에게만 트래킹을 시도하도록 조정
            if len(dets) > max_players:
                sorted_indices = sorted(range(len(dets)), key=lambda k: (dets[k][2]-dets[k][0]) * (dets[k][3]-dets[k][1]), reverse=True)
                dets_to_track = np.array([dets[i] for i in sorted_indices[:max_players]])
                keypoints_list_to_track = [keypoints_list[i] for i in sorted_indices[:max_players]]
            else:
                dets_to_track = dets
                keypoints_list_to_track = keypoints_list
                
            tracks = tracker.update(dets_to_track) # 'tracks' 변수가 여기서 정의됨

            # --- [신규] 디버그 로그 ---
            print(f"[LOG] Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: {len(dets_to_track)} people detected, {len(tracks)} people tracked.")

            # --- 키보드 입력 처리 ---
            if key == ord('q'): break
            elif key == 2490368: KICK_THRESH_PIXELS_Y += 1
            elif key == 2621440: KICK_THRESH_PIXELS_Y = max(1, KICK_THRESH_PIXELS_Y - 1)
            elif key == 2424832: KICK_THRESH_PIXELS_X = max(1, KICK_THRESH_PIXELS_X - 1)
            elif key == 2555904: KICK_THRESH_PIXELS_X += 1
            elif key == 2162688: KICK_THRESH_RATIO_Z += 0.01
            elif key == 2228224: KICK_THRESH_RATIO_Z = max(0.01, KICK_THRESH_RATIO_Z - 0.01)
            elif key == 2359296: KICK_COOLDOWN_FRAMES += 1
            elif key == 2293760: KICK_COOLDOWN_FRAMES = max(1, KICK_COOLDOWN_FRAMES - 1)
            
            # [추가] 캘리브레이션 박스 높이 조절 (숫자 8, 2)
            elif key == ord('8'): CALIB_BOX_BOTTOM_MARGIN += 5
            elif key == ord('2'): CALIB_BOX_BOTTOM_MARGIN = max(5, CALIB_BOX_BOTTOM_MARGIN - 5)
            
            # [★★★ 사용자 요청 수정 2/5 ★★★] V키로 UI 토글
            elif key == ord('v'):
                show_debug_ui = not show_debug_ui
            
            # [수정] ESC 키로 '인원 선택' 화면 복귀 
            elif key == 27: 
                current_state = MENU_PLAYER_SELECT # MENU_START -> MENU_PLAYER_SELECT
                base_data.clear(); final_scores.clear(); kick_counters.clear()
                l_kick_state.clear(); r_kick_state.clear(); l_reset_counter.clear()
                r_reset_counter.clear(); person_kick_timer.clear(); 
                floor_timers.clear(); floor_y_history.clear();
                track_id_to_player_id.clear(); player_id_counter = 1
                active_calib_boxes.clear() # 캘리브 박스 상태 초기화
                print("인원 선택 화면으로 복귀. 모든 데이터 초기화.")
                continue 

            # --- ID별 로직 처리 ---
            active_track_ids = set()
            matched = set()
            
            # [신규] 현재 프레임에서 활성화된 캘리브 박스 초기화
            active_calib_boxes.clear()
            
            # [수정] tracks가 0명일 때 에러가 나지 않도록 if문으로 감쌉니다.
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
                    
                    l_ankle_conf, r_ankle_conf = person_conf[15], person_conf[16]
                    if l_ankle_conf < JOINT_CONF_THRESH and r_ankle_conf < JOINT_CONF_THRESH: continue
                    
                    foot_y = max(person_xy[15][1], person_xy[16][1])
                    min_x, min_y, max_x, max_y = int(x1), int(y1), int(x2), int(y2)
                    current_bbox_height, current_bbox_width = max_y - min_y, max_x - min_x
                    head_conf = person_conf[0]
                    is_head_visible = head_conf >= JOINT_CONF_THRESH
                    
                    major_joints_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
                    is_full_body_visible = all(person_conf[idx] >= JOINT_CONF_THRESH for idx in major_joints_indices)

                    # 5. 캘리브레이션 단계
                    if track_id not in base_data:
                        
                        # [신규] 캘리브레이션 트리거: 발목이 박스 안에 있는지 확인
                        l_ankle_xy, r_ankle_xy = person_xy[15][:2], person_xy[16][:2]
                        is_in_any_box = False
                        for i, box in enumerate(calibration_boxes):
                            if is_point_in_box(l_ankle_xy, box) or is_point_in_box(r_ankle_xy, box):
                                is_in_any_box = True
                                active_calib_boxes[i] = track_id # 박스 활성화 (초록색으로 그림)
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
                        l_leg_dist_2d = calculate_distance(l_hip_xy, l_ankle_xy)
                        r_leg_dist_2d = calculate_distance(r_hip_xy, r_ankle_xy)
                        l_leg_dist_y = abs(l_ankle_xy[1] - l_hip_xy[1])
                        r_leg_dist_y = abs(r_ankle_xy[1] - r_hip_xy[1])
                        LEG_STRAIGHTNESS_RATIO = 0.80
                        is_l_leg_straight = (l_leg_dist_2d > 10) and ((l_leg_dist_y / l_leg_dist_2d) > LEG_STRAIGHTNESS_RATIO)
                        is_r_leg_straight = (r_leg_dist_2d > 10) and ((r_leg_dist_y / r_leg_dist_2d) > LEG_STRAIGHTNESS_RATIO)
                        is_standing = is_l_leg_straight or is_r_leg_straight
                        
                        # [수정] 캘리브레이션 시작 조건에 'is_in_any_box' 추가
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
                                        base_l_ankle_x, base_l_ankle_y = person_xy[15][0], person_xy[15][1]
                                        base_r_ankle_x, base_r_ankle_y = person_xy[16][0], person_xy[16][1]
                                        l_hip_xy = person_xy[11][:2]
                                        r_hip_xy = person_xy[12][:2]
                                        base_l_hip_ankle_dist = calculate_distance(l_hip_xy, (base_l_ankle_x, base_l_ankle_y))
                                        base_r_hip_ankle_dist = calculate_distance(r_hip_xy, (base_r_ankle_x, base_r_ankle_y))

                                        Y_MID = KICK_THRESH_PIXELS_Y; X_MID = KICK_THRESH_PIXELS_X;
                                        Z_MID_EST_L = base_l_hip_ankle_dist * KICK_THRESH_RATIO_Z 
                                        Z_MID_EST_R = base_r_hip_ankle_dist * KICK_THRESH_RATIO_Z
                                        Y_HIGH = Y_MID * 2.0; Z_HIGH_EST_L = Z_MID_EST_L * 1.5; Z_HIGH_EST_R = Z_MID_EST_R * 1.5 
                                        Y_RST = Y_MID * 0.75; Z_RST_EST_L = Z_MID_EST_L * 0.4; Z_RST_EST_R = Z_MID_EST_R * 0.4

                                        base_data[track_id] = {
                                            "base_height": base_height, "base_bbox_height": current_bbox_height, "base_bbox_width": current_bbox_width,
                                            "base_l_ankle_x": base_l_ankle_x, "base_l_ankle_y": base_l_ankle_y, "base_l_hip_ankle_dist": base_l_hip_ankle_dist,
                                            "base_r_ankle_x": base_r_ankle_x, "base_r_ankle_y": base_r_ankle_y, "base_r_hip_ankle_dist": base_r_hip_ankle_dist,
                                            "Y_MID": Y_MID, "X_MID": X_MID, "Z_MID_EST_L": Z_MID_EST_L, "Z_MID_EST_R": Z_MID_EST_R,
                                            "Y_HIGH": Y_HIGH, "Z_HIGH_EST_L": Z_HIGH_EST_L, "Z_HIGH_EST_R": Z_HIGH_EST_R,
                                            "Y_RST": Y_RST, "Z_RST_EST_L": Z_RST_EST_L, "Z_RST_EST_R": Z_RST_EST_R
                                        }
                                        
                                        if track_id not in track_id_to_player_id:
                                            new_player_id = player_id_counter
                                            track_id_to_player_id[track_id] = new_player_id
                                            player_id_counter += 1
                                            print(f"ID {track_id} 캘리브레이션 완료 (Player {new_player_id})")
                                        else:
                                            print(f"ID {track_id} (Player {track_id_to_player_id[track_id]}) 재캘리브레이션 완료")
                                        
                                        kick_counters[track_id] = 0; l_kick_state[track_id] = 0; r_kick_state[track_id] = 0;
                                        l_reset_counter[track_id] = 0; r_reset_counter[track_id] = 0; person_kick_timer[track_id] = 0;
                                    else:
                                        print(f"ID {track_id} 캘리브레이션 실패: Y좌표 흔들림 {y_movement}px")
                                        floor_timers[track_id], floor_y_history[track_id] = None, []
                        else:
                            floor_timers[track_id], floor_y_history[track_id] = None, []
                    else:
                        floor_timers[track_id], floor_y_history[track_id] = None, []
                    
                    # [최적화 4/10] 캘리브레이션 게이지 (CV2로 도형 그리고, PIL정보 저장)
                    if is_head_visible and floor_timers[track_id] is not None and fonts:
                        ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                        elapsed = time.time() - floor_timers[track_id]
                        angle = min((elapsed / CALIBRATION_TIME) * 360, 360)
                        
                        # OpenCV로 도형 그리기 (빠름)
                        cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, (100,100,100), 2)
                        cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, -90, int(angle-90), (0,255,255), 4) # 각도는 정수여야 함
                        
                        # PIL로 그릴 텍스트 정보 저장
                        text_percent = f'{min(int(angle/3.6), 100)}%'
                        
                        # [★★★ 사용자 요청 수정 ★★★] (temp_draw 생성 코드 삭제)
                        if hasattr(temp_draw, 'textbbox'):
                            bbox = temp_draw.textbbox((0,0), text_percent, font=fonts['ui_percent'])
                            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        else:
                            text_w, text_h = temp_draw.textsize(text_percent, font=fonts['ui_percent'])
                        
                        # (pos_top_left 기준)
                        text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2) # Y축 중앙 정렬
                        pil_draw_list.append( (text_percent, text_pos, fonts['ui_percent'], (255,255,255), None) ) # 배경 없음

                    # 6. 킥 카운트 단계
                    else:
                        # ★★★★★ [안전 가드] ★★★★★
                        # 캘리브레이션이 완료되었는지 한번 더 확인
                        if track_id not in base_data:
                            # print(f"!!! [LOGIC WARNING] ID {track_id} is not calibrated but entered kick block. Skipping.")
                            continue # 이 사람을 건너뛰어 프로그램이 꺼지는 것을 방지
                        
                        pd = base_data[track_id] 
                        
                        is_too_close = (current_bbox_height > pd['base_bbox_height'] * 2.0) or (current_bbox_width > pd['base_bbox_width'] * 2.0) 

                        if is_too_close:
                            final_scores[track_id_to_player_id.get(track_id, 0)] = kick_counters[track_id]; del base_data[track_id]; print(f"ID {track_id} (Player {track_id_to_player_id.get(track_id, '?')}) 너무 가까움. 리셋.")
                        else:
                            # (킥 카운트 로직은 고객님 코드 그대로 유지)
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
                            Y_HIGH = pd['Y_HIGH']; Y_RST = pd['Y_RST']
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
                                            kick_counters[track_id] += 1; l_kick_state[track_id] = 1; l_reset_counter[track_id] = 0
                                            person_kick_timer[track_id] = KICK_COOLDOWN_FRAMES; kick_detected_this_frame = True 
                                            reason = "Mid" if l_mid else ("High(Y)" if l_y_diff > Y_HIGH else "High(Z)")
                                            print(f"=== Player {player_id} L-Kick!({reason})(Y:{l_y_diff:.1f}, X:{l_x_diff:.1f}, Z:{l_z_diff_est:.1f})(Tot:{kick_counters[track_id]}) ===")
                                elif l_kick_state[track_id] == 1:
                                    l_base_cond = (is_l_ankle_visible and l_y_diff < Y_RST and l_z_diff_est < Z_RST_EST_L)
                                    if l_base_cond: l_reset_counter[track_id] += 1
                                    else: l_reset_counter[track_id] = 0 
                                    if l_reset_counter[track_id] >= RESET_FRAME_COUNT: l_kick_state[track_id] = 0; print(f"Player {player_id} L-Kick RESET.")

                            # --- 5. 오른발 킥 로직 ---
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
                        
                        # [최적화 5/10] 킥 카운트 UI (CV2로 도형 그리고, PIL정보 저장)
                        if is_head_visible and fonts:
                            ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                            
                            if track_id in base_data: 
                                player_id = track_id_to_player_id.get(track_id, "?")
                                text_count = f'{player_id}'
                                # (BGR) 쿨다운(주황), 평시(초록)
                                color_bgr = (0, 100, 255) if person_kick_timer[track_id] > 0 else (0, 255, 0) 
                                # (RGB) 텍스트 배경색용
                                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) 

                                # OpenCV로 원 그리기 (빠름)
                                cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, color_bgr, -1) # 꽉 채우기
                                
                                # [★★★ 사용자 요청 수정 ★★★] (temp_draw 생성 코드 삭제)
                                
                                # 1. 플레이어 ID 텍스트 정보 저장
                                if hasattr(temp_draw, 'textbbox'):
                                    bbox = temp_draw.textbbox((0,0), text_count, font=fonts['ui_player'])
                                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                else:
                                    text_w, text_h = temp_draw.textsize(text_count, font=fonts['ui_player'])
                                text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2) # Y축 중앙 (top-left)
                                pil_draw_list.append( (text_count, text_pos, fonts['ui_player'], (0,0,0), None) ) # 배경 없음 (원은 CV2가 그림)

                                # 2. 킥 카운트 텍스트 정보 저장
                                count_str = f"K: {kick_counters[track_id]}"
                                if hasattr(temp_draw, 'textbbox'):
                                    bbox_kick = temp_draw.textbbox((0,0), count_str, font=fonts['ui_kick'])
                                    text_w_k, text_h_k = bbox_kick[2] - bbox_kick[0], bbox_kick[3] - bbox_kick[1]
                                else:
                                    text_w_k, text_h_k = temp_draw.textsize(count_str, font=fonts['ui_kick'])
                                pos_kick = (ui_center_x + radius + 5, ui_center_y + (radius//2) - (text_h_k//2) - 2) # 원 중앙 기준 Y (top-left)
                                pil_draw_list.append( (count_str, pos_kick, fonts['ui_kick'], (255, 255, 255), (0,0,0)) ) # 검은 배경

                            elif floor_timers[track_id] is not None:
                                # (이 코드는 캘리브레이션 게이지와 동일 - L533)
                                elapsed = time.time() - floor_timers[track_id]
                                angle = min((elapsed / CALIBRATION_TIME) * 360, 360)
                                cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, (100,100,100), 2)
                                cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, -90, int(angle-90), (0,255,255), 4)
                                
                                text_percent = f'{min(int(angle/3.6), 100)}%'
                                # [★★★ 사용자 요청 수정 ★★★] (temp_draw 생성 코드 삭제)
                                if hasattr(temp_draw, 'textbbox'):
                                    bbox = temp_draw.textbbox((0,0), text_percent, font=fonts['ui_percent'])
                                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                else:
                                    text_w, text_h = temp_draw.textsize(text_percent, font=fonts['ui_percent'])
                                text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2)
                                pil_draw_list.append( (text_percent, text_pos, fonts['ui_percent'], (255,255,255), None) )
            
            # --- 8. 사라진 ID 처리 ---
            active_ids_set = set(t[4] for t in tracks)
            tracked_ids = set(base_data.keys()) | set(k for k,v in floor_timers.items() if v is not None)
            lost_ids = tracked_ids - active_ids_set
            
            for lost_track_id in lost_ids:
                if lost_track_id in base_data:
                    player_id = track_id_to_player_id.get(lost_track_id, 0)
                    if player_id != 0:
                        final_scores[player_id] = kick_counters[lost_track_id]
                        print(f"Player {player_id} (ID {int(lost_track_id)}) 화면 이탈. 점수 저장.")
                    else:
                        print(f"ID {int(lost_track_id)} 화면 이탈. 리셋.")
                elif lost_track_id in floor_timers and floor_timers[lost_track_id] is not None:
                    print(f"ID {int(lost_track_id)} 캘리브 중 이탈. 리셋.")
                
                for d in [base_data, kick_counters, floor_timers, floor_y_history, track_id_to_player_id, 
                            l_kick_state, r_kick_state, l_reset_counter, r_reset_counter, person_kick_timer]:
                    if lost_track_id in d: del d[lost_track_id]

            # --- [최적화 6/10] 9. UI 그리기 (모든 텍스트를 마지막에 한번에) ---
            if fonts:
                # [최적화 7/10] OpenCV 프레임을 PIL 이미지로 한번만 변환
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                # [★★★ 사용자 요청 수정 3/5 ★★★]
                # 짙은 파란색과 흰색 정의
                dark_blue_rgb = (65, 105, 225)  # 짙은 파란색 (RoyalBlue)
                white_rgb = (255, 255, 255)     # 흰색
                
                # ESC UI (우상단 원형)
                esc_text = "ESC"
                esc_font = fonts['ui_main'] # 24pt 폰트
                
                radius = 35 # 원의 반지름
                margin = 30 # 화면 우상단 모서리와의 여백
                center_x = frame_width - radius - margin
                center_y = radius + margin
                
                # 1. 짙은 파란색 원 그리기
                draw_final.ellipse([(center_x - radius, center_y - radius), 
                                    (center_x + radius, center_y + radius)], fill=dark_blue_rgb)
                
                # 2. "ESC" 텍스트 크기 계산
                if hasattr(draw_final, 'textbbox'):
                    bbox = draw_final.textbbox((0,0), esc_text, font=esc_font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    text_w, text_h = draw_final.textsize(esc_text, font=esc_font)
                
                # 3. 텍스트를 원 중앙에 그리기 (흰색 글씨)
                text_pos = (center_x - text_w // 2, center_y - text_h // 2 - 2) # Y축 미세조정
                draw_final.text(text_pos, esc_text, font=esc_font, fill=white_rgb)

                # [최적화 8/10] 메인 UI 텍스트 그리기 (좌상단, 좌하단)
                # [★★★ 사용자 요청 수정 4/5 ★★★] show_debug_ui 변수로 토글
                if show_debug_ui:
                    # (x, y) 좌표는 좌상단(top-left) 기준
                    draw_pil_text_on_image(draw_final, f"Ankle Height (Y): {KICK_THRESH_PIXELS_Y}px (Up/Down)", (10, 30), fonts['ui_main'], (255, 0, 0), (0,0,0)) # BGR(0,0,255) -> RGB(255,0,0)
                    draw_pil_text_on_image(draw_final, f"Ankle Dist (X): {KICK_THRESH_PIXELS_X}px (Left/Right)", (10, 60), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Z-Est Ratio: {KICK_THRESH_RATIO_Z*100:.0f}% (PgUp/PgDn)", (10, 90), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Cooldown: {KICK_COOLDOWN_FRAMES}f (Home/End)", (10, 120), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Box Margin (Y): {CALIB_BOX_BOTTOM_MARGIN}px (8/2 keys)", (10, 150), fonts['ui_main'], (255, 0, 0), (0,0,0))
                else:
                    # UI가 꺼져있을 때 힌트 버튼 표시
                    v_text = "V"
                    v_font = fonts['ui_main'] # 24pt 폰트
                    radius_v = 35
                    margin_v = 30
                    center_x_v = radius_v + margin_v
                    center_y_v = radius_v + margin_v

                    # 1. 짙은 파란색 원 그리기
                    draw_final.ellipse([(center_x_v - radius_v, center_y_v - radius_v), 
                                        (center_x_v + radius_v, center_y_v + radius_v)], fill=dark_blue_rgb)
                    
                    # 2. "V" 텍스트 크기 계산
                    if hasattr(draw_final, 'textbbox'):
                        bbox_v = draw_final.textbbox((0,0), v_text, font=v_font)
                        text_w_v, text_h_v = bbox_v[2] - bbox_v[0], bbox_v[3] - bbox_v[1]
                    else:
                        text_w_v, text_h_v = draw_final.textsize(v_text, font=v_font)
                    
                    # 3. 텍스트를 원 중앙에 그리기 (흰색 글씨)
                    text_pos_v = (center_x_v - text_w_v // 2, center_y_v - text_h_v // 2 - 2)
                    draw_final.text(text_pos_v, v_text, font=v_font, fill=white_rgb)

                
                # [최적화 9/10] 루프에서 저장된 플레이어 UI 텍스트 그리기 (캘리브 게이지, ID, 카운트, 박스 텍스트)
                for (text, pos, font, txt_col, bg_col) in pil_draw_list:
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col)
                
                # [★★★ 사용자 요청 수정 5/5 ★★★]
                # 4. 최종 점수판 그리기 (우측 중앙)
                x_pos = frame_width - 250 # X 위치 (우측에서 250px)
                score_text = "== Final Scores =="
                
                # 텍스트 높이 계산
                if hasattr(draw_final, 'textbbox'):
                    bbox_score = draw_final.textbbox((0,0), score_text, font=fonts['ui_kick'])
                    text_h_score = bbox_score[3] - bbox_score[1]
                else:
                    _, text_h_score = draw_final.textsize(score_text, font=fonts['ui_kick'])
                
                # Y위치를 (화면중앙 - 점수판전체높이/2) 근사치로 계산
                sorted_scores = sorted([(int(id_num), count) for id_num, count in final_scores.items()])
                total_scores_height = text_h_score + (30 * len(sorted_scores)) # 30px 간격
                current_y_top = (frame_height // 2) - (total_scores_height // 2)
                
                # 점수판 그리기
                draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))
                
                for id_num, count in sorted_scores:
                    current_y_top += 30 # Y는 아래로 증가
                    text = f"Player {id_num} : {count}"
                    draw_pil_text_on_image(draw_final, text, (x_pos + 20, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))

                # [최적화 10/10] PIL 이미지를 다시 OpenCV BGR로 한번만 변환
                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
            

        # --- 최종 화면 표시 ---
        cv2.imshow('Kick Counter - Multi Person Tracking', display_frame)

    # === 메인 루프 끝 ===
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()