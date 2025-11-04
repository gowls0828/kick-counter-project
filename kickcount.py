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
    H, W = frame_dims[1], frame_dims[0]
    
    PLAYER_SELECT_ZONES = [
        (98, 303, 402, 902),   # 1인 구역
        (509, 303, 813, 902),  # 2인 구역
        (920, 303, 1224, 902), # 3인 구역
        (1331, 303, 1635, 902) # 4인 구역
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

# [★★★ 실루엣 가이드 추가 ★★★]
def overlay_transparent(background_img, overlay_rgba, x, y):
    """
    배경 이미지(BGR) 위에 투명한 오버레이 이미지(BGRA)를 합성합니다.
    x, y는 오버레이 이미지가 시작될 배경 이미지의 좌상단 좌표입니다.
    """
    try:
        h, w = overlay_rgba.shape[:2]
        bg_h, bg_w = background_img.shape[:2]

        if x < 0: w += x; x = 0
        if y < 0: h += y; y = 0
        if x + w > bg_w: w = bg_w - x
        if y + h > bg_h: h = bg_h - y
        
        if w <= 0 or h <= 0:
            return background_img

        overlay_rgba_cropped = overlay_rgba[0:h, 0:w]

        overlay_bgr = overlay_rgba_cropped[:,:,0:3]
        alpha = overlay_rgba_cropped[:,:,3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        roi = background_img[y:y+h, x:x+w]
        blended_roi = (overlay_bgr * alpha) + (roi * (1.0 - alpha))
        background_img[y:y+h, x:x+w] = blended_roi.astype(np.uint8)

        return background_img
    except Exception as e:
        return background_img


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
    
    # --- 상태 변수 (UI 통합) ---
    current_state = MENU_START 
    max_players = 0 

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
    KICK_THRESH_PIXELS_X = 1 
    KICK_THRESH_RATIO_Z = 0.10 # 10%

    JOINT_CONF_THRESH = 0.1 # 10%
    
    CALIB_BOX_BOTTOM_MARGIN = 50 
    
    active_calib_boxes = {} 

    show_debug_ui = False # 기본 비가시화

    calibrated_box_indices = set()

    MODEL_INPUT_W = 640
    MODEL_INPUT_H = 360

    # [★★★ 게임 모드 추가 ★★★] 게임 모드용 상태 변수
    game_mode_start_time = 0.0
    game_mode_stage = 0 # 0: 대기, 1: 설명, 2: 3, 3: 2, 4: 1, 5: 시작
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

    mouse_clicked = False
    mouse_pos = (0, 0) 
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_clicked, mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True

    WIN_NAME = 'Kick Counter - Multi Person Tracking'
    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)

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
            'ui_percent': ImageFont.truetype(FONT_PATH, 20),
            'game_button': ImageFont.truetype(FONT_PATH, 30) # [★★★ 게임 종료 추가 ★★★]
        }
    except IOError:
        print(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
        fonts = None # 폰트 로드 실패

    # [★★★ 사용자 요청 수정 1/4 ★★★]
    # 모든 이미지 미리 로드
    splash_image_resized = None
    player_select_image_resized = None
    guide_img_resized = None
    countdown_imgs = {}
    game_instructions_img = None
    instructions_normal_img = None # 일반 모드 설명서
    timer_bg_img = None
    timer_fg_img = None
    
    try:
        # 1. 스플래시 이미지 로드
        splash_image_path = resource_path(os.path.join('image', 'splash.png'))
        splash_image = cv2.imread(splash_image_path)
        if splash_image is None: raise FileNotFoundError("image/splash.png")
        splash_image_resized = cv2.resize(splash_image, (frame_width, frame_height))
        print("스플래시 이미지 로드 성공.")
        
        # 2. 인원 선택 이미지 로드
        player_select_image_path = resource_path(os.path.join('image', 'player_select.png'))
        player_select_image = cv2.imread(player_select_image_path)
        if player_select_image is None: raise FileNotFoundError("image/player_select.png")
        player_select_image_resized = cv2.resize(player_select_image, (frame_width, frame_height))
        print("인원 선택 이미지 로드 성공.")
        
        # 3. 가이드 실루엣 이미지 로드
        guide_image_path = resource_path(os.path.join('image', 'guide.png'))
        guide_img_rgba = cv2.imread(guide_image_path, cv2.IMREAD_UNCHANGED)
        if guide_img_rgba is None: raise FileNotFoundError("image/guide.png")
        
        guide_h, guide_w = guide_img_rgba.shape[:2]
        scale = (frame_height * 0.7) / guide_h # 화면 높이의 70%
        new_w = int(guide_w * scale)
        new_h = int(frame_height * 0.7)
        guide_img_resized = cv2.resize(guide_img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print("가이드 실루엣 이미지 로드 및 리사이즈 성공.")

        # 4. 카운트다운 이미지 로드
        for i in [1, 2, 3]:
            img_path = resource_path(os.path.join('image', f'countdown_{i}.png'))
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: raise FileNotFoundError(f"image/countdown_{i}.png")
            countdown_imgs[i] = img
        print("카운트다운 이미지 로드 성공.")

        # 5. 게임 설명 텍스트 이미지 로드
        img_path = resource_path(os.path.join('image', 'game_instructions.png'))
        game_instructions_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if game_instructions_img is None: raise FileNotFoundError("image/game_instructions.png")
        print("게임 설명 이미지 로드 성공.")

        # 6. 타이머 바 이미지 로드
        img_path = resource_path(os.path.join('image', 'timer_bg.png'))
        timer_bg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 시계 아이콘
        if timer_bg_img is None: raise FileNotFoundError("image/timer_bg.png")
        
        img_path = resource_path(os.path.join('image', 'timer_fg.png'))
        timer_fg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 시계 + 초록색 바
        if timer_fg_img is None: raise FileNotFoundError("image/timer_fg.png")
        print("타이머 바 이미지 로드 성공.")
        
        # 7. 일반 모드 설명 텍스트 이미지 로드
        img_path = resource_path(os.path.join('image', 'instructions_normal.png'))
        instructions_normal_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if instructions_normal_img is None: raise FileNotFoundError("image/instructions_normal.png")
        print("일반 모드 설명 이미지 로드 성공.")

    except Exception as e:
        print(f"경고: 필수 이미지 로드 실패. {e}")
        # 실패 시 검은 배경을 비상용으로 사용
        if splash_image_resized is None:
            splash_image_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if player_select_image_resized is None:
            player_select_image_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # (다른 이미지들은 None으로 유지)


    # === 3. 메인 루프 (프로그램 실행) ===
    tracks = [] 
    
    active_hover_zone = -1 
    
    radius = 35 
    margin = 30
    center_x_esc = frame_width - radius - margin
    center_y_esc = radius + margin
    ESC_BUTTON_ZONE = (center_x_esc - radius, center_y_esc - radius, center_x_esc + radius, center_y_esc + radius)
    
    center_x_v = radius + margin
    center_y_v = radius + margin
    V_BUTTON_ZONE = (center_x_v - radius, center_y_v - radius, center_x_v + radius, center_y_v + radius)
    
    center_x_game = frame_width - radius - margin
    center_y_game = center_y_esc + radius + radius + 10 # ESC 버튼 아래
    GAME_BUTTON_ZONE = (center_x_game - radius, center_y_game - radius, center_x_game + radius, center_y_game + radius)
    
    btn_w, btn_h = 300, 80
    btn_y = frame_height // 2 + 200
    btn_restart_x1 = (frame_width // 2) - btn_w - 20
    btn_restart_y1 = btn_y
    RESTART_BUTTON_ZONE = (btn_restart_x1, btn_restart_y1, btn_restart_x1 + btn_w, btn_restart_y1 + btn_h)
    
    btn_normal_x1 = (frame_width // 2) + 20
    btn_normal_y1 = btn_y
    NORMAL_MODE_BUTTON_ZONE = (btn_normal_x1, btn_normal_y1, btn_normal_x1 + btn_w, btn_normal_y1 + btn_h)


    while True:
        ret, frame = cap.read()
        if not ret: break
            
        display_frame = frame.copy()

        key = cv2.waitKeyEx(1)
        
        # -----------------------------------------------------
        # A. UI 및 상태 전환 로직
        # -----------------------------------------------------

        if current_state == MENU_START:
            display_frame, _ = draw_menu_ui(MENU_START, 
                                            frame_dims=(frame_width, frame_height), 
                                            splash_img=splash_image_resized)
            
            if key == ord('q'):
                break 
            
            elif (key != -1 and key != 255) or mouse_clicked:
                current_state = MENU_PLAYER_SELECT
                mouse_clicked = False 
        
        elif current_state == MENU_PLAYER_SELECT:
            display_frame, active_hover_zone = draw_menu_ui(MENU_PLAYER_SELECT, 
                                                            frame_dims=(frame_width, frame_height), 
                                                            fonts=fonts, 
                                                            mouse_pos=mouse_pos, 
                                                            player_select_img=player_select_image_resized)
            
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
            
            pil_draw_list = [] 

            temp_pil_img = Image.new("RGB", (1,1))
            temp_draw = ImageDraw.Draw(temp_pil_img)

            calibration_boxes = get_calibration_boxes(max_players, frame_width, frame_height, CALIB_BOX_BOTTOM_MARGIN)
            
            # [★★★ 사용자 요청 수정 2/4 ★★★]
            # 일반 모드 설명 텍스트 (아직 캘리브 안 끝났으면)
            if instructions_normal_img is not None and len(calibrated_box_indices) < max_players:
                h, w = instructions_normal_img.shape[:2]
                x_pos = (frame_width - w) // 2
                y_pos = 50 # 상단에 위치
                # 검은색 배경 그리기 (BGR)
                cv2.rectangle(display_frame, (x_pos - 5, y_pos - 5), (x_pos + w + 5, y_pos + h + 5), (0,0,0), -1)
                # 그 위에 텍스트 이미지 오버레이
                overlay_transparent(display_frame, instructions_normal_img, x_pos, y_pos)

            
            for i, box in enumerate(calibration_boxes):
                x1, y1, x2, y2 = box
                
                is_being_calibrated = (i in active_calib_boxes)
                is_calibrated = (i in calibrated_box_indices)

                if is_being_calibrated:
                    color = (0, 255, 0) 
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                elif is_calibrated:
                    color = (0, 0, 255) 
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                else:
                    color = (0, 0, 255) 
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if guide_img_resized is not None:
                        guide_h, guide_w = guide_img_resized.shape[:2]
                        box_center_x = x1 + (x2 - x1) // 2
                        paste_x = box_center_x - (guide_w // 2)
                        paste_y = y2 - guide_h - 5
                        overlay_transparent(display_frame, guide_img_resized, paste_x, paste_y)

                    text = "여기 서주세요"
                    font = fonts['subtitle'] 
                    
                    if hasattr(temp_draw, 'textbbox'):
                        bbox = temp_draw.textbbox((0,0), text, font=font)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        text_w, text_h = temp_draw.textsize(text, font=font)
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    text_x = x1 + (box_width - text_w) // 2
                    text_y = y1 + (box_height - text_h) // 2
                    
                    pil_draw_list.append( (text, (text_x, text_y), font, (0, 0, 255), None) ) # 파란색

            # --- 사람 감지 및 ID 추적 ---
            small_frame = cv2.resize(frame, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_AREA)
            results = model(small_frame, conf=0.6, verbose=False, device='cpu') 
            scale_x = frame_width / MODEL_INPUT_W
            scale_y = frame_height / MODEL_INPUT_H

            dets, keypoints_list = [], []
            for r in results:
                if r.keypoints is not None:
                    xy_data, conf_data = getattr(r.keypoints, "xy", []), getattr(r.keypoints, "conf", [])
                    if len(xy_data) != len(conf_data): continue
                    for i in range(len(xy_data)):
                        person_kp, person_conf = xy_data[i].cpu().numpy(), conf_data[i].cpu().numpy()
                        person_kp[:, 0] *= scale_x
                        person_kp[:, 1] *= scale_y
                        valid_kps = person_kp[person_kp[:, 1] > 10]
                        if len(valid_kps) == 0: continue
                        min_x, max_x = np.min(valid_kps[:,0]), np.max(valid_kps[:,0])
                        min_y, max_y = np.min(valid_kps[:,1]), np.max(valid_kps[:,1])
                        dets.append([min_x, min_y, max_x, max_y, 1.0])
                        keypoints_list.append((person_kp, person_conf))
                        
            dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
            
            if len(dets) > max_players:
                sorted_indices = sorted(range(len(dets)), key=lambda k: (dets[k][2]-dets[k][0]) * (dets[k][3]-dets[k][1]), reverse=True)
                dets_to_track = np.array([dets[i] for i in sorted_indices[:max_players]])
                keypoints_list_to_track = [keypoints_list[i] for i in sorted_indices[:max_players]]
            else:
                dets_to_track = dets
                keypoints_list_to_track = keypoints_list
                
            tracks = tracker.update(dets_to_track) 

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
            elif key == ord('8'): CALIB_BOX_BOTTOM_MARGIN += 5
            elif key == ord('2'): CALIB_BOX_BOTTOM_MARGIN = max(5, CALIB_BOX_BOTTOM_MARGIN - 5)
            elif key == ord('v'): show_debug_ui = not show_debug_ui
            elif key == 27: 
                current_state = MENU_PLAYER_SELECT 
                base_data.clear(); final_scores.clear(); kick_counters.clear()
                l_kick_state.clear(); r_kick_state.clear(); l_reset_counter.clear()
                r_reset_counter.clear(); person_kick_timer.clear(); 
                floor_timers.clear(); floor_y_history.clear();
                track_id_to_player_id.clear(); player_id_counter = 1
                active_calib_boxes.clear(); calibrated_box_indices.clear() 
                print("인원 선택 화면으로 복귀. 모든 데이터 초기화.")
                continue 

            if mouse_clicked and is_point_in_box(mouse_pos, GAME_BUTTON_ZONE):
                current_state = GAME_COUNTDOWN
                game_mode_start_time = 0.0 
                game_mode_stage = 0 
                kick_counters.clear() 
                final_scores.clear()  
                mouse_clicked = False
                print("게임 모드 진입. 스페이스바 대기 중...")
                continue 

            # --- ID별 로직 처리 ---
            active_track_ids = set()
            matched = set()
            active_calib_boxes.clear()
            
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
                        l_ankle_xy, r_ankle_xy = person_xy[15][:2], person_xy[16][:2]
                        is_in_any_box = False
                        current_box_index = -1 
                        for i, box in enumerate(calibration_boxes):
                            if is_point_in_box(l_ankle_xy, box) or is_point_in_box(r_ankle_xy, box):
                                if i in calibrated_box_indices: continue
                                is_in_any_box = True
                                active_calib_boxes[i] = track_id 
                                current_box_index = i 
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
                        
                        if is_in_any_box and is_full_body_visible and base_height > 100 and is_standing and is_facing_front and is_standing_aspect_ratio:
                            current_time = time.time()
                            if floor_timers[track_id] is None:
                                floor_timers[track_id] = current_time
                                floor_y_history[track_id] = [foot_y]
                                base_data[track_id] = {"box_index": current_box_index} 
                            else:
                                floor_y_history[track_id].append(foot_y)
                                elapsed = current_time - floor_timers[track_id]
                                if elapsed > CALIBRATION_TIME:
                                    history = floor_y_history[track_id]
                                    y_movement = np.max(history) - np.min(history)
                                    if y_movement < STABILITY_THRESH:
                                        temp_data = base_data.get(track_id, {})
                                        box_index_to_calibrate = temp_data.get("box_index", -1)
                                        if box_index_to_calibrate != -1:
                                            calibrated_box_indices.add(box_index_to_calibrate)

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
                                            "Y_RST": Y_RST, "Z_RST_EST_L": Z_RST_EST_L, "Z_RST_EST_R": Z_RST_EST_R,
                                            "box_index": box_index_to_calibrate 
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
                                        if track_id in base_data: del base_data[track_id] 
                        else:
                            floor_timers[track_id], floor_y_history[track_id] = None, []
                            if track_id in base_data: del base_data[track_id] 
                    else:
                        floor_timers[track_id], floor_y_history[track_id] = None, []
                    
                    # 캘리브레이션 게이지 UI
                    if is_head_visible and floor_timers[track_id] is not None and fonts:
                        ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                        elapsed = time.time() - floor_timers[track_id]
                        angle = min((elapsed / CALIBRATION_TIME) * 360, 360)
                        
                        cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, (100,100,100), 2)
                        cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, -90, int(angle-90), (0,255,255), 4)
                        
                        text_percent = f'{min(int(angle/3.6), 100)}%'
                        
                        if hasattr(temp_draw, 'textbbox'):
                            bbox = temp_draw.textbbox((0,0), text_percent, font=fonts['ui_percent'])
                            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        else:
                            text_w, text_h = temp_draw.textsize(text_percent, font=fonts['ui_percent'])
                        
                        text_pos = (ui_center_x - text_w // 2, ui_center_y - text_h // 2 - 2) 
                        pil_draw_list.append( (text_percent, text_pos, fonts['ui_percent'], (255,255,255), None) ) 

                    # 6. 킥 카운트 단계
                    else:
                        if track_id not in base_data:
                            continue 
                        
                        if "base_height" not in base_data[track_id]:
                            continue
                            
                        pd = base_data[track_id] 
                        
                        is_too_close = (current_bbox_height > pd['base_bbox_height'] * 2.0) or (current_bbox_width > pd['base_bbox_width'] * 2.0) 

                        if is_too_close:
                            final_scores[track_id_to_player_id.get(track_id, 0)] = kick_counters[track_id]; del base_data[track_id]; print(f"ID {track_id} (Player {track_id_to_player_id.get(track_id, '?')}) 너무 가까움. 리셋.")
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
                            Y_MID, X_MID = pd['Y_MID'], pd['X_MID']
                            Y_HIGH = pd['Y_HIGH']; Y_RST = pd['Y_RST']
                            Z_MID_EST_L, Z_MID_EST_R = pd['Z_MID_EST_L'], pd['Z_MID_EST_R']
                            Z_HIGH_EST_L, Z_HIGH_EST_R = pd['Z_HIGH_EST_L'], pd['Z_HIGH_EST_R']
                            Z_RST_EST_L, Z_RST_EST_R = pd['Z_RST_EST_L'], pd['Z_RST_EST_R']
                            player_id = track_id_to_player_id.get(track_id, '?')
                            kick_detected_this_frame = False
                            if person_kick_timer[track_id] > 0: person_kick_timer[track_id] -= 1
                            
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
                        
                        # 킥 카운트 UI
                        if is_head_visible and fonts and track_id in base_data: 
                            ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                            player_id = track_id_to_player_id.get(track_id, "?")
                            text_count = f'{player_id}'
                            color_bgr = (0, 100, 255) if person_kick_timer[track_id] > 0 else (0, 255, 0) 
                            cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, color_bgr, -1) 
                            
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

            # --- 8. 사라진 ID 처리 ---
            active_ids_set = set(t[4] for t in tracks)
            tracked_ids = set(base_data.keys()) | set(k for k,v in floor_timers.items() if v is not None)
            lost_ids = tracked_ids - active_ids_set
            
            for lost_track_id in lost_ids:
                if lost_track_id in base_data and "box_index" in base_data[lost_track_id]:
                    box_idx = base_data[lost_track_id].get("box_index", -1)
                    if box_idx != -1 and box_idx in calibrated_box_indices:
                        calibrated_box_indices.remove(box_idx)
                
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

            # --- 9. UI 그리기 (텍스트) ---
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                dark_blue_rgb = (0, 0, 205)
                white_rgb = (255, 255, 255) 
                
                # ESC UI (우상단 원형)
                esc_text = "ESC"
                esc_font = fonts['ui_main'] 
                radius = 35 
                margin = 30
                center_x = frame_width - radius - margin
                center_y = radius + margin
                draw_final.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox = draw_final.textbbox((0,0), esc_text, font=esc_font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    text_w, text_h = draw_final.textsize(esc_text, font=esc_font)
                text_pos = (center_x - text_w // 2, center_y - text_h // 2 - 2)
                draw_final.text(text_pos, esc_text, font=esc_font, fill=white_rgb)

                # "GAME" 버튼 그리기
                game_text = "GAME"
                game_font = fonts['ui_player'] # 22pt
                radius_g = 35
                center_x_g = frame_width - radius_g - margin
                center_y_g = center_y + radius + radius_g + 10 # ESC 버튼 아래
                draw_final.ellipse([(center_x_g - radius_g, center_y_g - radius_g), 
                                    (center_x_g + radius_g, center_y_g + radius_g)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox_g = draw_final.textbbox((0,0), game_text, font=game_font)
                    text_w_g, text_h_g = bbox_g[2] - bbox_g[0], bbox_g[3] - bbox_g[1]
                else:
                    text_w_g, text_h_g = draw_final.textsize(game_text, font=game_font)
                text_pos_g = (center_x_g - text_w_g // 2, center_y_g - text_h_g // 2 - 2)
                draw_final.text(text_pos_g, game_text, font=game_font, fill=white_rgb)

                # V-Key UI 토글
                if show_debug_ui:
                    draw_pil_text_on_image(draw_final, f"Ankle Height (Y): {KICK_THRESH_PIXELS_Y}px (Up/Down)", (10, 30), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Ankle Dist (X): {KICK_THRESH_PIXELS_X}px (Left/Right)", (10, 60), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Z-Est Ratio: {KICK_THRESH_RATIO_Z*100:.0f}% (PgUp/PgDn)", (10, 90), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Cooldown: {KICK_COOLDOWN_FRAMES}f (Home/End)", (10, 120), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Box Margin (Y): {CALIB_BOX_BOTTOM_MARGIN}px (8/2 keys)", (10, 150), fonts['ui_main'], (255, 0, 0), (0,0,0))
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
                for (text, pos, font, txt_col, bg_col) in pil_draw_list:
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col)
                
                # 최종 점수판 그리기 (우측 중앙)
                x_pos = frame_width - 250 
                score_text = "== Final Scores =="
                
                if hasattr(draw_final, 'textbbox'):
                    bbox_score = draw_final.textbbox((0,0), score_text, font=fonts['ui_kick'])
                    text_h_score = bbox_score[3] - bbox_score[1]
                else:
                    _, text_h_score = draw_final.textsize(score_text, font=fonts['ui_kick'])
                
                # [수정] 점수(item[1]) 기준으로 내림차순 정렬 (랭킹)
                sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

                total_scores_height = text_h_score + (30 * len(sorted_scores))
                current_y_top = (frame_height // 2) - (total_scores_height // 2)
                
                draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))
                
                for id_num, count in sorted_scores:
                    current_y_top += 30 
                    text = f"Player {id_num} : {count}"
                    draw_pil_text_on_image(draw_final, text, (x_pos + 20, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))

                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
            
        # [★★★ 게임 모드 추가 ★★★]
        # C. 게임 카운트다운 상태
        # -----------------------------------------------------
        elif current_state == GAME_COUNTDOWN:
            display_frame = frame.copy() 
            
            if game_mode_stage == 0:
                # "스페이스바를 눌러 게임을 시작하세요"
                if fonts:
                    img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    draw_final = ImageDraw.Draw(img_pil_final)
                    text = "스페이스바를 눌러 게임을 시작하세요"
                    draw_pil_text_on_image(draw_final, text, (frame_width // 2, 50), fonts['subtitle'], (255, 255, 0), (0,0,0), align="center")
                    display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
                
                if key == ord(' '): # 스페이스바
                    game_mode_stage = 1
                    game_mode_start_time = time.time()
                    print("게임 카운트다운 시작...")
            
            else:
                elapsed = time.time() - game_mode_start_time
                
                # 1. 상단 텍스트 그리기
                if game_instructions_img is not None:
                    h, w = game_instructions_img.shape[:2]
                    x_pos = (frame_width - w) // 2
                    y_pos = 50
                    
                    # [★★★ 사용자 요청 수정 ★★★] 검은색 배경 추가
                    cv2.rectangle(display_frame, (x_pos - 5, y_pos - 5), (x_pos + w + 5, y_pos + h + 5), (0,0,0), -1)
                    
                    overlay_transparent(display_frame, game_instructions_img, x_pos, y_pos)

                # 2. 카운트다운 숫자 그리기
                number_to_show = 0
                if elapsed < 1.0:
                    number_to_show = 3
                elif elapsed < 2.0:
                    number_to_show = 2
                elif elapsed < 3.0:
                    number_to_show = 1
                elif elapsed >= 3.0:
                    current_state = GAME_TIMER_RUNNING
                    game_mode_start_time = time.time() # 30초 타이머 시작
                    kick_counters.clear() 
                    final_scores.clear()
                    print("게임 시작!")
                    continue
                
                if number_to_show > 0 and countdown_imgs.get(number_to_show) is not None:
                    img_num = countdown_imgs[number_to_show]
                    h, w = img_num.shape[:2]
                    x_pos = (frame_width - w) // 2
                    y_pos = (frame_height - h) // 2
                    overlay_transparent(display_frame, img_num, x_pos, y_pos)

            if key == ord('q'): break
            if key == 27:
                current_state = MENU_PLAYER_SELECT
                calibrated_box_indices.clear()
                print("인원 선택 화면으로 복귀.")
                continue
                
        # [★★★ 게임 모드 추가 ★★★]
        # D. 게임 타이머 실행 상태
        # -----------------------------------------------------
        elif current_state == GAME_TIMER_RUNNING:
            
            # --- 1. 킥 인식 로직 (GAME_RUNNING과 동일) ---
            pil_draw_list = []
            temp_pil_img = Image.new("RGB", (1,1))
            temp_draw = ImageDraw.Draw(temp_pil_img)
            
            small_frame = cv2.resize(frame, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_AREA)
            # [★★★ YOLO 자동 감지 ★★★] device 파라미터 제거
            results = model(small_frame, conf=0.6, verbose=False) 
            scale_x = frame_width / MODEL_INPUT_W
            scale_y = frame_height / MODEL_INPUT_H
            
            dets, keypoints_list = [], []
            for r in results:
                if r.keypoints is not None:
                    xy_data, conf_data = getattr(r.keypoints, "xy", []), getattr(r.keypoints, "conf", [])
                    if len(xy_data) != len(conf_data): continue
                    for i in range(len(xy_data)):
                        person_kp, person_conf = xy_data[i].cpu().numpy(), conf_data[i].cpu().numpy()
                        person_kp[:, 0] *= scale_x
                        person_kp[:, 1] *= scale_y
                        valid_kps = person_kp[person_kp[:, 1] > 10]
                        if len(valid_kps) == 0: continue
                        min_x, max_x = np.min(valid_kps[:,0]), np.max(valid_kps[:,0])
                        min_y, max_y = np.min(valid_kps[:,1]), np.max(valid_kps[:,1])
                        dets.append([min_x, min_y, max_x, max_y, 1.0])
                        keypoints_list.append((person_kp, person_conf))
                        
            dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
            
            if len(dets) > max_players:
                sorted_indices = sorted(range(len(dets)), key=lambda k: (dets[k][2]-dets[k][0]) * (dets[k][3]-dets[k][1]), reverse=True)
                dets_to_track = np.array([dets[i] for i in sorted_indices[:max_players]])
                keypoints_list_to_track = [keypoints_list[i] for i in sorted_indices[:max_players]]
            else:
                dets_to_track = dets
                keypoints_list_to_track = keypoints_list
                
            tracks = tracker.update(dets_to_track) 
            
            if key == ord('q'): break
            if key == 27:
                current_state = MENU_PLAYER_SELECT
                calibrated_box_indices.clear()
                print("인원 선택 화면으로 복귀.")
                continue

            active_track_ids = set()
            matched = set()
            active_calib_boxes.clear() 
            
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
                    
                    head_conf = person_conf[0]
                    is_head_visible = head_conf >= JOINT_CONF_THRESH
                    
                    if track_id not in base_data: continue
                    if "base_height" not in base_data[track_id]: continue
                            
                    pd = base_data[track_id] 
                    
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
                    Y_MID, X_MID = pd['Y_MID'], pd['X_MID']
                    Y_HIGH = pd['Y_HIGH']; Y_RST = pd['Y_RST']
                    Z_MID_EST_L, Z_MID_EST_R = pd['Z_MID_EST_L'], pd['Z_MID_EST_R']
                    Z_HIGH_EST_L, Z_HIGH_EST_R = pd['Z_HIGH_EST_L'], pd['Z_HIGH_EST_R']
                    Z_RST_EST_L, Z_RST_EST_R = pd['Z_RST_EST_L'], pd['Z_RST_EST_R']
                    player_id = track_id_to_player_id.get(track_id, '?')
                    kick_detected_this_frame = False
                    if person_kick_timer[track_id] > 0: person_kick_timer[track_id] -= 1
                    
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
                    
                    # 킥 카운트 UI
                    if is_head_visible and fonts and track_id in base_data: 
                        ui_center_x, ui_center_y, radius = int(head_x), int(head_y) - 60, 30
                        player_id = track_id_to_player_id.get(track_id, "?")
                        text_count = f'{player_id}'
                        color_bgr = (0, 100, 255) if person_kick_timer[track_id] > 0 else (0, 255, 0) 
                        cv2.ellipse(display_frame, (ui_center_x, ui_center_y), (radius, radius), 0, 0, 360, color_bgr, -1) 
                        
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

            # --- 2. 타이머 UI 그리기 ---
            elapsed = time.time() - game_mode_start_time
            time_left = GAME_DURATION_SECONDS - elapsed
            progress = elapsed / GAME_DURATION_SECONDS

            if time_left <= 0:
                # 30초 종료!
                print("게임 종료!")
                # [★★★ 게임 종료 추가 ★★★]
                # 점수를 ID 기반이 아닌, 캘리브레이션 된 플레이어 ID 기반으로 저장
                final_scores.clear()
                for track_id, player_id in track_id_to_player_id.items():
                    if track_id in kick_counters:
                        final_scores[player_id] = kick_counters[track_id]
                
                current_state = GAME_OVER # 게임 종료 상태로 전환
                mouse_clicked = False # 혹시 모를 클릭 방지
                continue

            if timer_bg_img is not None and timer_fg_img is not None:
                # [★★★ 타이머 바 수정 ★★★]
                # 1. 타이머 바 배경(시계) 그리기
                h_bg, w_bg = timer_bg_img.shape[:2]
                h_fg, w_fg = timer_fg_img.shape[:2]
                
                # 전체 UI (시계 + 꽉찬 바) 기준으로 중앙 정렬
                x_pos = (frame_width - w_fg) // 2
                y_pos = 50
                overlay_transparent(display_frame, timer_bg_img, x_pos, y_pos)

                # 2. 타이머 바 채우기 (크롭)
                clock_width = h_bg # 시계 아이콘의 너비 (정사각형 가정)
                total_bar_width = w_fg - clock_width # 초록색 바 영역의 전체 너비
                
                fill_width = int(total_bar_width * progress) # 채워야 할 너비

                if fill_width > 0:
                    # 3. '시계+초록바' 이미지에서 '초록바' 부분만 잘라냄
                    bar_crop = timer_fg_img[:, clock_width : clock_width + fill_width]
                    
                    # 4. 시계 아이콘 *옆*에(x_pos + clock_width) 붙여넣기
                    overlay_transparent(display_frame, bar_crop, x_pos + clock_width, y_pos)


            # --- 3. PIL 텍스트 그리기 (카운터) ---
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)
                
                # 플레이어 카운트 그리기
                for (text, pos, font, txt_col, bg_col) in pil_draw_list:
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col)
                
                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)

        # [★★★ 게임 종료 추가 ★★★]
        # E. 게임 종료 (결과) 상태
        # -----------------------------------------------------
        elif current_state == GAME_OVER:
            # 배경은 카메라 원본을 계속 보여줌
            display_frame = frame.copy()
            
            # --- 1. 키/마우스 입력 처리 ---
            if key == ord('q'): break
            if key == 27: # ESC
                current_state = MENU_PLAYER_SELECT
                calibrated_box_indices.clear()
                print("인원 선택 화면으로 복귀.")
                continue
            
            if mouse_clicked:
                if is_point_in_box(mouse_pos, RESTART_BUTTON_ZONE):
                    # 리스타트
                    current_state = GAME_COUNTDOWN
                    game_mode_start_time = 0.0
                    game_mode_stage = 0 # 스페이스바 대기
                    kick_counters.clear()
                    final_scores.clear()
                    mouse_clicked = False
                    print("게임 모드 재시작. 스페이스바 대기 중...")
                    continue
                elif is_point_in_box(mouse_pos, NORMAL_MODE_BUTTON_ZONE):
                    # 일반 모드
                    current_state = GAME_RUNNING
                    kick_counters.clear()
                    final_scores.clear()
                    mouse_clicked = False
                    print("일반 모드로 복귀.")
                    continue
            
            mouse_clicked = False # 버튼 안 눌렀으면 초기화

            # --- 2. UI 그리기 (OpenCV) ---
            # 반투명 검은색 배경
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0,0,0), -1)
            display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
            
            # 버튼 그리기 (BGR 색상)
            cv2.rectangle(display_frame, (RESTART_BUTTON_ZONE[0], RESTART_BUTTON_ZONE[1]), (RESTART_BUTTON_ZONE[2], RESTART_BUTTON_ZONE[3]), (0, 200, 0), -1) # 초록색
            cv2.rectangle(display_frame, (NORMAL_MODE_BUTTON_ZONE[0], NORMAL_MODE_BUTTON_ZONE[1]), (NORMAL_MODE_BUTTON_ZONE[2], NORMAL_MODE_BUTTON_ZONE[3]), (205, 0, 0), -1) # 짙은 파란색
            
            # --- 3. UI 그리기 (PIL 텍스트) ---
            if fonts:
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                white_rgb = (255, 255, 255)
                yellow_rgb = (0, 255, 255) # BGR(255,255,0) -> RGB(0,255,255)
                
                # 1. "Final Scores" 랭킹 그리기
                x_pos = frame_width // 2 # 중앙 정렬
                score_text = "== Final Scores =="
                font_title = fonts['subtitle']
                font_score = fonts['ui_main']
                
                if hasattr(draw_final, 'textbbox'):
                    bbox_score = draw_final.textbbox((0,0), score_text, font=font_title)
                    text_h_score = bbox_score[3] - bbox_score[1]
                else:
                    _, text_h_score = draw_final.textsize(score_text, font=font_title)
                
                # [수정] 점수(item[1]) 기준으로 내림차순 정렬 (랭킹)
                sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
                
                total_scores_height = text_h_score + (40 * len(sorted_scores)) # 40px 간격
                current_y_top = (frame_height // 2) - (total_scores_height // 2) - 50 # 중앙보다 살짝 위
                
                # "Final Scores" 타이틀
                draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), font_title, yellow_rgb, (0,0,0), align="center")
                
                current_y_top += 60 # 타이틀 아래 간격
                
                # 랭킹 목록
                for rank, (id_num, count) in enumerate(sorted_scores):
                    text = f"{rank + 1}위 - Player {id_num} : {count} 회"
                    draw_pil_text_on_image(draw_final, text, (x_pos, current_y_top), font_score, white_rgb, None, align="center")
                    current_y_top += 40 
                
                # 2. 버튼 텍스트 그리기
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

        # -----------------------------------------------------
        # (GAME_RUNNING UI 그리기는 게임 종료/카운트다운 화면에선 안 그림)
        # -----------------------------------------------------
            if fonts and current_state == GAME_RUNNING: # [수정] GAME_RUNNING 일때만 그림
                img_pil_final = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw_final = ImageDraw.Draw(img_pil_final)

                dark_blue_rgb = (0, 0, 205)
                white_rgb = (255, 255, 255) 
                
                # ESC UI (우상단 원형)
                esc_text = "ESC"
                esc_font = fonts['ui_main'] 
                radius = 35 
                margin = 30
                center_x = frame_width - radius - margin
                center_y = radius + margin
                draw_final.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox = draw_final.textbbox((0,0), esc_text, font=esc_font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    text_w, text_h = draw_final.textsize(esc_text, font=esc_font)
                text_pos = (center_x - text_w // 2, center_y - text_h // 2 - 2)
                draw_final.text(text_pos, esc_text, font=esc_font, fill=white_rgb)

                # "GAME" 버튼 그리기
                game_text = "GAME"
                game_font = fonts['ui_player'] # 22pt
                radius_g = 35
                center_x_g = frame_width - radius_g - margin
                center_y_g = center_y + radius + radius_g + 10 # ESC 버튼 아래
                draw_final.ellipse([(center_x_g - radius_g, center_y_g - radius_g), 
                                    (center_x_g + radius_g, center_y_g + radius_g)], fill=dark_blue_rgb)
                if hasattr(draw_final, 'textbbox'):
                    bbox_g = draw_final.textbbox((0,0), game_text, font=game_font)
                    text_w_g, text_h_g = bbox_g[2] - bbox_g[0], bbox_g[3] - bbox_g[1]
                else:
                    text_w_g, text_h_g = draw_final.textsize(game_text, font=game_font)
                text_pos_g = (center_x_g - text_w_g // 2, center_y_g - text_h_g // 2 - 2)
                draw_final.text(text_pos_g, game_text, font=game_font, fill=white_rgb)

                # V-Key UI 토글
                if show_debug_ui:
                    draw_pil_text_on_image(draw_final, f"Ankle Height (Y): {KICK_THRESH_PIXELS_Y}px (Up/Down)", (10, 30), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Ankle Dist (X): {KICK_THRESH_PIXELS_X}px (Left/Right)", (10, 60), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Z-Est Ratio: {KICK_THRESH_RATIO_Z*100:.0f}% (PgUp/PgDn)", (10, 90), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Cooldown: {KICK_COOLDOWN_FRAMES}f (Home/End)", (10, 120), fonts['ui_main'], (255, 0, 0), (0,0,0))
                    draw_pil_text_on_image(draw_final, f"Box Margin (Y): {CALIB_BOX_BOTTOM_MARGIN}px (8/2 keys)", (10, 150), fonts['ui_main'], (255, 0, 0), (0,0,0))
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
                for (text, pos, font, txt_col, bg_col) in pil_draw_list:
                    draw_pil_text_on_image(draw_final, text, pos, font, txt_col, bg_col)
                
                # 최종 점수판 그리기 (우측 중앙)
                x_pos = frame_width - 250 
                score_text = "== Final Scores =="
                
                if hasattr(draw_final, 'textbbox'):
                    bbox_score = draw_final.textbbox((0,0), score_text, font=fonts['ui_kick'])
                    text_h_score = bbox_score[3] - bbox_score[1]
                else:
                    _, text_h_score = draw_final.textsize(score_text, font=fonts['ui_kick'])
                
                # [수정] 점수(item[1]) 기준으로 내림차순 정렬 (랭킹)
                sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

                total_scores_height = text_h_score + (30 * len(sorted_scores))
                current_y_top = (frame_height // 2) - (total_scores_height // 2)
                
                draw_pil_text_on_image(draw_final, score_text, (x_pos, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))
                
                for id_num, count in sorted_scores:
                    current_y_top += 30 
                    text = f"Player {id_num} : {count}"
                    draw_pil_text_on_image(draw_final, text, (x_pos + 20, current_y_top), fonts['ui_kick'], (0, 255, 255), (0,0,0))

                display_frame = cv2.cvtColor(np.array(img_pil_final), cv2.COLOR_RGB2BGR)
            

        # --- 최종 화면 표시 ---
        cv2.imshow(WIN_NAME, display_frame)

    # === 메인 루프 끝 ===
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()