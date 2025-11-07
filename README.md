# 🤸‍♂️ 실시간 킥 카운터 프로젝트 (Real-time Kick Counter)

[여기에 프로젝트에 대한 간단한 한두 줄 설명을 적어주세요. 예: 웹캠을 사용하여 사용자의 발차기 동작을 실시간으로 감지하고 횟수를 카운트하는 프로그램입니다.]



---

## 📜 주요 기능

* 실시간 웹캠 영상 분석
* MediaPipe 또는 YOLO를 이용한 신체/객체 감지
* [발차기(Kick) 동작 인식 및 카운트]
* [점프(Jump) 동작 인식 및 카운트]
* [기타 구현한 주요 기능]

---

## 💻 사용 기술

* **Python**
* **OpenCV:** 실시간 영상 처리를 위해 사용
* **MediaPipe / YOLO (Ultralytics):** [둘 중 사용한 것, 혹은 둘 다 적기] (객체/포즈 감지)
* [그 외 사용한 주요 라이브러리, 예: SORT (객체 추적)]

---

## 🚀 설치 및 실행 방법

### 1. 프로젝트 복제

```bash
git clone [GitHub 저장소 주소.git]
cd kickcount

# 가상 환경 생성 (최초 1회)
python -m venv venv

# 가상 환경 활성화
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 필요한 라이브러리 설치
pip install -r requirements.txt