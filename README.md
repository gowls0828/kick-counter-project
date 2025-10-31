# 🥋 태권도 킥 카운터 및 자세 분석 프로젝트 (Kick Counter and Pose Analysis Project)

이 프로젝트는 실시간 영상 스트림 또는 녹화된 영상을 분석하여 태권도 킥 동작을 감지하고, 킥 횟수를 카운트하며, 자세 정보를 추적하는 인공지능 기반 시스템입니다.

## 🚀 시작하기 (Getting Started)

프로젝트를 로컬 환경에서 실행하기 위한 단계별 안내입니다.

### 📋 사전 준비 사항

* **Python 3.x**
* **Git**

### 💻 설치 방법

1.  **프로젝트 클론 (Clone the Repository)**
    ```bash
    git clone [https://github.com/gowls0828/kick-counter-project.git](https://github.com/gowls0828/kick-counter-project.git)
    cd kick-counter-project
    ```

2.  **가상 환경 설정 (Set up Virtual Environment)**
    시스템 환경과의 충돌을 막기 위해 가상 환경 사용을 권장합니다.
    ```bash
    # 가상 환경 생성
    python -m venv venv

    # 가상 환경 활성화 (Windows PowerShell 기준)
    .\venv\Scripts\Activate
    ```

3.  **필수 라이브러리 설치 (Install Dependencies)**
    `requirements.txt`에 명시된 모든 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

### ▶️ 프로젝트 실행

프로젝트의 메인 스크립트를 실행합니다. (파일명이 `kickcount.py`라고 가정합니다.)

```bash
python kickcount.py