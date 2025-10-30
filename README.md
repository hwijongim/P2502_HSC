# P2502_HSC
---

**믹서기 최적 파라미터 가이던스 시스템 구축**

---

## 📁 Directory Structure

```text
temp_project/
├─ pyproject.toml             # PEP 621 메타데이터 + 의존성(uv가 읽음)
├─ uv.lock                    # uv 락파일(자동 생성, 커밋 권장)
├─ README.md
├─ .gitignore
├─ .pre-commit-config.yaml    # ruff/black 등 사전 훅
├─ .ruff.toml                 # ruff 규칙
├─ .env.example               # 환경변수 샘플(민감정보는 제외)
│
├─ src/                       # 실제 패키지 코드(installable)
│  └─ temp_project/
│     ├─ __init__.py
│     ├─ config.py            # 설정/경로 유틸
│     ├─ data/                # 데이터 로딩/저장 로직
│     │  ├─ __init__.py
│     │  └─ datasets.py
│     ├─ features/            # 전처리/특성 엔지니어링
│     │  ├─ __init__.py
│     │  └─ build_features.py
│     ├─ models/              # 학습/추론 코드
│     │  ├─ __init__.py
│     │  ├─ train.py
│     │  └─ predict.py
│     └─ utils/               # 공통 유틸
│        ├─ __init__.py
│        └─ io.py
│
├─ scripts/                   # 커맨드라인 스크립트(빠른 실행용)
│  ├─ download_data.py
│  └─ evaluate.py
│
├─ notebooks/                 # 실험/EDA 노트북
│  └─ 01_explore.ipynb
│
├─ data/                      # (대용량/민감 데이터는 보통 미커밋)
│  ├─ raw/.gitkeep
│  └─ processed/.gitkeep
│
├─ models/                    # 학습된 모델 아티팩트(보통 미커밋)
│  └─ .gitkeep
│
└─ tests/                     # 단위/통합 테스트
   └─ test_smoke.py



