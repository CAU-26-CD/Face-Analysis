# 🎭 Rehearsal Feedback Platform — Backend

**POST** http://127.0.0.1:8000/api/v1/camera-session/create?pwa_base_url=https://reaction-camera-connection.netlify.app
-> QR 생성해서 세션과 카메라 url 매핑해주는 api

**GET** http://127.0.0.1:8000/api/v1/camera-session/Iddtrs2Zmrc7V0tO/status
-> 촬영중인지 아닌지 판단하는 api -> status 반환 하며, 노트북 화면에 "연결됨/아님" 여부 띄워주는 api

**POST** http://127.0.0.1:8000/api/v1/camera-session/fNzHb3BDQjFJ1RAD/done
-> 영상 촬영 마무리되면 status done으로 바꿔주는 api

## MAIN LOGIC

핸드폰 촬영 종료
    ↓
video/upload 로 영상 서버에 저장
    ↓
저장 완료되면 camera-session/{id}/done 자동 호출  ← 이거
    ↓
노트북 status 폴링에서 "done" 감지
    ↓
노트북 화면이 매핑 화면으로 전환

---

## 📁 프로젝트 구조

```
rehearsal-platform/
├── app/
│   ├── main.py                  # FastAPI 앱 진입점
│   ├── core/
│   │   ├── config.py            # 환경변수 설정
│   │   ├── database.py          # SQLAlchemy async 엔진
│   │   ├── security.py          # JWT / 비밀번호 해싱
│   │   └── deps.py              # 공통 Depends (현재 유저 등)
│   ├── models/
│   │   └── models.py            # ORM 모델 (ERD 기반)
│   ├── schemas/
│   │   └── schemas.py           # Pydantic Request/Response
│   └── api/v1/endpoints/
│       ├── auth.py              # 회원가입 / 로그인
│       ├── projects.py          # 프로젝트 CRUD + 조인
│       ├── sessions.py          # 세션 CRUD
│       ├── actors.py            # 배우 태그 관리
│       ├── feedbacks.py         # 피드백 작성/조회/삭제
│       ├── video.py             # 영상 업로드/녹화 관리
│       └── report.py            # 피드백 레포트 생성
└── requirements.txt
```

---

## 🚀 로컬 개발 시작

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate        # mac/linux

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 파일 열어서 DATABASE_URL, SECRET_KEY 수정

# 4. PostgreSQL 실행 (Docker 권장)
docker run -d \
  --name rehearsal-db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rehearsal_db \
  -p 5432:5432 \
  postgres:16

# 5. 서버 실행
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

---

## 📋 API 스펙 요약

### Auth
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/auth/register` | 회원가입 → 토큰 반환 |
| POST | `/api/v1/auth/login` | 로그인 → 토큰 반환 |

### Projects
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects` | 내 프로젝트 목록 |
| POST | `/api/v1/projects` | 프로젝트 생성 (자동 4자 코드 발급) |
| POST | `/api/v1/projects/join` | 코드로 프로젝트 조인 |
| GET | `/api/v1/projects/{id}` | 프로젝트 상세 |

### Sessions
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects/{pid}/sessions` | 세션 목록 |
| POST | `/api/v1/projects/{pid}/sessions` | 세션 생성 |
| GET | `/api/v1/projects/{pid}/sessions/{sid}` | 세션 상세 |
| PATCH | `/api/v1/projects/{pid}/sessions/{sid}` | 세션 수정 (in_progress 등) |

### Actors (배우 태그)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects/{pid}/sessions/{sid}/actors` | 배우 목록 |
| POST | `/api/v1/projects/{pid}/sessions/{sid}/actors` | 배우 등록 (최대 8명) |
| DELETE | `/api/v1/projects/{pid}/sessions/{sid}/actors/{aid}` | 배우 삭제 |

### Feedbacks
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects/{pid}/sessions/{sid}/feedbacks` | 피드백 목록 (actor/category 필터) |
| POST | `/api/v1/projects/{pid}/sessions/{sid}/feedbacks` | 피드백 작성 |
| GET | `/api/v1/projects/{pid}/sessions/{sid}/feedbacks/{fid}` | 피드백 단건 |
| DELETE | `/api/v1/projects/{pid}/sessions/{sid}/feedbacks/{fid}` | 피드백 삭제 |

### Video
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects/{pid}/sessions/{sid}/video` | 영상 정보 조회 |
| POST | `/api/v1/projects/{pid}/sessions/{sid}/video/start` | 녹화 시작 (타임스탬프 기록) |
| POST | `/api/v1/projects/{pid}/sessions/{sid}/video/upload` | 영상 파일 업로드 |

### Report
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/projects/{pid}/sessions/{sid}/report` | 세션 피드백 레포트 (배우별/카테고리별 통계) |

---

## 🔮 향후 확장 포인트

- **배우 자동 식별**: `video/` 엔드포인트에서 영상 처리 후 actor 매핑 (OpenCV/DeepSORT 연동)
- **실시간 피드백**: WebSocket 추가 (`/ws/sessions/{session_id}`)
- **단축키 지원**: 프론트엔드에서 처리, 백엔드는 현재 API 그대로 사용
- **User-Project N:M**: 현재 단순 FK → `user_project` 조인 테이블로 확장
- **S3 영상 저장**: `video.py`의 로컬 저장 로직을 boto3 S3 업로드로 교체
