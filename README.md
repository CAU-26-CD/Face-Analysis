# Rehearsal Feedback Platform — Backend

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

# Re:Action 서버 시작 가이드

## 순서

### 1. Docker DB 시작
```bash
docker start rehearsal-db
```

---

### 2. FastAPI 서버 (터미널 1)
```bash
cd ~/Desktop/rehearsal-platform
source venv/bin/activate
uvicorn app.main:app --reload
```
→ `http://127.0.0.1:8000/docs` 에서 camera-session 섹션 확인

---

### 3. ngrok (터미널 2)
```bash
npx ngrok http 8000
```
→ 출력된 `https://xxxx.ngrok-free.dev` 주소 복사

---

### 4. index.html ngrok 주소 교체
```bash
cd ~/Desktop/rehearsal-platform
grep -n "ngrok" reaction-pwa/index.html
```

주소가 바뀌었으면 교체:
```bash
# 기존 ngrok 주소를 새 주소로 교체 (xxxx 부분만 바꾸기)
sed -i '' 's|https://기존주소.ngrok-free.dev|https://새주소.ngrok-free.dev|g' reaction-pwa/index.html
sed -i '' 's|https://기존주소.ngrok-free.dev|https://새주소.ngrok-free.dev|g' reaction-pwa/camera.html
```

---

### 5. Netlify 재배포
```bash
cd ~/Desktop/rehearsal-platform/reaction-pwa
npx netlify deploy --prod
```

---

## 최종 확인

| 확인 항목 | 주소 |
|-----------|------|
| FastAPI docs | http://127.0.0.1:8000/docs |
| ngrok 요청 로그 | http://127.0.0.1:4040 |
| PWA (QR 페이지) | https://reaction-camera-connection.netlify.app |

---

## 주의사항

- ngrok은 재시작할 때마다 **주소가 바뀜** → 4번 단계 반복 필요
- DB가 없으면 서버 시작 실패 → 반드시 1번 먼저
- 카메라는 **HTTPS에서만 동작** → Netlify 주소로 접속해야 함
- 영상 업로드 확인: `ls ~/Desktop/rehearsal-platform/uploads/1/1/`