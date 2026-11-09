# 빠른 3D 가스 확산 모델 검토와 실제 실행

2026-11-06. QES 원본 C++/CUDA 실행파일을 빌드하고 실행했다. 자체 작성한 가짜 QES/QUIC 솔버로 대체하지 않았다. 사용자의 후속 지시에 따라 QUIC 정식 배포본 없이 QES를 우선 진행했다.

**결론:** 장애물·풍향·풍속을 반영한 3D 수동 추적가스 계산은 실행된다. 현재 설정에서 실측 농도 재현은 제한적이다. 액화수소 전체 누출 물리는 아직 구현·검증되지 않았다.

2026-11-07 추가: QES C++ 입자 모델에 저온 혼합·기화 잠열·밀도 기반 부력의 **연구용 근사 결합**을 구현하고 FFI 7개 시험·210개 농도 평균을 공개 SLABx-LH2와 직접 비교했다. 전체 정확도 개선에는 실패했으며, 완전한 LH2 해석 모델로 검증된 것은 아니다. [LH2 직접 비교 보고서](docs/LH2_direct_benchmark.md), [원시 예측값](results/lh2_benchmark/sensor_predictions.csv), [지표](results/lh2_benchmark/metrics.json).

후속 CFD 검토: 같은 FFI Test 1/3의 공개 FLACS 그림에서 30개 공통 센서를 대조하고 계산시간 보고값을 확인했다. 원시 CFD field·동일 경계 조건 검증과 구분했다. [CFD 비교 결과](docs/CFD_LH2_comparison.md).

Python 전체 이식은 [별도 패키지](python_qes/README.md)에서 진행 중이다. 열역학·혼합 규칙·수치 솔버·입자 적분·경계·관측기를 교체 가능하게 분리했다. 원본 부가 기능까지의 전체 이식은 아직 완료되지 않았으며 [기능별 상태](python_qes/component_status.json)와 원본 파일 목록을 추적한다.

- [검토 보고서](docs/report.md): 실제 시간, 오차, 데이터 적합성, 제한.
- [데이터 목록](docs/datasets.json): 17개 자료군. 다운로드·실행·후보·비공개를 구분.
- [계산 지표](results/metrics.json), [원시 데이터 검토](results/data_review.json), [검증 결과](results/verification.json).
- [3D 계산 결과](results/qes_3d.png), [풍향·풍속·장애물 변화](results/qes_scenarios.png), [실측 농도 비교](results/epa_concentration.png).

## 확보된 실행환경

- Windows 호스트, Docker Linux 컨테이너, NVIDIA GTX 1660 SUPER 6 GB.
- QES v2.3.1, commit `dced7ec571487be7538eca17bfc25bb3262fb6d0`.
- Sandia HyRAM commit `b45abf9a6d995951311be6aad836f1874e4d420b`.
- QES Docker 이미지: `gas-qes:2.3.1`. 이미지 빌드와 Python 환경 로그는 `results/`에 보존.
- QES·HyRAM 원본 라이선스는 각 `external/` 저장소에 보존. 두 원본 저장소의 실제 LICENSE/COPYING은 GPL-3.0이다.

## 재실행

프로젝트 루트 PowerShell에서 실행한다. Docker Desktop이 실행되어 있어야 한다.

```powershell
docker build -t gas-qes:2.3.1 .
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/work" gas-qes:2.3.1 python3 scripts/qes_suite.py --prepare
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/work" gas-qes:2.3.1 python3 scripts/qes_suite.py --case epa_sampling --solver 3
```

`--case empty_west`, `block_west`, `block_south`, `block_fast`는 작은 3D 민감도 사례다. `--wind-only`는 입자확산을 생략한다. `--solver 1`은 CPU, `--solver 3`은 GPU다. `epa_array`는 NaN 문제를 재현하는 원본 옵션 2 사례이고, 실제 비교에는 수정한 `epa_sampling`을 사용한다.

균일 흐름 입력은 `scripts/uniform_case.py`로 생성한다. 실행 명령 예:

```powershell
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/work" gas-qes:2.3.1 python3 scripts/uniform_case.py
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/work" gas-qes:2.3.1 /opt/QES/build/qesPlume/qesPlume -p data/cases/uniform_x/plume.xml -w data/cases/uniform_x/input_windsWk.nc -t data/cases/uniform_x/input_turbOut.nc -o results/uniform_x
```

`uniform_y`도 같은 방식으로 실행한다. 형상·기상·입자 수·수집 격자는 생성된 XML에 기록된다. QES 원본 난수 초기화가 시각을 사용하므로 입자 결과는 실행마다 통계적으로 달라질 수 있다. 공개 실험값에 맞춰 난수나 계수를 조정하지 않았다.

Python 후처리는 `.venv/Scripts/python.exe`와 `requirements.txt`를 사용한다. `scripts/analyze_qes.py`, `review_data.py`, `plot_results.py`, `verify_results.py` 순서로 실행한다. 단, 분석은 보고서에 명시된 전체 사례의 출력이 이미 존재해야 한다.

## 데이터 확보

`scripts/fetch_data.py`는 Dataverse/H2SAFE/QES 실측 자료와 메타데이터를 수집한다. `scripts/fetch_more.py`는 RADAR 측정파일과 NIST FDS의 작은 LNG 검증 CSV를 수집한다. RADAR의 HTTP 429는 추가 다운로드를 중단하도록 처리했다. 기존 실행에서는 trial 10 한 파일을 확보했고 다른 파일은 서버 429/시간초과가 발생했다.

대용량 LES 전체 36.5 GB를 내려받는 대신 `scripts/remote_hdf5.py`의 HTTP Range reader로 선택 필드·노드 좌표를 읽었다. `data/PPMLES/`에는 200개 기상조건, 탐침 좌표, LES 1번의 선택 농도·속도, 같은 인덱스의 좌표, 전체 노드 좌표가 있다. 선택 데이터는 전체 원본 파일의 MD5와 별도로 구분한다.

`scripts/prepare_must.py`는 LES 격자의 지붕 노드로 120개 컨테이너의 형상을 확인하고 QES 입력을 만든다. 원본 LES의 농도 소스는 아직 복원하지 않았으므로 비교는 바람장에 한정한다.

## 남은 물리

QES 수동 추적가스에 물질 이름만 수소로 바꾸는 것은 LH2 해석이 아니다. 2상 방출·플래싱·증발·열전달·부력/밀도 변화가 필요하다. HyRAM 보조 검증은 별도 적분형 제트 모델의 검증이며, QES와 결합된 3D LH2 모델을 검증한 것은 아니다. 안전거리나 LFL 경계를 검증된 결과로 제공하지 않는다.
