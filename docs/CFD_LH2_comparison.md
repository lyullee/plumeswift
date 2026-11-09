# 공개 CFD 자료 확인과 QES+LH2 비교

2026-11-07. 같은 FFI LH2 시험을 계산한 FLACS 연구를 확보하고, Figure 6/8의 공통 센서 **30개**에서 직접 수치 대조했다. **고유량 시험에서는 FLACS의 pseudo-source가 QES 근사 결합보다 관측 평균에 더 가까웠다. 저유량에서는 QES의 오차가 더 작았다. CFD와 항상 동등한 정확도라고 할 수 없다.**

이번은 공개 그림을 이용한 문헌 비교다. FLACS 3D 원시장은 확보하지 못했고, 완전한 동일 조건 재실행을 했다고 주장하지 않는다. QES와 SLABx 예측은 앞선 실제 실행 결과를 사용했다. CFD를 이번 컴퓨터에서 재실행하지 않았다.

## 확보·확인한 자료

| 자료 | 물질·상황 | 실제 확보 범위 | 이번 사용 |
|---|---|---|---|
| Choi et al. 2023, [Applied Sciences 13, 12838](https://doi.org/10.3390/app132312838) | FFI Test 1/3, LH2 야외 방출 | 논문 PDF·XML·그림 원본. 원시 CFD 데이터는 요청 제공 방식 | 평균농도 30개 공통 센서의 그림 판독 비교, 문헌 계산시간 |
| Huang et al. 2026, [H2Vent](https://github.com/cenghuang-debug/H2Vent) | 상온 GH2, 자연환기 1/2 m³ 공간 | 논문 및 Git 저장소 clone 완료. 32개 case 설정과 solver 소스 | 재실행 가능한 CFD 입력 후보. 계산된 field·probe 결과는 저장소에 없음 |
| [PPMLES](https://zenodo.org/records/11394347) | 장애물 120개 주변 수동 추적가스, LES 200조건 | 이전에 메쉬·선택 3D 필드 확보 | 앞선 바람 비교 결과가 있음. 수소 열역학 검증 자료가 아님 |
| [GASFLOW-MPI 180조건](https://zenodo.org/records/18745400) | GH2 주차장 누출·환기 | 메타데이터만 공개, 2027-05-01까지 embargo | 수치 자료가 공개되지 않아 실행 비교에서 제외 |
| PRESLHY D3.2 | 저온/액화수소 연구의 여러 CFD 비교 | 공식 보고서 PDF 확보 | 후속 자료 경로 검토. 통일된 원시 3D field 묶음으로 간주하지 않음 |

H2Vent commit: `78386c2edb18fc69cba62423e3cebffc78e66e50`. 상온 수소 사례를 LH2 검증으로 세지 않았다. 논문 공개, case 설정 공개, 계산장 공개는 각각 구분해야 한다.

## 공통 센서의 농도 오차

FFI Test 1(0.225 kg/s) 및 Test 3(0.730 kg/s)에서 각 15개 센서를 사용했다. 위치는 30/50 m의 EN/E 방향 각 3높이, 100 m의 EN 방향 3높이다. 센서 ID는 OC04/05/06/07/08/09/16/17/18/19/20/21/25/26/27이다. EN의 bearing은 30/50 m에서 67.5°, 100 m에서 80°이며, E는 90°다.

실험값은 기존 FFI 원문 추출표의 **시간평균**을 사용했다. CFD는 Figure 6/8의 정상 평균농도에 대응하는 색상 마커를 추출했다. 시간 최대값과 섞지 않았다. CFD 그림의 검은 역삼각형은 실험값이며 CFD 예측으로 읽지 않았다.

| 모델/공개 경로 | Test 1 RMSE | Test 3 RMSE | 합계 30센서 RMSE | 합계 MAE |
|---|---:|---:|---:|---:|
| 기존 수동 QES | 0.60 | 0.77 | 0.69 | 0.52 |
| QES+LH2 근사 결합 | **0.12** | **1.23** | **0.87** | **0.51** |
| 공개 SLABx-LH2 기존 경로 | 0.47 | 1.17 | 0.89 | 0.59 |
| FLACS pseudo-source, χ=1 | 약 1.10 | 약 **0.64** | 약 0.90 | 약 0.53 |
| FLACS hybrid, χ=0.6 | 약 1.81 | 약 0.77 | 약 1.39 | 약 0.72 |
| FLACS liquid pool, χ=0 | 약 0.48 | 약 1.06 | 약 **0.82** | 약 0.56 |

단위는 체적농도 백분율의 차이인 **percentage point**다. 이 표는 30개 선택 센서의 오차이므로 앞선 210개 전체 센서 RMSE와 직접 비교하면 안 된다. SLABx는 최신 비공개 연구 경로가 아니다.

전체 30개 지표만 보면 QES와 일부 CFD 경로가 비슷해 보인다. 그러나 Test 3에서 QES가 센서에 도달하는 고농도 plume을 크게 과소예측한 반면, pseudo-source CFD는 그 구간을 더 잘 포착한다. 소스 모형과 유량에 따라 성능이 바뀌므로 하나의 합산 지표로 동등성을 주장할 수 없다. 실험값을 보고 센서마다 가장 잘 맞는 CFD 경로를 골라 합친 성능은 만들지 않았다.

## 그림 판독 정확도와 비교의 한계

PDF에 삽입된 원래 PNG 6개를 추출했다. 이미지의 표시된 축 눈금으로 선형 보정하고, 녹색/파랑/빨강 마커를 색으로 분리했다. 원호 좌표 및 센서 ID로 기존 모델 결과와 결합했다. 판독 마커 위치를 원 그림 위에 표시해 시각적으로 검토했다.

다른 색상에 가려진 마커는 정확한 값으로 간주하지 않았다. 보이는 마커는 ±14 pixel, 완전히 가려진 마커는 겹친 녹색 마커 중심의 ±26 pixel을 보수적인 구간으로 부여했다. 이는 기계 판독 오차의 경험적 경계이며, 통계적 95% 신뢰구간이나 CFD 자체의 모델 불확실성이 아니다. 특히 0 부근의 겹친 마커는 중앙값이 작게 달라질 수 있다.

| CFD 경로 | 판독 중앙값 기반 전체 RMSE | pixel 구간에 따른 RMSE 범위 |
|---|---:|---:|
| pseudo-source | 0.90 | 약 0.83–0.98 |
| hybrid | 1.39 | 약 1.32–1.47 |
| pool | 0.82 | 약 0.71–0.94 |

이 범위는 실험 불확실성·형상·습도 차이를 포함하지 않는다. 소수점 넷째 자리의 CFD 오차를 사실처럼 해석하면 안 된다.

추가로 다음 조건이 다르다:

- 논문 CFD는 누출점 주변 ISO 컨테이너를 포함한다. 앞선 QES FFI 기준 계산은 장애물 없는 야외 지형이다. 이번 자료에서 드러난 중요한 형상 차이다.
- CFD는 건조 공기 설정이다. 기존 QES/SLABx 계산은 상대습도 75% 가정을 썼다.
- CFD는 점 probe, QES/SLABx는 2×2×0.2 m 센서 상자의 평균이다.
- 근거리 CFD 최소 격자는 약 2.25 cm이며, QES는 2×2×0.5 m 바람 격자와 미해상 점 소스 근사다.
- QES는 밀도에 따른 압력장을 재계산하지 않고, 지면 열전달·풀·액적을 해상하지 않는다.
- 같은 실험을 이용한 비교지만, 경계 조건과 소스 처리까지 동일한 solver-to-solver 검증은 아니다. 원시 CFD 입력·field가 없어 이를 완전히 맞추지 못했다.

따라서 **CFD와 비슷한 합산 오차라는 관찰은 가능하지만, 동등한 모델이라는 결론은 불가능하다.**

## 계산시간

문헌의 대표 FLACS 격자(713,934 cells) 계산시간은 **76시간**, Intel Xeon Silver 4214 2.20 GHz의 40개 코어를 이용한 보고값이다. 전체 기술에는 case·격자에 따라 약 48–76시간이라고 기재돼 있고, 물리 계산시간은 800초다. 이 PC에서 FLACS를 실행해 측정한 값이 아니다.

같은 두 시험에 대한 기존 QES+LH2 실측 wall time은 Test 1 **48.43초**(물리시간 900초), Test 3 **26.50초**(물리시간 500초)다. 기존 공개 SLABx 경로는 각각 약 **0.67초**, **0.55초**다.

이 숫자는 근거리 격자, 계산영역, 출력, 물리 모형, 물리 계산시간, CPU/GPU가 다르다. 따라서 **CFD는 문헌상 수일, 현재 QES는 수십 초라는 비용 차이는 보이지만, 이를 동일 조건에서 측정한 수천/수만 배 speedup으로 발표할 수 없다.**

## 확보 파일과 재현

- `data/CFD_LH2/manifest.json`: 다운로드 URL·크기·SHA-256.
- `data/CFD_LH2/choi2023.pdf`, `.xml`, 원본 figure 이미지.
- `external/H2Vent/`: 공개 CFD 입력과 solver source.
- `results/cfd_comparison/matched_sensor_values.csv`: 센서별 실험/QES/SLABx/CFD 추정값과 pixel 오차·가림 여부.
- `results/cfd_comparison/metrics.json`: 동일 센서 지표 및 제한.
- `results/cfd_comparison/digitization_audit.json`, `digitization_overlay.png`: 판독 근거.
- `results/cfd_comparison/matched_comparison.png`: 비교 그림.

```powershell
.venv\Scripts\python.exe scripts/fetch_cfd_lh2.py
.venv\Scripts\python.exe scripts/compare_published_cfd.py
```

그림 추출은 `choi2023.pdf`의 PDF page 11과 14에 삽입된 image xref를 이용했다. 비교 스크립트의 calibration 표에 pixel 좌표와 원본 figure 매핑을 보존했다. 이 결과는 원시 CFD 파일의 대체물이 아니라 공개 문헌에서 읽을 수 있는 범위의 재현 가능한 예비 대조다.
