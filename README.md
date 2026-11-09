# PlumeSwift

**Fast 3D Gas Leak Dispersion**

PlumeSwift는 QES v2.3.1을 바탕으로 한 독립 Python 이식판입니다. 이전 공개 이름은 QESx입니다.

PlumeSwift는 풍향·풍속, 지형과 장애물을 반영해 3차원 바람장을 만들고, 라그랑지안 입자로 가스 누출의 이동과 농도를 계산하는 연구 도구입니다. Python API로 열역학, 풍장, 난류, 적분기, 경계, 수집기를 따로 조립할 수 있습니다. 원본 QES v2.3.1의 진단형 풍장 및 입자 확산 방법을 참조했으며, 액화수소에는 연구용 평형 물성과 부력·항력 결합을 제공합니다.

**상태** · 0.1.0a4 연구 프리릴리스 · Python 3.11 이상 · GPL-3.0-only

- [상세 사용 설명서](docs/USER_GUIDE.md)
- [개발 과정과 검증 재실행](docs/DEVELOPMENT.md)
- [모듈별 함수·클래스 인덱스](docs/MODULE_REFERENCE.md)
- [원본 대응 상태](python_qes/component_status.json) 및 [이식 파일 인벤토리](python_qes/port_inventory.json)
- [인용](CITATION.cff), [이전 QESx 릴리스의 Zenodo 기록](https://doi.org/10.5281/zenodo.24073832) 및 [원본·제3자 고지](python_qes/NOTICE.md)

## 설치와 첫 실행

```shell
python -m pip install --pre plumeswift
plumeswift info
plumeswift plume-managed-container --steps 20 --release-per-step 16 --seed 0
```

개발 체크아웃에서는 `python -m pip install -e "./python_qes[fast,test]"`를 사용합니다. `fast`는 Numba 가속, `thermo`는 CoolProp, `gis`는 GIS·DEM 입력, `cuda11`은 별도 NVIDIA 환경을 위한 선택 의존성입니다. Python 모듈 이름은 `plumeswift`입니다. 레거시 명령은 `plumeswift qesWinds`, `plumeswift qesPlume`, `plumeswift qesFire`, `plumeswift qes`로 실행합니다.

## 액화수소 누출 예제와 3D 결과

Python 3.11 이상 환경에서 아래 명령을 실행하세요. 예제 입력은 배포본에 들어 있어 저장소를 내려받거나 별도 데이터 파일을 구할 필요가 없습니다.

```shell
python -m pip install "plumeswift[visual]==0.1.0a4"
plumeswift lh2-demo --output lh2-result --open
```

`--open`은 브라우저에서 `lh2-result/plume.html`을 엽니다. 3D 장면을 드래그해 회전하고, 범례에서 농도별 등농도면을 켜거나 끌 수 있습니다. HTML에 표시용 코드와 데이터가 포함되어 있어 저장한 파일은 오프라인에서도 열 수 있습니다.

입력을 확인하거나 바꾸려면 아래처럼 JSON을 꺼낸 뒤 편집하고 재실행하세요.

```shell
plumeswift lh2-demo --write-input lh2-case.json
plumeswift lh2-demo --input lh2-case.json --output lh2-custom --open
```

[기본 입력 파일](python_qes/plumeswift/example_data/lh2.json)은 합성 사례입니다. 기본값은 누출률 0.35 kg/s, 서풍 4 m/s, 높이 5 m 장애물, 계산시간 24초입니다. `grid_cells`와 `cell_size_m`의 순서는 x·y·z이며, 풍향은 바람이 오는 방향입니다. `isosurfaces_volpct`는 표시할 농도 3개를 지정합니다. 계산 범위에 존재하는 등농도면만 표시합니다.

| 결과 | 내용 |
| --- | --- |
| `case.json` | 이번 계산에 사용한 입력 사본 |
| `plume.html` | 8–24초 평균 기상 수소 농도, 장애물, 누출점의 인터랙티브 3D 장면 |
| `concentration.npz` | `h2_mean_volpct[z,y,x]`의 체적농도(%), `solid` 장애물 마스크, `coord_x/y/z` 셀 중심 좌표(m) |

예제는 PlumeSwift의 연구용 평형 물성과 입자 부력·항력 결합을 사용합니다. FFI 실험 재현 및 검증 절차는 [개발·검증 기록](docs/DEVELOPMENT.md)을 참고하세요.

## FFI 액화수소 실증 7개 조건: 실측 비교와 3D

[FFI 공식 보고서 페이지](https://www.ffi.no/en/publications-archive/large-scale-leakage-of-liquid-hydrogen-lh2-tests-related-to-bunkering-and-maritime-use-of-liquid-hydrogen)에서 **Download publication**을 눌러 PDF를 `ffi-report.pdf`로 저장하세요. 아래 명령은 로컬 PDF의 Appendix A에서 야외 Test 1–7의 센서 평균농도 210개를 읽고 계산값과 비교합니다. 패키지에는 [모델 계산 조건](python_qes/plumeswift/example_data/ffi_cases.json)이 들어 있습니다.

```shell
python -m pip install "plumeswift[fast,visual,ffi]==0.1.0a4"
plumeswift lh2-ffi --report ffi-report.pdf --workers 5 --output ffi-results --open
```

완료되면 `ffi-results/index.html`이 열립니다. 시험별 **3D plume**을 클릭한 뒤 장면을 드래그해 회전하세요. 센서 점에 마우스를 올리면 실측값과 해당 센서 위치의 예측값이 함께 표시됩니다. 등농도면은 모델의 3D 격자 평균장(0.5·2·4 vol% 중 계산 범위에 존재하는 면)이고, 표의 예측값은 각 센서 주변 2 × 2 × 0.2 m 상자에서 별도로 집계한 값입니다.

```shell
# Test 3만 실행
plumeswift lh2-ffi --report ffi-report.pdf --tests 3 --output ffi-test3 --open
# 전체 시험을 계산하되 센서 비교만 저장
plumeswift lh2-ffi --report ffi-report.pdf --workers 5 --no-3d --output ffi-sensors
```

`--workers`를 생략하면 순차로 실행합니다. 5개는 서로 다른 시험을 동시에 계산하는 프로세스 수입니다. `fast`는 선택 사항이며 설치하지 않아도 실행됩니다. `--open` 없이 계산한 뒤 HTML을 직접 열어도 됩니다.

| 출력 | 내용 |
| --- | --- |
| `index.html` | 시험별 오차와 3D/CSV 링크 |
| `sensor_comparison.csv` | 센서 좌표, 관측·예측 평균농도(vol%), 오차(예측 − 관측, %p), 두 평균 구간 |
| `summary.json` | 전체·시험별 RMSE, MAE, 편향, 4 vol% 포착 개수, 질량수지, 실행시간 |
| `testN/case.json`, `testN/winds.xml` | 시험별 실제 사용 입력 |
| `testN/plume.html`, `testN/concentration.npz` | 오프라인 3D 뷰어, `h2_mean_volpct[z,y,x]`와 셀 중심 좌표(m). 수평 좌표 원점은 누출점 |

이 명령은 기존 **장애물 없는 야외 기준 계산**을 재현합니다. 보고서의 실측 농도는 산소 결핍에서 환산한 평균이며, 모델은 100초 준비 계산 후 보고서와 동일한 **평균 길이**로 비교합니다. 원시 풍속·풍향 시계열, 현장 컨테이너와 점화는 포함하지 않습니다. 세부 설정은 [FFI 비교 프로토콜](docs/LH2_direct_benchmark.md)에 있습니다.

## 모듈과 입출력

기본 좌표는 **x 동쪽, y 북쪽, z 지면 기준 높이**이며 위치는 m, 속도는 m/s, 시간은 s, 질량은 kg입니다. 풍향은 바람이 **오는 방향**의 북쪽 기준 방위각입니다. 셀 배열은 보통 `[k,j,i]=[z,y,x]`, 입자 위치·속도는 `(N,3)`입니다. 레거시 파일 경로는 각 파일의 단위와 축 순서를 따르므로 일반 Python 배열과 혼용할 때 변환을 확인하세요.

| 단위 모듈 | 목적 | 주요 입력 | 출력 / 다음 모듈 |
| --- | --- | --- | --- |
| `assembly`, `contracts` | 수치 정책, 물리 계수와 구현 조립 | `AssemblyConfig`, 구성요소 선택·옵션, seed | 물성 공급자, 솔버, 설정·실행 정보 |
| `domain`, `geometry`, `gis`, `terrain` | 격자, 장애물, 지형 구성 | 셀 수·격자 간격, 폴리곤·건물 높이, DEM 또는 Shapefile | 격자 축, 지형·고체 셀, 벽·면 구조 |
| `cutcell_building`, `cutcell_terrain`, `cutcell_walls` | 절단 셀 기하와 경계 처리 | 지표·건물 면, 셀 모서리·격자 | 셀 분류, 열린 면 계수, 절단 셀 보조값 |
| `meteorology`, `measured_profile`, `native_sensor` | 초기 바람장 구성 | 풍향·풍속, 관측 높이, 거칠기, 안정도 | 수직 프로파일 및 격자 바람 입력 |
| `wrf`, `hrrr`, `wrf_exchange` | 기상 모델 입력 및 일부 연동 | 시간별 NetCDF, 좌표·지형 또는 교환 마커 | 바람·기상 프레임 및 연동 상태 |
| `building_pipeline`, `upwind`, `wake`, `rooftop`, `sidewall`, `street_canyon` | 장애물 주변 진단형 바람 보정 | 건물 형상, 기준 풍속, 설정 플래그 | 보정 초기 속도와 셀 상태 |
| `vegetation`, `canopy_*` | 수목·방풍림·캐노피 영향 | 식생 배치와 높이, 밀도 및 풍장 | 풍속·난류 보정 또는 식생 셀 분류 |
| `wind_case`, `wind_projection`, `numerics.poisson` | 바람장 조립 및 발산 보정 | 풍장 XML, 격자·벽 stencil, 초기 속도, 허용오차 | `u,v,w`, 셀 플래그, 작업공간, 솔버 수렴 정보 |
| `turbulence`, `mixing_length`, `ray_mixing` | 입자 구동용 난류장 | 풍속 구배, 벽·지형, 난류 방식 | 응력 6성분, 응력 발산, CoEps, 혼합길이 |
| `interpolation.FieldSampler` | 격자에서 입자 위치로 보간 | 좌표, staggered `u,v,w`, 난류장, `positions[N,3]` | 평균풍 `(N,3)`, 응력 `(N,3,3)`, 발산·소산·혼합길이 |
| `sources`, `plume_sources` | 누출 위치·스케줄 정의 | 점·선·상자 형상, kg/s, 입자 방출률, 방출 시각·초기 속도 | 새 입자 위치와 질량 |
| `thermo`, `thermodynamics.*` | LH₂ 혼합 평형·물성 평가 | 대기 온도 K, 액상 질량분율, RH %, 농도 kg/m³, 물성 공급자 | 온도, 혼합 밀도 kg/m³, 기상 H₂ 체적분율, 기화분율 |
| `lh2.LH2Momentum` | LH₂ 부력·항력을 입자에 결합 | 위치, 질량, drift, 셀 ID·부피, 시간간격 | 갱신된 drift `(N,3)` 및 힘 진단 |
| `numerics.langevin`, `compiled_gle` | 난류 입자 속도 갱신 | 이전 속도·응력, 새 응력, 응력 발산, CoEps, dt, 정규 난수 | 새 변동 속도 및 응력 |
| `particles`, `plume_transport` | 방출·이류·수집 진행 | 방출원, 보간기, 난수·경계, 관측기, 시간 설정 | 입자 상태, 누적 관측량, 질량수지 |
| `boundaries`, `walls`, `mesh_reflection` | 유출·반사·주기 경계 처리 | 좌표, 이동 벡터, 경계 모드, 고체 배열 또는 삼각형 | 보정 좌표와 속도, 활성 상태 |
| `deposition` | 입자 침적 계산 | 입자·셀 상태, 침적 파라미터 | 잔여 질량 및 침적 집계 |
| `statistics`, `plume_collection` | 센서 및 격자 농도 수집 | 센서 중심과 상자 폭, 입자 위치·질량, 평균 시간창 | 시간평균 농도, 격자 농도와 샘플 시각 |
| `io`, `netcdf_legacy`, `*_output` | 작업공간·농도·입자 저장 | 배열, 좌표, 시각, 출력 스키마 | NetCDF/CSV 및 진단 정보 |
| `particle_sampling_pool` | 정적 풍장 보간의 사례 내부 병렬 | 읽기 전용 보간기, 워커 수, 최소 입자 수 | 입력 입자 순서로 결합한 풍장·난류 샘플 |
| `cuda_*`, `optix_ray_mixing` | 선택 GPU·광선 추적 경로 | 장치 버퍼, 배치/셀/입자 상태, CUDA 또는 OptiX 환경 | 장치 계산 결과, 해당 경로의 진단 |
| `fire_case`, `fire_front`, `fire_grid`, `fire_run` | 화재 전선·연료·연기 경로 | 연료·점화, 경계와 바람장, 시간·출력 설정 | 확산 전선, 연소 및 연기 농도 관련 출력 |
| `analytical_cases`, `*_diagnostics`, `*_results` | 검증 사례와 오차 계산 | 해석·기준 해 또는 계산 출력 | 비교 지표, 진단 배열과 그림 |
 
## 검증에서 확인한 범위

FFI 7개 조건의 210개 센서에서 Python 순차, 사례별 5프로세스, 사례 내부 5워커의 시간평균 농도와 질량수지가 일치했습니다. 해당 실행에서 Python 사례별 5프로세스는 78.1초, 원본 C++ QES에 연구용 LH₂ 결합을 적용한 대응 실행은 92.9초였습니다. 두 코드의 출력 작업은 달랐습니다. Python 실측 RMSE는 1.9843 %p였고, 실측 평균이 4 vol% 이상인 센서 9개 중 2개를 포착했습니다. [조건·근거·한계](docs/DEVELOPMENT.md)

PlumeSwift는 독립 연구용 이식입니다. QES 원본은 University of Utah, University of Minnesota Duluth 및 기여자들이 개발했습니다. 기능별 비교와 미완료 항목은 [이식 상태](python_qes/component_status.json)에 기록되어 있습니다. 공개 소프트웨어의 출처와 적용 라이선스는 [NOTICE](python_qes/NOTICE.md)를 참고하세요.

![컨테이너를 포함한 FFI Test 3 모식 계산의 수소 등농도면](docs/evidence/plumeswift-lh2-side.png)

청록색과 주황색은 각각 2 vol%, 8 vol% 등농도면입니다. 이 화면은 건조 공기와 모식 컨테이너를 적용한 후처리 사례이며, 높이 축을 확대했습니다.
