# PlumeSwift 사용 설명서

## 1. 환경과 첫 검증

```shell
python -m venv .venv
# Activate the environment, then run:
python -m pip install -e "./python_qes[fast,test]"
plumeswift info
plumeswift runtime-info
plumeswift plume-managed-container --steps 20 --release-per-step 16 --seed 0
python -m pytest python_qes/tests/test_components.py python_qes/tests/test_numerics.py -q
```

`info`는 등록된 구성요소와 버전을 JSON으로 출력한다. `plume-managed-container`는 외부 데이터 없이 방출·삭제·슬롯 재사용의 불변조건을 확인한다. 뒤의 구성요소 시험은 합성 바람장에서 입자 이류와 수치 적분도 검사한다. CUDA·GIS·원본 비교를 포함한 전체 시험에는 해당 라이브러리, 장치 또는 참조 파일이 필요하다.

## 2. 풍장, 입자와 수집

1. 지형, 건물과 누출 좌표를 동일한 좌표계로 옮긴다. 셀 간격과 지면·상부 경계를 정하고 격자가 건물 폭과 센서 위치를 표현하는지 확인한다.
2. 풍장 XML에 풍속, 풍향 FROM, 관측 높이와 거칠기를 지정한다. `plumeswift wind --help`로 XML과 출력 옵션을 확인한다. `plumeswift.wind_case.read_wind_case(path).run()`은 Python에서 솔버 결과, 작업공간 및 난류 배열을 반환한다.
3. 플룸 XML 또는 `Source`로 방출원을 지정한다. `plumeswift plume --help`의 풍장·플룸·출력 인자를 확인한다. `ParticleIntegrator`는 보간기, 경계, 벽 모델, 난수 생성기 및 수치 정책을 받는다. `ParticleSimulation.run(duration_s, dt)`은 방출과 이동을 진행하고 질량수지를 반환한다.
4. `SensorBoxes(centres, widths)`에서 센서 중심과 x/y/z 평균 폭을 m로 지정한다. `TimeAveragedSensors(boxes, start_s, end_s, thermo)`는 계산 시간창의 겹치는 길이로 가중평균을 계산한다. 물성을 넣으면 `mean()`은 0–1 단위의 기상 체적분율을 반환한다. 출력에 100을 곱하면 vol%다.
5. `PlumeCollection`은 격자 농도를, `particle_output`은 입자 스냅샷을 저장한다. 질량농도 kg/m³와 체적분율은 같은 값이 아니다. LH₂ 변환은 해당 시간·셀의 농도에 물성을 먼저 평가한 뒤 시간평균을 구한다.

## 3. 액화수소 연구 API

```python
import numpy as np
from plumeswift.lh2 import LH2Momentum
from plumeswift.thermo import LegacyEquilibrium

equilibrium = LegacyEquilibrium(
    ambient_temperature=276.85, liquid_fraction=0.559, relative_humidity=75.0
)
force = LH2Momentum(equilibrium)
positions = np.array([[0.0, 0.0, 0.5]])
masses = np.array([0.005])
drift = np.zeros((1, 3))
next_drift, diagnostics = force.update(
    positions, masses, drift, np.array([0]), 2.0, 0.2
)
print(next_drift, diagnostics)
```

질량을 같은 셀 안에서 먼저 합산하고, 셀 부피로 나눈 농도를 물성에 전달한다. 물성 공급자는 평가한 온도, 밀도, 체적분율과 기화분율을 반환한다. `CoolPropEquilibrium`을 선택할 때는 지원 범위와 범위를 벗어난 입력의 처리 옵션을 먼저 확인한다. `LH2Momentum`이 반환한 drift는 평균 바람장에 더하는 결합 성분이다.

설치본에서 Python PlumeSwift의 실측 비교를 재실행하려면 [README의 `plumeswift lh2-ffi` 명령](../README.md#reproduce-the-ffi-benchmarks)을 사용한다. 외부 모델 비교나 이전 연구 스크립트 재현에는 `scripts/benchmark_python_lh2.py`를 사용한다. 외부 입력을 확보하면 아래 디렉터리 규약을 따른다.

```text
data/FFI/lh2_ffi_sensors.csv
data/FFI/lh2_ffi_conditions.csv
data/cases/lh2/test1/{config.json,winds.xml,plume.xml,receptors.txt,cold.txt,warm.txt}
...
data/cases/lh2/test7/{config.json,winds.xml,plume.xml,receptors.txt,cold.txt,warm.txt}
```

센서 CSV와 기상·방출 조건의 열 이름은 `scripts/benchmark_lh2.py`의 `prepare()`에서 확인한다. 원 자료와 확보 경로는 [데이터 목록](datasets.json) 및 [FFI 프로토콜](LH2_direct_benchmark.md)에 있다. 준비 경로는 외부 SLABx-LH2 예제와 기준 풍장 XML을 요구한다. `receptors.txt`는 각 행에 동쪽·북쪽·높이 m 좌표를 담고 센서 CSV와 행 순서를 맞춘다.

```shell
python scripts/benchmark_python_lh2.py --model qes --tests 5 --compiled-transport --compiled-gle --tag serial
python scripts/benchmark_python_lh2.py --model qes --compiled-transport --compiled-gle --case-workers 5 --tag cases5
python scripts/benchmark_python_lh2.py --model qes --tests 5 --compiled-transport --compiled-gle --particle-workers 5 --tag particles5
```

사례별 병렬과 사례 내부 보간 병렬은 한 실행에서 하나를 선택한다. 각 결과 JSON에 질량수지와 시간이 기록되며, CSV에는 실측·예측 vol%가 들어간다. Numba 첫 실행의 컴파일 비용을 별도로 기록한다.

## 4. CFD 스타일 후처리

입자 상태를 규칙 격자에 모아 셀별 질량을 부피로 나눈다. LH₂이면 같은 물성 모델로 체적분율을 계산한다. 관측 시간 평균, 짧은 시간 구간 평균, 순간 값을 각각 구분한다. NetCDF나 셀 배열을 3D 뷰어로 옮기면 정해진 vol%의 등농도면, 높이별 단면 또는 시간 애니메이션을 표현할 수 있다.

개발 영상은 FFI Test 3 기상·유량에 건조 공기와 모식 컨테이너를 적용한 별도 사례다. 0–72초 동안 1초 구간 평균을 집계했고, 격자 간격은 수평 1 m, 수직 0.5 m였다. 지면 0–1 m 평균과 공중 2·8 vol% 면을 표시했다. 수직 축 확대와 화면 표시용 평활화는 계산 자체의 입력이 아니다.

## 5. 범위와 인용

안전거리, 과압 또는 점화 거동에 적용하려면 해당 조건과 물리 모델을 별도로 검증해야 한다. 원본 대응 상태는 [구성요소 JSON](../python_qes/component_status.json), 근거 및 재실행 계획은 [개발 기록](DEVELOPMENT.md)을 참고한다. 공개 연구에서 PlumeSwift를 사용하면 [CITATION.cff](../CITATION.cff)와 해당 원본 QES 방법을 함께 인용한다.
