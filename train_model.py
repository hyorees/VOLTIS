import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# 1. 병합 데이터 불러오기
df = pd.read_csv('voltis_merged_data_all.csv')

# 2. 피처 엔지니어링: 지역별 발전량 스케일링 및 일사량 괴리도 계산
# 1) 지역별 최대 발전량으로 나눈 정규화 발전량 (0 ~ 1)
df['정규화_발전량'] = df.groupby('지역')['전력거래량(MWh)'].transform(lambda x: x / (x.max() + 1e-5))

# 2) 기상-발전 괴리도: 일사량(최대 약 4.0 MJ/m2 기준 정규화)과 발전량 비율의 차이
df['기상대비_발전괴리도'] = np.abs(df['정규화_발전량'] - (df['일사(MJ/m2)'] / 4.0))

features = ['정규화_발전량', '기온(°C)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', '기상대비_발전괴리도']
X_train = df[features]

print("1. Isolation Forest 모델 재학습 시작...")
# contamination: 약 3% 수준으로 설정
model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
model.fit(X_train)
print("학습 완료!")

# 3. 모델 저장
joblib.dump(model, 'voltis_isolation_forest.pkl')
print("모델 파일 저장 완료: voltis_isolation_forest.pkl")

# -------------------------------------------------------------
# 4. 가상 이상 데이터 탐지 테스트
# -------------------------------------------------------------
print("\n2. 가상 이상 데이터 탐지 테스트 진행...")

# 테스트 케이스 (발전소 최대 용량을 300 MWh로 가정한 테스트)
test_cases = pd.DataFrame([
    # 케이스 A: 정상 데이터 (일사량 2.8, 발전량 150 -> 정규화 약 0.5)
    {'정규화_발전량': 0.50, '기온(°C)': 22.0, '일조(hr)': 1.0, '일사(MJ/m2)': 2.8, '전운량(10분위)': 0.0},
    # 케이스 B: 계측기 고장 (일사량 3.0의 한낮인데 발전량 0.0)
    {'정규화_발전량': 0.00, '기온(°C)': 25.0, '일조(hr)': 1.0, '일사(MJ/m2)': 3.0, '전운량(10분위)': 0.0},
    # 케이스 C: 부정 조작 (폭우/흐림 일사량 0.1인데 발전량은 거의 최대치 0.90)
    {'정규화_발전량': 0.90, '기온(°C)': 10.0, '일조(hr)': 0.0, '일사(MJ/m2)': 0.1, '전운량(10분위)': 10.0}
])

# 괴리도 컬럼 계산
test_cases['기상대비_발전괴리도'] = np.abs(test_cases['정규화_발전량'] - (test_cases['일사(MJ/m2)'] / 4.0))

predictions = model.predict(test_cases[features])
scores = model.decision_function(test_cases[features])

case_names = ["정상 데이터", "계측기 이상(낮인데 0)", "부정 조작(흐린데 폭증)"]
for name, pred, score in zip(case_names, predictions, scores):
    status = "✅ 정상" if pred == 1 else "🚨 이상 감지(Anomaly)"
    print(f"[{name}] 판정: {status} (이상치 스코어: {score:.4f})")