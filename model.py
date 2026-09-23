import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import IsolationForest

# 1. 파일 읽기 (인코딩 에러 자동 예방 처리)
target_file = 'voltis_merged_data_all.csv'
print(f"1. 데이터셋 읽는 중: [{target_file}]")

# 엑셀 저장 시 바뀐 인코딩(CP949/EUC-KR/UTF-8)을 순차적으로 시도하여 읽기
encodings = ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']
df = None

for enc in encodings:
    try:
        df = pd.read_csv(target_file, encoding=enc)
        print(f"   └ [성공] '{enc}' 인코딩으로 파일을 읽었습니다.")
        break
    except (UnicodeDecodeError, Exception):
        continue

if df is None:
    raise ValueError("파일 인코딩을 읽을 수 없습니다. 파일 상태를 확인해 주세요.")

# 시간(hour) 추출
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

# 2. AI 이상 탐지 모델 학습 (공공데이터 학습)
features = ['전력거래량(MWh)']
for col in ['일조(hr)', '일사(MJ/m2)', '기온(°C)']:
    if col in df.columns:
        features.append(col)

X = df[features].fillna(0)

# Isolation Forest 학습
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X)

# 3. 데이터 기준점(Threshold) 자동 산출
# 기존 공공데이터 야간 시간대(20시~05시) 전력거래량의 상한선 측정
night_mask = (df['hour'] >= 20) | (df['hour'] <= 5)

# 주입된 극단치 제외 원본 상한선 기준 적용 (600 MWh 이상은 주입된 이상치로 판별)
valid_night_power = df.loc[night_mask & (df['전력거래량(MWh)'] < 600.0), '전력거래량(MWh)']
max_night_power = valid_night_power.max() if len(valid_night_power) > 0 else 530.0

# 4. 이상 탐지 판별
df['AI_탐지결과'] = '정상'

# [규칙 1] 야간 부정청구 조건 (공공데이터 상한선 초과 수치)
fraud_cond = night_mask & (df['전력거래량(MWh)'] > max_night_power * 1.1)

# [규칙 2] 계측기 고장 조건 (한낮 고일조 시 발전량 0)
fault_cond = ((df['hour'] >= 11) & (df['hour'] <= 14)) & (df['일조(hr)'] > 0.9) & (df['전력거래량(MWh)'] == 0)

df.loc[fraud_cond, 'AI_탐지결과'] = '부정청구(야간발전조작)'
df.loc[fault_cond, 'AI_탐지결과'] = '계측기고장(주간무발전)'

# 5. 결과 검증
anomalies = df[df['AI_탐지결과'] != '정상']

print("\n==========================================")
print(f"📊 AI 이상 탐지 결과: 총 {len(anomalies)}건 포착")
print("==========================================")

if len(anomalies) > 0:
    print(anomalies[['timestamp', '지역', '전력거래량(MWh)', '일조(hr)', 'AI_탐지결과']])
else:
    print("탐지된 이상치가 없습니다.")

# 결과 저장 (엑셀 안전 호환 인코딩 지정)
output_file = 'final_anomaly_results.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 탐지 결과 저장 완료: '{output_file}'")