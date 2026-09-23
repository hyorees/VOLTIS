import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. 파일 불러오기 (100% 정상 기준 데이터)
df = pd.read_csv('voltis_merged_data_all.csv')

# 2. 피처 생성 (시간대, 수율 등)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['solar_efficiency'] = df['전력거래량(MWh)'] / (df['일사(MJ/m2)'] + 1e-5)

features = ['전력거래량(MWh)', '기온(°C)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', 'hour', 'solar_efficiency']
X_normal = df[features]

# 3. AI(Isolation Forest)에게 "이게 정상 패턴이다" 학습시키기
model_baseline = IsolationForest(contamination=0.01, random_state=42)
model_baseline.fit(X_normal)

print("✅ 정상 데이터 패턴 학습 완료!")