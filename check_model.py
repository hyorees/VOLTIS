import joblib

# 1. 저장된 모델 불러오기
model = joblib.load('voltis_isolation_forest.pkl')

# 2. 모델 내부 정보 확인
print("=== 모델 로드 성공 ===")
print(f"생성된 의사결정 트리 개수: {len(model.estimators_)}")
print(f"이상치 판단 기준 임계값(offset): {model.offset_:.4f}")
print(f"학습에 사용된 피처 개수: {model.n_features_in_}")