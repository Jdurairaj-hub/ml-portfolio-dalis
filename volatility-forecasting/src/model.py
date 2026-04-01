import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.features import create_advanced_features


def train_ml_model(data):
    data_enhanced = create_advanced_features(data).dropna().copy()

    feature_cols = [
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility', 'vol_ratio',
        'volume_ma_ratio', 'volume_momentum',
        'ma_ratio', 'rsi', 'rsi_change',
        'macd_histogram', 'bb_position', 'high_low_ratio',
        'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3'
    ]

    for f in feature_cols + ['next_day_log_return']:
        if f not in data_enhanced.columns:
            raise KeyError(f"Required column '{f}' not found in data_enhanced")

    X = data_enhanced[feature_cols]
    y = data_enhanced['next_day_log_return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    ridge_model = Ridge(alpha=1.0)

    ensemble = VotingRegressor([('rf', rf_model), ('gb', gb_model), ('ridge', ridge_model)])
    ensemble.fit(X_train_scaled, y_train)

    y_train_pred = ensemble.predict(X_train_scaled)
    y_test_pred = ensemble.predict(X_test_scaled)

    train_score = ensemble.score(X_train_scaled, y_train)
    test_score = ensemble.score(X_test_scaled, y_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_direction = np.mean((y_train_pred > 0) == (y_train > 0))
    test_direction = np.mean((y_test_pred > 0) == (y_test > 0))

    print(f"Model Performance:")
    print(f"  Training R²: {train_score:.6f} | Testing R²: {test_score:.6f}")
    print(f"  Training MSE: {train_mse:.8f} | Testing MSE: {test_mse:.8f}")
    print(f"  Training MAE: {train_mae:.8f} | Testing MAE: {test_mae:.8f}")
    print(f"  Train Direction Acc: {train_direction:.2%} | Test Direction Acc: {test_direction:.2%}")

    feature_importance = []
    if hasattr(ensemble, 'named_estimators_') and 'rf' in ensemble.named_estimators_:
        rf_fitted = ensemble.named_estimators_['rf']
        if hasattr(rf_fitted, 'feature_importances_'):
            importances = rf_fitted.feature_importances_
            feature_importance = sorted(list(zip(feature_cols, importances)), key=lambda x: x[1], reverse=True)
            print("\nTop 8 Feature Importances (RF):")
            for feat, imp in feature_importance[:8]:
                print(f"  {feat}: {imp:.6f}")

    return ensemble, scaler, feature_cols, X_test_scaled, y_test, data_enhanced
