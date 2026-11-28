MODEL_NAMES = ['Training mean', 'Linear regression', 'Ridge fixed', 'RF baseline', 'Ridge tuned', 'RF tuned']

def score(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    denominator = np.sum((y-y.mean())**2)
    return {'MAE': float(np.mean(np.abs(prediction-y))),
            'RMSE': float(np.sqrt(np.mean((prediction-y)**2))),
            'R2': float(1-np.sum((prediction-y)**2)/denominator) if denominator > 0 else np.nan}

def grouped_splits(groups, n_splits):
    groups = np.asarray(groups)
    k = min(n_splits, len(np.unique(groups)))
    if k < 2: raise ValueError('Need at least two distinct condition groups for CV.')
    splits = list(GroupKFold(n_splits=k).split(np.zeros((len(groups), 1)), groups=groups))
    for tr, va in splits:
        assert not set(groups[tr]) & set(groups[va]), 'A condition group crossed a CV fold.'
    return splits

def make_pipeline(name, seed=SEED):
    linear = name.startswith('Ridge') or name == 'Linear regression'
    numeric = StandardScaler() if linear else 'passthrough'
    preparation = ColumnTransformer([
        ('numeric', numeric, FEATURES),
        ('magnification', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['magnification']),
    ])
    if name == 'Training mean': estimator = DummyRegressor(strategy='mean')
    elif name == 'Linear regression': estimator = LinearRegression()
    elif name.startswith('Ridge'): estimator = Ridge(alpha=1.0)
    elif name in ('RF baseline', 'RF tuned'):
        estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
    else: raise ValueError(name)
    return Pipeline([('prepare', preparation), ('model', estimator)])

def fit_training_only(name, X, y, groups, seed=SEED):
    estimator = make_pipeline(name, seed)
    if name.endswith('tuned'):
        inner = grouped_splits(groups, INNER_FOLDS)
        if name == 'Ridge tuned':
            search = GridSearchCV(estimator, {'model__alpha': np.logspace(-3, 4, 20)},
                                   scoring='neg_mean_absolute_error', cv=inner, n_jobs=N_JOBS)
        else:
            space = {'model__n_estimators': [100, 200, 400], 'model__max_depth': [None, 3, 6, 12],
                     'model__min_samples_leaf': [1, 2, 4], 'model__max_features': ['sqrt', 0.5, 1.0]}
            search = RandomizedSearchCV(estimator, space, n_iter=RF_SEARCH_ITERATIONS,
                                        scoring='neg_mean_absolute_error', cv=inner,
                                        random_state=seed, n_jobs=N_JOBS)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    return estimator.fit(X, y), {}

def nested_evaluation(X, y, groups, names=MODEL_NAMES):
    y, groups = np.asarray(y), np.asarray(groups)
    outer = grouped_splits(groups, OUTER_FOLDS)
    metrics, fold_rows, predictions = [], [], {}
    for name in names:
        out_of_fold = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(outer, 1):
            estimator, parameters = fit_training_only(name, X.iloc[tr], y[tr], groups[tr], SEED+fold)
            out_of_fold[va] = estimator.predict(X.iloc[va])
            fold_rows.append({'model': name, 'fold': fold, **score(y[va], out_of_fold[va]),
                              'training_groups': len(set(groups[tr])), 'validation_groups': len(set(groups[va])),
                              'parameters': json.dumps(parameters)})
            print(f'{name}: outer fold {fold}/{len(outer)} complete')
        assert np.isfinite(out_of_fold).all()
        predictions[name] = out_of_fold
        metrics.append({'model': name, **score(y, out_of_fold)})
    return pd.DataFrame(metrics).set_index('model'), pd.DataFrame(fold_rows), predictions
