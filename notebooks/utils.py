import pandas as pd
from pandarallel import pandarallel
import numpy as np
import matplotlib.pyplot as plt

random_seed = 42

# cluster_dict = {
#     'cluster1': {'products': ["FFWED70284","FFWED70007","FFWED70267","FFWED70103","FFWED70199","FFSED70438","FFWED70033"],
#                  'var': 'T5'},
#     'cluster2': {'products': ["FFWES60194","FFSED70498","FFSED70533","FFWED70321"],
#                  'var': 'T5'},
#     'cluster3': {'products': ["FFWED70019","FFWED70102","FFWED70283","FFHED70076","FFWED70338"],
#                  'var': 'T3'},
#     'cluster4': {'products': ["FFHED70014","FFSED70032","FFHED70147"],
#                  'var': 'T3'},
#     'cluster5': {'products': ["FFHED60009","FFHED60006"], 
#                  'var': 'T3'}
#     }

cluster_dict = {
    'cluster1': {'products': ["FFWED70284","FFWED70007","FFWED70267","FFWED70103","FFWED70199","FFSED70438","FFWED70033",
                              "FFWES60194","FFSED70498","FFSED70533","FFWED70321"],
                 'var': 'T5'},
    'cluster2': {'products': ["FFWED70019","FFWED70102","FFWED70283","FFHED70076","FFWED70338",
                              "FFHED70014","FFSED70032","FFHED70147",
                              "FFHED60009","FFHED60006"],
                 'var': 'T3'},
    'cluster3': {'products': ["HCSED50105", "HCSED60072",
                              "HCWED60031", "HCSED50391", "HCSED70092", "HCSED60024",],
                 'var': 'M/B 점도 (ML)'},
    'cluster4': {'products': ["HCSED60010", "HCSES60015", "HCWES60017",
                              "HCSED70584", "HCSED60530", "HCSED50047",
                              "HCSED40011", "HCSED20006",
                              "HCWED70019", "FCHED60002", "FCWED70009",
                              "HCSED60017", "HCSED70143"],
                 'var': 'M/B 점도 (MS)'}
    }

def divide_step_log_df(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    공정 데이터 스텝 나누는 코드, 병렬 처리로 효율적
    
    required_columns:
    - mix시간
    - 작업지시번호-배치
    """
    # 병렬 초기화
    pandarallel.initialize(progress_bar=True)

    # 스텝 할당 함수
    def assign_steps(df):
        mix = df["mix시간"].values
        steps = np.zeros(len(mix), dtype=int)

        i = 0
        step_num = 1
        while i < len(mix) and step_num <= 3:
            while i < len(mix) and mix[i] == 0:
                i += 1
            start = i

            while i < len(mix) and mix[i] > 0:
                i += 1
            end = i

            steps[start:end] = step_num
            step_num += 1

        df["step"] = steps
        return df

    # 병렬로 그룹별 step 할당
    log_df = log_df.groupby("작업지시번호-배치", group_keys=False).parallel_apply(
        assign_steps
    )

    return log_df

def log_to_train_df(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    초단위 데이터를 "작업지시번호-배치"와 "step"별로 통계값을 구하고 (램압력과 로터스피드는 mean, 믹스온도는 max),
    step별로 파생변수를 생성
    
    required_columns:
    - Ram 압력
    - Rotor speed
    - mix온도
    - mix시간
    - 작업지시번호-배치: 작업지시번호와 배치번호를 합친 문자열
    - 시간: log 데이터의 시간 정보
    - step
    - 연월일
    """

    pandarallel.initialize(progress_bar=True)
    
    # === 1) 필요한 컬럼 정의: '연월일'을 반드시 포함하도록 수정 ===
    log_features = ["Ram 압력", "Rotor speed", "mix온도", "전력량"]
    required_columns = ["작업지시번호-배치", "step", "시간", "연월일"] + log_features
    df = log_df.loc[:, required_columns].copy()

    # === 2) 시간 전처리: 마이크로초 제거는 벡터화로 대체(성능/간결성) ===
    #    기존: 문자열->regex->to_datetime
    #    개선: 곧바로 to_datetime 후 초 단위로 내림
    df["시간"] = pd.to_datetime(
        df["시간"].astype(str).str.replace(r"\.\d*", "", regex=True), errors="coerce"
    )

    # === 3) 유효 row 필터 ===
    df = df[df["시간"].notna() & (df["step"] != 0)]

    # === 4) (배치, step) 집계: mean/max와 time(second) 한 번에 구성 ===
    grouped = df.groupby(["작업지시번호-배치", "step"], sort=False)

    agg_df = grouped.agg({
        "Ram 압력": "mean",
        "Rotor speed": "mean",
        "mix온도": "max",
        "전력량": 'max', 
    })
    time_sec = grouped["시간"].agg(lambda s: (s.max() - s.min()).total_seconds()).rename("time")

    step_df = pd.concat([agg_df, time_sec], axis=1).reset_index()

    # === 5) 와이드화(pivot) ===
    wide = step_df.pivot_table(
        index="작업지시번호-배치",
        columns="step",
        values=["Ram 압력", "Rotor speed", "mix온도", "전력량", "time"],
        aggfunc="first",
    )

    # 컬럼명 정리: step{n}_{col}
    wide.columns = [f"step{int(step)}_{col}" for col, step in wide.columns]
    wide = wide.reset_index()

    # === 6) 배치 단위 '연월일' 산출 후 조인(길이 불일치 해결의 핵심) ===
    # 배치 내 중복 날짜가 있을 수 있으므로 mode 우선, 없으면 첫 값 사용
    date_per_batch = (
        df.groupby("작업지시번호-배치")["연월일"]
          .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
          .reset_index()
    )

    result = wide.merge(date_per_batch, on="작업지시번호-배치", how="left")

    return result

def plot_predictions(y_true, y_pred, title, subtitle=None):
    df = pd.DataFrame({"true": y_true, "pred": y_pred}).reset_index(drop=True)
    df.sort_values("true").reset_index(drop=True).plot(
        alpha=0.6, title=title
    )
    if subtitle:
        plt.xlabel(subtitle)
    plt.legend(["true", "pred"])
    plt.show()
    

def inject_ref_features(params_dict: dict) -> dict:
    """영향변수(ref_features: stepk_time)를 관계식으로 계산해 params_dict에 덮어씀"""
    for k in [1, 2, 3]:
        m_key  = f"step{k}_mix온도"
        mt_key = f"step{k}_mix온도Xtime"
        t_key  = f"step{k}_time"

        m  = params_dict.get(m_key, None)
        mt = params_dict.get(mt_key, None)

        # 두 값이 모두 있어야 계산 가능. 없으면 그대로 두되, 모델 입력 전에 NaN 검사 권장.
        if m is not None and mt is not None:
            # 0 나눗셈 방지
            if m == 0 or (isinstance(m, float) and np.isclose(m, 0.0)):
                params_dict[t_key] = np.nan  # 혹은 큰 패널티를 주는 방식
            else:
                params_dict[t_key] = mt / m
    return params_dict


def create_objective_function(
    fixed_params, all_features_list, opt_features_list, 
    ct_target=None, st_target=None, c_target=None, v_target=None, h_target=None,
    ct_model=None, st_model=None, c_model=None, v_model=None, h_model=None,
    penalty_invalid=1e6, scales=None, weights=None
):
    """고정 조건과 최적화 대상을 조합해 모델을 예측하는 함수를 생성"""
    
    scales = scales or {}
    weights = weights or {}
    

    def _safe_scale(key: str, default: float = 1.0) -> float:
        s = scales.get(key, default)
        try:
            s = float(s)
        except Exception:
            s = default
        # 0 / 비정상값 방지
        if (s is None) or (not np.isfinite(s)) or (s <= 0):
            s = default
        return s
    
    def _safe_weight(key: str, default: float = 1.0) -> float:
        w = weights.get(key, default)
        try:
            w = float(w)
        except Exception:
            w = default
        # 0 / 비정상값 방지
        if (w is None) or (not np.isfinite(w)) or (w <= 0):
            w = default
        return w

    def objective(opt_params_values):
        # 값 생성
        params_dict = dict(fixed_params)  # 고정변수 삽입
        for name, val in zip(opt_features_list, opt_params_values):
            params_dict[name] = val
            
        # ▼ 순서 제약: step1 < step2 < step3  (엄격 '>' 조건)
        t1 = params_dict.get("step1_mix온도", None)
        t2 = params_dict.get("step2_mix온도", None)
        t3 = params_dict.get("step3_mix온도", None)
        if (t1 is None) or (t2 is None) or (t3 is None) or \
           (not np.isfinite(t1)) or (not np.isfinite(t2)) or (not np.isfinite(t3)) or \
           not (t1 <= t2 <= t3):
            return penalty_invalid
        # (만약 '크거나 같다'로 완화하려면 위 마지막 조건을: not (t1 <= t2 <= t3) 로 변경)

        params_dict = inject_ref_features(params_dict)

        try:
            final_params_vector = [params_dict[f] for f in all_features_list]
        except KeyError as e:
            # 빠진 피처가 있으면 바로 패널티
            return penalty_invalid
        
        # df 생성
        prediction_input_df = pd.DataFrame(
            [final_params_vector], columns=all_features_list
        )
        
        # NaN/inf 검사 → 불능 조합 패널티
        if not np.isfinite(prediction_input_df.to_numpy(dtype=float)).all():
            return penalty_invalid

        preds = {}
        if ct_target  is not None: preds["ct"]  = ct_model.predict(prediction_input_df)[0]
        if st_target  is not None: preds["st"]  = st_model.predict(prediction_input_df)[0]
        if c_target   is not None: preds["c"]   = c_model.predict(prediction_input_df)[0]
        if v_target   is not None: preds["v"]   = v_model.predict(prediction_input_df)[0]
        if h_target   is not None: preds["h"]   = h_model.predict(prediction_input_df)[0]

        norm_errors = []
        if ct_target is not None: 
            e = abs(preds["ct"] - ct_target) / _safe_scale("ct")
            norm_errors.append(_safe_weight("ct") * e)
        if st_target is not None:
            e = abs(preds["st"] - st_target) / _safe_scale("st")
            norm_errors.append(_safe_weight("st") * e)
        if c_target is not None:
            e = abs(preds["c"] - c_target) / _safe_scale("c")
            norm_errors.append(_safe_weight("c") * e)
        if v_target is not None:
            e = abs(preds["v"] - v_target) / _safe_scale("v")
            norm_errors.append(_safe_weight("v") * e)
        if h_target is not None:
            e = abs(preds["h"] - h_target) / _safe_scale("h")
            norm_errors.append(_safe_weight("h") * e)

        if not norm_errors:
            return penalty_invalid

        labels = {
            "ct":  ("Ct 90", ct_target),
            "st":  ("Scorch", st_target),
            "c":   ("Cycle Time", c_target),
            "v":   ("Viscosity", v_target),
            "h":   ("Hardness", h_target),
        }
        logs = []
        for k, (label, tgt) in labels.items():
            if k in preds:
                logs.append(f"{label}: {preds[k]:.4f} (target {tgt})")
        if logs:
            print(" | ".join(logs))


        return float(sum(norm_errors))

    return objective