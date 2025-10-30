### Library Import 
import os
import re
import sys
import glob
from typing import Any
import shap
import optuna
import pickle
import warnings
import numpy as np 
import pandas as pd 
import seaborn as sns
import xgboost as xgb
from math import ceil
from tqdm import tqdm
from scipy import stats
import statsmodels.api as sm
from functools import partial
from collections import Counter
from pandarallel import pandarallel
from xgboost import XGBRegressor, callback as xgb_callback
from datetime import timedelta, time, datetime
from scipy.stats import truncnorm
from IPython.display import Image
from optbinning import OptimalBinning
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
warnings.filterwarnings('ignore')
plt.rc('font', family='Apple SD Gothic Neo')

# Pandas 옵션 설정 
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

# Scikit-Learn 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence 
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline

# Import fns 
from utils import * 

############################################################### Import Raw data ###############################################################  
def import_raw_data(path): 
    ''' 
    Log, Qa, Recipe, info, weather Import 

    Returns: Log, Qa, Recipe, Info, Weather Dataset 
    ''' 
    # Import Datasets 
        # Log 
    log_df = pd.read_csv(f"{path}/log_df_10s.csv")
        # QA 
    qa_df = pd.read_csv(f"{path}/qa_df.csv") 
        # Recipe
    recipe_df = pd.read_csv(f"{path}/recipe_df.csv") 
        # INFO 
    info = pd.read_excel(f"{path}/투입자재 CODE별 비중 정보.xlsx", header=1, index_col=0).reset_index(drop=True)
        # Weather 
    weather_df = pd.read_csv(f"{path}/weather_dg.csv") 
    weather_df['연월일'] = weather_df['연월일'].astype(str)


    return log_df, qa_df, recipe_df, info, weather_df 

############################################################### Log Data Clustering ###############################################################  
def get_cluster_p_codes(df, p_type): 
    ''' 
    p_type에 맞는 Log DF만 가져오기 

    Returns: Log DF 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True) 

    # Clusters 
    if p_type=='FMB': 
        p_codes = [
                "FFWED70284", "FFWED70007", "FFWED70267", "FFWED70103", "FFWED70199", "FFSED70438",
                "FFWED70033", "FFWES60194", "FFSED70498", "FFSED70533", "FFWED70321",
                "FFWED70019", "FFWED70102", "FFWED70283", "FFHED70076", "FFWED70338",
                "FFHED70014", "FFSED70032", "FFHED70147", "FFHED60009", "FFHED60006"
                ]
                
    elif p_type=='CMB': 
        p_codes = [
                "HCSED50105", "HCSED60072", "HCWED60031", "HCSED50391", "HCSES60015", "HCWES60017",
                "HCSED70584", "HCSED60530", "HCSED50047", "HCSED70092", "HCSED60024", "HCSED40011",
                "HCWED70019", "FCHED60002", "FCWED70009", "HCSED60017", "HCSED70143"
                ]

    # Get Log 
    dataset = dataset[dataset['제품코드'].isin(p_codes)].reset_index(drop=True) 

    return dataset 

############################################################### 정규분포 곡선 ###############################################################  
def feature_normal_distribution(dataset, feature, xlim=None): 
    # 데이터 정제
    data = pd.to_numeric(dataset[feature], errors="coerce").dropna()

    # 플롯 크기 설정
    plt.figure(figsize=(12, 6))

    # Histogram + KDE (정규분포 형태 시각화)
    sns.histplot(data, kde=True, bins=30, color='#69b3a2', edgecolor='black', alpha=0.8)

    # 평균 / 중앙선 표시
    plt.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(data.median(), color='blue', linestyle='--', linewidth=2, label='Median')

    # 시각적 설정
    plt.title(f"Histogram and KDE of {feature}", fontsize=16, fontweight='bold')
    plt.xlabel('Value', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # x축 제한 설정 (선택적)
    if xlim:
        plt.xlim(xlim)

    # 그래프 출력
    plt.tight_layout()
    plt.show()

######################################################################## Prep Log ######################################################################## 
def prep_log(df): 
    ''' 
    log_df => 정제 및 전처리 

    Returns: prep_log_df 
    ''' 
    # Copy 
    dataset = df.copy() 

    # B/T na 제거 
    dataset = dataset[dataset["b/t"].notna()]

    # B/T 순서 1번 제거 
    dataset = dataset[dataset['b/t']!=1.0] 

    # 작업지시번호 배치 
    dataset['작업지시번호-배치'] = (dataset["작업지시번호"] + "-" + dataset["b/t"].astype(int).astype(str))

    # Step 생성 
    dataset = divide_step_log_df(dataset)

    # DateTime + 연월일 생성 
    dataset["연월일"] = pd.to_datetime(dataset["시간"], errors="coerce").dt.strftime("%y%m%d")
    log_df = dataset.reset_index(drop=True).copy()  

    # To Train 
    log_train_df = log_to_train_df(dataset)
        # CYCLE TIME 생성 
    log_train_df["cycle time"] = (log_train_df["step1_time"] + log_train_df["step2_time"] + log_train_df["step3_time"])

    return log_df, log_train_df  

######################################################################## Prep Qa ######################################################################## 
def prep_qa(df): 
    '''
    qa_df => 전처리 

    Returns: prep_qa_df  
    '''
    # Copy 
    dataset = df.copy() 
    target_list = ['Ct 90','Scorch (T5)','Scorch (T3)','Vm (T5)', "Vm (T3)", "M/B 점도 (ML)", "M/B 점도 (MS)",'경도']

    # 작업지시번호-배치 생성 
    dataset["작업지시번호-배치"] = (dataset["작업지시번호"] + "-" + dataset["B/T"].astype(int).astype(str)) 
    
    # 컬럼 항목 성택 
    dataset = dataset[dataset["검사항목명"].isin(target_list)][["작업지시번호-배치", "결과", "기준", "검사항목명"]]
    qa_df = dataset.reset_index(drop=True).copy()  

    ######################################## QA 피벗 테이블 생성 ########################################
    # 피벗 테이블 생성 
    dataset = dataset.pivot_table(
                                index="작업지시번호-배치",
                                columns="검사항목명",
                                values=["결과", "기준"],
                                aggfunc='first'
                                )
    qa_train_df = dataset.copy() 
    
    # 컬럼 정리 
    qa_train_df.columns = [f"{col[1]}_{col[0]}" for col in qa_train_df.columns]
    qa_train_df = qa_train_df.reset_index()
    dataset = qa_train_df.copy() 

    ######################################## 파생변수 생성 ########################################
    # Vm
    dataset['Vm_feature'] = np.select(
                                    [dataset['Vm (T3)_결과'].notna(), dataset['Vm (T5)_결과'].notna()],
                                    [0, 1],
                                    default=np.nan
                                    )

    # M/B 
    dataset['MB_feature'] = np.select(
                                    [dataset['M/B 점도 (ML)_결과'].notna(), dataset['M/B 점도 (MS)_결과'].notna()],
                                    [0, 1],
                                    default=np.nan
                                    )

    # Scorch 
    dataset['Scorch_feature'] = np.select(
                                        [dataset['Scorch (T3)_결과'].notna(), dataset['Scorch (T5)_결과'].notna()],
                                        [0, 1],
                                        default=np.nan
                                        )

    ######################################## Vm, MB, Scorch 통합 ########################################
    # Vm 결과 + 기준 
    dataset['Vm (T5)_결과'] = np.where(
                                    dataset['Vm (T5)_결과'].isna() & dataset['Vm (T3)_결과'].notna(), 
                                    dataset['Vm (T3)_결과'] * 1.76, 
                                    dataset['Vm (T5)_결과'] 
                                    ) 
        # Vm 결과 컬럼 생성 
    dataset['Vm_결과'] = dataset['Vm (T5)_결과'] 
        # Vm 기준 컬럼 생성 
    dataset['Vm_기준'] = dataset['Vm (T3)_기준'].combine_first(dataset['Vm (T5)_기준']) 
    
    # M/B 
    dataset['M/B 점도 (ML)_결과'] = np.where(
                                        dataset['M/B 점도 (ML)_결과'].isna() & dataset['M/B 점도 (MS)_결과'].notna(), 
                                        dataset['M/B 점도 (MS)_결과'], 
                                        dataset['M/B 점도 (ML)_결과'] 
                                        )
        # MB 결과 컬럼 생성 
    dataset['M/B_결과'] = dataset['M/B 점도 (ML)_결과'] 
        # MB 기준 컬럼 생성 
    dataset['M/B_기준'] = dataset['M/B 점도 (ML)_기준'].combine_first(dataset['M/B 점도 (MS)_기준']) 

    # Scorch
    dataset['Scorch (T3)_결과'] = np.where(
                                        dataset['Scorch (T3)_결과'].isna() & dataset['Scorch (T5)_결과'].notna(), 
                                        dataset['Scorch (T5)_결과'], 
                                        dataset['Scorch (T3)_결과'] 
                                        )
        # Scorch 결과 컬럼 생성 
    dataset['Scorch_결과'] = dataset['Scorch (T3)_결과'] 
        # Scorch 기준 컬럼 생성 
    dataset['Scorch_기준'] = dataset['Scorch (T3)_기준'].combine_first(dataset['Scorch (T5)_기준']) 

    # Drop 
    dataset = dataset.drop(columns=['Vm (T5)_결과','Vm (T3)_결과','Vm (T5)_기준','Vm (T3)_기준', 
                                    'M/B 점도 (ML)_결과','M/B 점도 (MS)_결과','M/B 점도 (ML)_기준','M/B 점도 (MS)_기준',
                                    'Scorch (T3)_결과','Scorch (T5)_결과','Scorch (T3)_기준','Scorch (T5)_기준']) 
    qa_train_df = dataset.copy()    

    return qa_df, qa_train_df

######################################################################## Prep Recipe ######################################################################## 
def prep_recipe(df, info): 
    '''
    recipe_df => 전처리 

    Returns: prep_recipe_df  
    '''
    # Copy 
    dataset = df.copy() 

    # INFO (자재코드별 원자재 비중) Merge 
    dataset = pd.merge(dataset, info, left_on="자재코드", right_on="자재코드", how="left") 

    # 투입 자재코드가 2개 이상인 '작업지시번호-배치' 제거 
    rare_batches = dataset["작업지시번호-배치"].value_counts(ascending=True)
    rare_batches = rare_batches[rare_batches <= 2].index 
        # 동일 배치 2개 이상 제거 
    dataset = dataset[~dataset["작업지시번호-배치"].isin(rare_batches)]

    # '원자재 비중' 결측값 추가 
    dataset["원자재 비중"] = dataset["원자재 비중"].fillna(1.32)

    # '작업지시번호-배치' 생성 
    dataset["작업지시번호-배치"] = (dataset["작업지시번호"] + "-" + dataset["B/T"].astype(int).astype(str))

    # 부피 생성 
    dataset['부피'] = dataset["실적중량"] / dataset["원자재 비중"] 

    # '필팩터' 계산 
    fillfactor = dataset.groupby("작업지시번호-배치").sum()["부피"] / 190

    # 필팩터 병합 
    dataset = dataset.merge(fillfactor.rename("필팩터"), 
                                left_on="작업지시번호-배치", 
                                right_index=True
                                )

    # 최종 컬럼 정리 
    dataset = dataset[["작업지시번호-배치", "필팩터"]]

    # '작업지시번호-배치'에 '필팩터' 하나 (최빈값)
    dataset = dataset.groupby("작업지시번호-배치", as_index=False).agg({"필팩터": lambda x: x.mode()[0] if not x.mode().empty else np.nan})
    
    # 정리 
    recipe_df = dataset.reset_index(drop=True).copy( )

    return recipe_df 

######################################################################## Train Test Split ########################################################################  
def create_train_test_dataset(df, target, p_type): 
    ''' 
    각 Target을 기준으로 Train Test를 분리하여 결과값을 반환한다 

    Returns: Train_dataset, Test_dataset 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True).copy() 
    dataset = dataset.dropna(subset=target)

    # Feature Selection 
    cluster_col = ['Cluster'] 
    y_col = target  

    # X Features 
    if p_type=='FMB': 
        X_col = [
            # 'step1_Ram 압력','step2_Ram 압력','step3_Ram 압력',
            'step1_Rotor speed','step2_Rotor speed','step3_Rotor speed',
            'step1_mix온도','step2_mix온도','step3_mix온도',	
            'step1_전력량','step2_전력량','step3_전력량', 
            'step1_time','step2_time','step3_time',
            '필팩터','TA_AVG','TA_MAX','TA_MIN',
            'Vm_feature', # 파생변수 
            ]
    elif p_type=='CMB': 
        X_col = [
            # 'step1_Ram 압력','step2_Ram 압력','step3_Ram 압력',
            'step1_Rotor speed','step2_Rotor speed','step3_Rotor speed',
            'step1_mix온도','step2_mix온도','step3_mix온도',	
            'step1_전력량','step2_전력량','step3_전력량', 
            'step1_time','step2_time','step3_time',
            '필팩터','TA_AVG','TA_MAX','TA_MIN',
            'MB_feature', # 파생변수 
            ]
    
    # Define Dataset 
    dataset = dataset[['작업지시번호-배치'] + X_col + cluster_col + [y_col]] 

    # 결측 제거 
    dataset = check_na_counts(dataset) 
    batch = dataset[['작업지시번호-배치']]
    
    # X, y
    X = dataset[X_col + cluster_col]
    y = dataset[y_col] 

    # Train Test Split 
    X_train, X_test, y_train, y_test, b_train, b_test = train_test_split(X, y, batch,
                                                                        test_size=0.19,
                                                                        random_state=22,
                                                                        shuffle=True,
                                                                        )

    # Scaler 제외 변수 정의     
    derived_cols = []
    if p_type == 'FMB':
        derived_cols = ['Vm_feature', 'Scorch_feature', 'Cluster']
    elif p_type == 'CMB':
        derived_cols = ['MB_feature', 'Cluster']

    # Scaler Cols 
    exclude_cols = set(derived_cols)

    # Scaler 적용 
    scale_cols = [col for col in X_train.columns if col not in exclude_cols]

    # Scaler 
    scaler = StandardScaler()
    X_train_scaled = X_train.copy() 
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols]) 

    X_test_scaled = X_test.copy() 
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])  

    # 반환용 DF 구성 (batch + X + y)
    train_df = pd.concat(
        [b_train.reset_index(drop=True),
         X_train_scaled.reset_index(drop=True),
         y_train.reset_index(drop=True)], axis=1
    ).rename(columns={'작업지시번호-배치': '작업지시번호-배치'})

    test_df = pd.concat(
        [b_test.reset_index(drop=True),
         X_test_scaled.reset_index(drop=True),
         y_test.reset_index(drop=True)], axis=1
    ).rename(columns={'작업지시번호-배치': '작업지시번호-배치'})

    return train_df, test_df, scaler, scale_cols

######################################################################## Plot Accuracy ######################################################################## 
def plot_acc(model, X_train, y_train, X_test, y_test):  
    # MAPE 
    def smape(y_true, y_pred):   
        # True Pred 
        y_true = np.asarray(y_true).astype(float) 
        y_pred = np.asarray(y_pred).astype(float)

        # 계산 
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 
        mask = denominator != 0 

        if mask.sum() == 0: 
            return np.nan 
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100 

    # 예측 
        # Train 
    y_train_pred = model.predict(X_train) 
    y_test_pred = model.predict(X_test) 
    
    # Residuals
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test  = pd.Series(y_test).reset_index(drop=True)
    y_train_pred = pd.Series(y_train_pred, index=y_train.index)
    y_test_pred  = pd.Series(y_test_pred,  index=y_test.index)
        # Residuals 
    train_residuals = y_train - y_train_pred
    test_residuals  = y_test  - y_test_pred

    ### Metrics 
        # RMSE 
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) 
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) 
        # R2 
    train_r2 = r2_score(y_train, y_train_pred) 
    test_r2 = r2_score(y_test, y_test_pred) 
        # SMAPE 
    train_smape = smape(y_train, y_train_pred) 
    test_smape = smape(y_test, y_test_pred)  

    # Subplots
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig.tight_layout(pad=5.0)

    ### 1. QQ-Plot
    stats.probplot(test_residuals, dist="norm", plot=ax[0, 0])
    ax[0, 0].set_title("QQ-Plot of Test Residuals", fontsize=14, fontweight='bold')
    ax[0, 0].grid(True, linestyle='--', alpha=0.6)

    ### 2. Residual Plot
    sns.scatterplot(x=y_test_pred, y=test_residuals, alpha=0.7, color="red", ax=ax[0, 1])
    ax[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    ax[0, 1].set_title("Residuals vs Predicted (Test)", fontsize=14, fontweight='bold')
    ax[0, 1].set_xlabel("Predicted")
    ax[0, 1].set_ylabel("Residuals")
    ax[0, 1].grid(True, linestyle='--', alpha=0.6)

    ### 3. Fitted vs Actual
    y_test = pd.Series(y_test)
    y_test_pred = pd.Series(y_test_pred, index=y_test.index)
    y_sorted = y_test.sort_values()
    y_pred_sorted = y_test_pred.loc[y_sorted.index]
    residuals = y_sorted - y_pred_sorted

    ax[1, 0].plot(range(len(y_sorted)), y_sorted, label="Actual", marker="o")
    ax[1, 0].plot(range(len(y_pred_sorted)), y_pred_sorted, label="Predicted", marker="x", linestyle="--", color="orange")

    for i in range(len(y_sorted)):
        ax[1, 0].plot([i, i], [y_sorted.iloc[i], y_pred_sorted.iloc[i]], color="gray", alpha=0.6, linestyle="--")

    ax[1, 0].set_title("Actual vs Predicted + Residuals", fontsize=14, fontweight='bold')
    ax[1, 0].set_xlabel("Sorted Index")
    ax[1, 0].set_ylabel("Value")
    ax[1, 0].grid(alpha=0.5, linestyle="--")

    ### 4. Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, color="green", ax=ax[1, 1])
    ax[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], ls='--', color='darkgreen')
    ax[1, 1].set_title("Actual vs Predicted", fontsize=14, fontweight='bold')
    ax[1, 1].set_xlabel("Actual")
    ax[1, 1].set_ylabel("Predicted")
    ax[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Model Diagnostics", fontsize=16, fontweight='bold')
    plt.show()

    ### Print Accuracy 
    print('='*100)
    print('Train Accuracy')
    print(f"Train RMSE : {train_rmse:.3f}")
    print(f"Train R2   : {train_r2:.3f}")
    print(f"Train SMAPE : {train_smape:.2f}")
    print('-'*100)
    print('Test Accuracy') 
    print(f"Test RMSE : {test_rmse:.3f}")
    print(f"Test R2   : {test_r2:.3f}")
    print(f"Test SMAPE : {test_smape:.2f}")
    print('='*100)

######################################################################## Na Counts ######################################################################## 
def check_na_counts(df, sort: bool = True, show_only_na: bool = True):
    '''
    DF -> 각 Feature na Print 

    Returns: Na Counts 
    '''
    # Copy 
    dataset = df.copy()  

    # Whole Na Counts 
    na_counts = dataset.isna().sum().reset_index()
    na_counts.columns = ["column", "na_count"]

    # Na Counts 
    if show_only_na:
        na_counts = na_counts[na_counts["na_count"] > 0]
    if sort:
        na_counts = na_counts.sort_values(by="na_count", ascending=False).reset_index(drop=True)

    # Print 
    '''
    print("="*100) 
    print(f"DataFrame 내 결측치 현황 (총 {len(df):,}개 행)") 
    print('-'*100) 
    if na_counts.empty:
        print("결측치 없음")
    else:
        for _, row in na_counts.iterrows():
            print(f"{row['column']:<30} : {int(row['na_count']):>6,} 개")
    print("="*100)
    '''

    # Blow Nas 
    dataset = dataset.dropna().reset_index(drop=True)

    return dataset

######################################################################## Modelling & Feature Importance ######################################################################## 
def train_tree(train_df, test_df, target_col, fold=False):  
    ''' 
    Optuna 학습 후 최적의 파라미터로 Model 설계 
    해당 Model 기반으로 Feature Importance 반환 

    Return: model, feature_importance_df 
    ''' 
    ################################### 필요 함수 정의 ###################################
    def valid_r2_smape(y_true, y_pred): 
        ''' 
        R2, sMAPE 결과 반환 

        Returns: R2 ,sMAPE 
        ''' 
        # y asarray 
        y_true = np.asarray(y_true, dtype=float) 
        y_pred = np.asarray(y_pred, dtype=float) 

        # R2 계산
        r2 = r2_score(y_true, y_pred) 

        # sMAPE 계산
        # denom 
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 

        # sMAPE 반환 
        mask = denom != 0 
        if mask.sum() == 0: 
            return np.nan, np.nan   

        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0 

        return r2, smape 

    ################################### Data 정리 ################################### 
    # Copy 
    train_dataset = train_df.reset_index(drop=True).copy() 
    test_dataset = test_df.reset_index(drop=True).copy() 

    # 작업지시번호-배치 Drop 
    train_dataset = train_dataset.drop(columns=['작업지시번호-배치'])
    test_dataset = test_dataset.drop(columns=['작업지시번호-배치'])

    # X, y Split 
    X_train = train_dataset.drop(columns=[target_col]) 
    y_train = train_dataset[target_col]
    X_test = test_dataset.drop(columns=[target_col]) 
    y_test = test_dataset[target_col]

    # Drop Cluster 
    if 'cluster' in X_train.columns:
        X_train = X_train.drop(columns=['cluster'])
        X_test = X_test.drop(columns=['cluster'])

    # Print 
    print(f'학습할 Tree Features: {X_train.shape[1]} | {X_train.columns.tolist()}') 
    print(f'학습 Tree 데이터 수: {X_train.shape[0]}') 

    ################################### Optuna Optimal Hyperparameter ################################### 
    # Optuna Objective 설정 및 학습 (RMSE 기준)
    if fold: 
        def objective(trial): 
            # Params 
            params =    {   
                    "verbosity": 0,
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "n_jobs": -1,

                    "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
                    "max_depth": trial.suggest_int("max_depth", 6, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0), # 트리 학습 시 사용하는 샘플 비율  

                    "gamma": trial.suggest_float("gamma", 0.0, 0.5), # 가지치기 (일반화) 
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True), # L2 정규화 항  
                }   

            # K-fold
            k_fold = KFold(n_splits=3, shuffle=True, random_state=22) 
            rmse_scores = [] 

            # Fitting 
            for train_idx, valid_idx in k_fold.split(X_train): 
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                # Fit 
                model = XGBRegressor(**params, random_state=22)
                model.fit(
                        X_tr, y_tr, 
                        eval_set=[(X_val, y_val)], 
                        verbose=False, 
                        )

                # Pred  
                y_pred = model.predict(X_val) 

                # RMSE 
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))  
                rmse_scores.append(rmse) 

            return np.mean(rmse_scores)
    else: 
        def objective(trial): 
            # Params 
            params =    {   
                    "verbosity": 0,
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "n_jobs": -1,

                    "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
                    "max_depth": trial.suggest_int("max_depth", 6, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0), # 트리 학습 시 사용하는 샘플 비율  

                    "gamma": trial.suggest_float("gamma", 0.0, 0.5), # 가지치기 (일반화) 
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True), # L2 정규화 항  
                } 

            # Train Val Test 
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.19, random_state=22) 

            # Fitting 
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False) 

            # RMSE 
            y_pred = model.predict(X_val) 
            rmse = np.sqrt(mean_squared_error(y_val, y_pred)) 

            return rmse

    # Train 
    study = optuna.create_study(direction='minimize') 
    study.optimize(objective, n_trials=10, show_progress_bar=True) 
    best_trial = min(study.best_trials, key=lambda t: t.value) 
    best_params = best_trial.params 

    # Best Param으로 초기 모델 설계 
    model = XGBRegressor(**best_params, random_state=22) 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 

    # 평가 지표 계산 
    r2, smape = valid_r2_smape(y_test, y_pred) 

    # Print  
    print(f'Tree 1차 모델 평가: R2: {round(r2,3)} | sMAPE: {round(smape,3)}') 
    plot_acc(model, X_train, y_train, X_test, y_test)   

    ################################### Feature Importance 계산 ################################### 
    # XGboost Feature Importance 
    feature_importance_df = pd.DataFrame({
                'Feature': X_train.columns, 
                'Importance': model.feature_importances_ 
                })
    
    # 중요도순 정리 
    feature_importance_df = (feature_importance_df
                            .sort_values('Importance', ascending=False)
                            .reset_index(drop=True)
                            .assign(rank=lambda df: df.index + 1)
                            )
    
    return model, feature_importance_df

######################################################################## Modelling & Feature Importance (Sample Weight) ######################################################################## 
def train_tree_sample_weight(train_df, test_df, target_col, cluster_cond, target_criterion, sample_weight):  
    ''' 
    Optuna 학습 후 최적의 파라미터로 Model 설계 
    해당 Model 기반으로 Feature Importance 반환 
        - Sample Weight 적용 -> cluster_col 데이터행만 sample weight에 넣은 후 cluster col 제거 
        - Target Sample Weight 적용 -> target_criterion 값을 기준으로 해당 최적값을 기준으로 Sample Weight 적용 
        - Sample Weight 
            1.0 → 기본 중요도
            2.0 → 다른 샘플보다 2배 중요
            0.5 → 절반 정도로 덜 중요
            5.0 → 매우 강하게 반영

    Return: model, feature_importance_df 
    ''' 
    ################################### 필요 함수 정의 ###################################
    def valid_r2_smape(y_true, y_pred): 
        ''' 
        R2, sMAPE 결과 반환 

        Returns: R2 ,sMAPE 
        ''' 
        # y asarray 
        y_true = np.asarray(y_true, dtype=float) 
        y_pred = np.asarray(y_pred, dtype=float) 

        # R2 계산
        r2 = r2_score(y_true, y_pred) 

        # sMAPE 계산
        # denom 
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 

        # sMAPE 반환 
        mask = denom != 0 
        if mask.sum() == 0: 
            return np.nan, np.nan   

        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0 

        return r2, smape 

    ################################### Data 정리 ################################### 
    # Copy 
    train_dataset = train_df.reset_index(drop=True).copy() 
    test_dataset = test_df.reset_index(drop=True).copy() 

    # 작업지시번호-배치 Drop 
    train_dataset = train_dataset.drop(columns=['작업지시번호-배치'])
    test_dataset = test_dataset.drop(columns=['작업지시번호-배치'])

    # X, y Split 
    X_train = train_dataset.drop(columns=[target_col]) 
    y_train = train_dataset[target_col]
    X_test = test_dataset.drop(columns=[target_col]) 
    y_test = test_dataset[target_col]

    # Print 
    print(f'학습할 Tree Features: {X_train.shape[1]} | {X_train.columns.tolist()}') 
    print(f'학습 Tree 데이터 수: {X_train.shape[0]}') 

    # Target 변수 지정 
    min_ratio = 0.10
    y_vals = y_train.values.astype(float)
    y_center = float(target_criterion)

    final_mask = None  

    # Sample Weight 지정 
    for std in [0.03, 0.05, 0.07]: 
        # Set Margin 
        delta = max(abs(y_center) * std, 1e-8)
        lower = y_center - delta
        upper = y_center + delta

        # Filltering Conds 
        cond_cluster = (X_train['cluster'].values == cluster_cond) 
        cond_target = (y_vals >= lower) & (y_vals <= upper)
        cond_both = cond_cluster & cond_target

        ratio = cond_both.mean()  

        # 로그(진행상황)
        print(f"Sample Weight std={std}, margin={delta:.2f}, ratio={ratio:.3%}")
    
        # 채택 조건: 10% 이상 확보되면 즉시 채택
        final_mask = cond_both

        # print(f"[INFO] k={k}, margin={margin:.4f}, ratio={ratio:.3f}")
        if ratio >= min_ratio:
            print(f"Sample Weight STD 채택 완료: ratio={ratio:.2%} (std={std})")
            break  

    # Sample Weight 최종 설정 
    sample_weights = np.ones(len(X_train), dtype=float) 
    sample_weights[final_mask] = sample_weight 
        # 정규화 
    sample_weights /= sample_weights.mean() 

    # Drop Cluster Col 
    X_train = X_train.drop(columns=['cluster'])
    X_test = X_test.drop(columns=['cluster'])

    ################################### Optuna Optimal Hyperparameter ################################### 
    # Optuna Objective 설정 및 학습 (RMSE 기준)
    def objective(trial): 
        # Params 
        params =    {   
                "verbosity": 0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": -1,

                "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0), # 트리 학습 시 사용하는 샘플 비율  

                "gamma": trial.suggest_float("gamma", 0.0, 0.5), # 가지치기 (일반화) 
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True), # L2 정규화 항  
            }   

        # K-fold
        k_fold = KFold(n_splits=3, shuffle=True, random_state=22) 
        rmse_scores = [] 

        # Fitting 
        for train_idx, valid_idx in k_fold.split(X_train): 
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            w_tr = sample_weights[train_idx]  

            # Fit 
            model = XGBRegressor(**params, random_state=22)
            model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    sample_weight=w_tr, 
                    verbose=False, 
                    )

            # Pred  
            y_pred = model.predict(X_val) 

            # RMSE 
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))  
            rmse_scores.append(rmse) 

        return np.mean(rmse_scores)

    # Train 
    study = optuna.create_study(direction='minimize') 
    study.optimize(objective, n_trials=10, show_progress_bar=True) 
    best_trial = min(study.best_trials, key=lambda t: t.value) 
    best_params = best_trial.params 

    # Best Param으로 초기 모델 설계 
    model = XGBRegressor(**best_params, random_state=22) 
    model.fit(X_train, y_train, sample_weight=sample_weights) 
    y_pred = model.predict(X_test) 

    # 평가 지표 계산 
    r2, smape = valid_r2_smape(y_test, y_pred) 

    # Print  
    print(f'Tree 1차 모델 평가: R2: {round(r2,3)} | sMAPE: {round(smape,3)}') 
    plot_acc(model, X_train, y_train, X_test, y_test)   

    ################################### Feature Importance 계산 ################################### 
    # XGboost Feature Importance 
    feature_importance_df = pd.DataFrame({
                'Feature': X_train.columns, 
                'Importance': model.feature_importances_ 
                })
    
    # 중요도순 정리 
    feature_importance_df = (feature_importance_df
                            .sort_values('Importance', ascending=False)
                            .reset_index(drop=True)
                            .assign(rank=lambda df: df.index + 1)
                            )
    
    return model, feature_importance_df

######################################################################## Feature Importance 시각화 ######################################################################## 
def plot_feature_importance(model, feature_importance_df): 
    ''' 
    Feature Importance 시각화 

    Returns: Plot 
    ''' 
    

######################################################################## Step1 Time 모델링  ######################################################################## 
def step_time_modelling(df, step_time): 
    ''' 
    기존 데이터셋에서 -> step_time에 따라 Train Dataset 생성 

    Retruns: Acc Results, Feature Importance 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True).copy() 
    step_prefix = step_time.split('_')[0] 

    # Feature Selection 
    target_col = step_time 
    common_X_cols = ['필팩터','TA_AVG','TA_MAX','TA_MIN']
    step_X_cols = [col for col in dataset.columns if col.startswith(step_prefix) and col!=target_col] 

    # 최종 컬럼 지정 
    y_col = target_col 
    X_cols =  ['작업지시번호-배치'] + step_X_cols + common_X_cols 

    # X y 구성 
    X = dataset[X_cols].copy() 
    y = dataset[y_col].copy()

    # Train Dataset 
    train_dataset = pd.concat([X,y],axis=1)
    train_dataset = train_dataset.dropna().reset_index(drop=True) 

    # Batch 
    batch = train_dataset[['작업지시번호-배치']] 

    # X y 재구성 
    X_cols = step_X_cols + common_X_cols
    y = train_dataset[y_col].copy()
    X = train_dataset[X_cols].copy() 

    # Train Test Split 
    X_train, X_test, y_train, y_test, b_train, b_test = train_test_split(X, y, batch, 
                                                                        test_size=0.22, 
                                                                        random_state=22, 
                                                                        shuffle=True, 
                                                                        )

    # Scaler 
    scaler = StandardScaler() 
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                columns=X.columns, 
                                index=X_train.index 
                                )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                columns=X.columns, 
                                index=X_test.index 
                                )     

    # To Train_df, Test_df 
        # Train df 
    train_df = pd.concat(
        [b_train.reset_index(drop=True),
         X_train_scaled.reset_index(drop=True),
         y_train.reset_index(drop=True)], axis=1
    ).rename(columns={'작업지시번호-배치': '작업지시번호-배치'})
        # Test df 
    test_df = pd.concat(
        [b_test.reset_index(drop=True),
         X_test_scaled.reset_index(drop=True),
         y_test.reset_index(drop=True)], axis=1
    ).rename(columns={'작업지시번호-배치': '작업지시번호-배치'})

    # Modelling 
    model, feature_importance = train_tree(train_df, test_df, target_col) 

    return model, feature_importance 

######################################################################## Shap Tree Feature Importance 추출  ######################################################################## 
def shap_tree(df, scaler, scaler_features, model, p_type, target_col, cluster):  
    '''
    Shap Tree 적용하여 Feature Importance 추출 
        - Cluster와 해당 부분의 Optimal Target(std 5%)에 해당하는 데이터를 원본 데이터에서 필터링하여 -> Feature Importance 추출 
        - Cluster 6개 각각마다 지표 5개에 대한 Feature Importance 보유 
        - Cluster 1에 대하여 모든 Target 지표의 최적값을 맞추는데 필요한 중요 변수를 Voting 으로 집계 -> 지표마다 따로한다면? 30 Cases 
    
    Returns: Feature Importance 
    ''' 
    ################################# 필요 컬럼 및 데이터셋 추출 #################################
    # Copy 
    dataset = df.reset_index(drop=True).copy() 
    print(f'Cluster {cluster} | {target_col} => Feature Importance 추출 시작')    

    # Cluster 
    cluster_df = dataset['Cluster'] 

    # Target Cols 
    target_prefix = target_col.split('_')[0]
    target_prefix = re.match(r'^(.*?)_', target_col).group(1).replace(' ', '')
    target_criterion = [
                        col for col in dataset.columns
                        if col.replace(' ', '').startswith(target_prefix) and "기준" in col and col != target_col
                    ][0]

    # Dataset X, y 
    X_scaled = pd.DataFrame(scaler.transform(dataset[scaler_features]), columns=scaler_features)  
    y = dataset[[target_col, target_criterion]] 

    # Concat data 
    dataset = pd.concat([cluster_df, X_scaled, y], axis=1).dropna().reset_index(drop=True).copy() 

    ################################# Cond Data Filtering (Cluster) #################################
    # Cluster 조건 
    cluster_mask = (dataset['Cluster'] == cluster)

    # Print 
    print(f'Cluster 조건 적용된 데이터 수: {dataset[cluster_mask].shape}') 

    ################################# Cond Data Filtering (Optimal Target) #################################
    # Criterion Value 조정 
    dataset[target_criterion] = (
                                dataset[target_criterion]
                                .astype(str)
                                .str.extract(r'(\d+(?:\.\d+)?)', expand=False)
                                .astype(float)
                                )

    # Otpimal Target 조건 
    std = 0.03
    center = dataset[target_criterion] 
        # Upper Lower 
    lower = center * (1-std) 
    upper = center * (1+std) 
    
    # 범위 확인 
    optimal_target_mask = (dataset[target_col] >= lower) & (dataset[target_col] <= upper) 
    optimal_target_data = dataset[optimal_target_mask] 

    # Print 
    print(f'Optimal Target 조건 적용된 데이터 수: {optimal_target_data.shape}') 

    ################################# Combine Filters ################################# 
    # Combine Filters 
    combined_mask = cluster_mask & optimal_target_mask 
    filtered_dataset = dataset.loc[combined_mask].copy()

    # Print 
    print(f'모든 조건이 필터링된 데이터 수: {filtered_dataset.shape}')

    # Drop 
    cols_to_drop = ['cluster', target_col, target_criterion]
    filtered_dataset = filtered_dataset.drop(columns=cols_to_drop, errors='ignore')

    ################################# Apply Tree Shap ################################# 
    # Tree Explainer 
    explainer = shap.TreeExplainer( 
            model, 
            data=filtered_dataset, # Cond 데이터 영역 안에서, 모델이 어떤 특성에 얼마나 반응하는가? => 데이터가 많을수록 좋다 std 확장 필요 
            # feature_perturbation="tree_path_dependent"
    )

    # Shap 값 계산 
    shap_values = explainer.shap_values(filtered_dataset)

    # Feature importance 계산 
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame({
                        'Target': target_col, 
                        "feature": scaler_features,
                        "mean_abs_shap": mean_abs_shap
                        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # 시각화를 위한 Shap Values 
    try:
        shap_values = explainer.shap_values(filtered_dataset)
    except AttributeError:
        # 최신 shap에서는 explainer(X) 형태를 권장
        shap_values = explainer(filtered_dataset).values

    return shap_values, filtered_dataset, feature_importance 

######################################################################## 패턴 시각화 ######################################################################## 
def voting_feature_importance(feature_importances): 
    ''' 
    각 Target별 Importance DF Voting 후 -> Top 7 Features 추출 

    Returns: Feature Importance 
    ''' 
    # Rank + Vote 컬럼 생성 
    long_frames = []

    for i, df in enumerate(feature_importances):
        # Target 이름: DataFrame.name 사용, 없으면 자동 생성
        target_name = df['Target'].iloc[0] 
        
        # Temp df 생성 
        temp_df = df[['feature', 'mean_abs_shap']].copy()

        # 내림차순 랭크 (값이 클수록 중요)
        temp_df['Rank'] = temp_df['mean_abs_shap'].rank(ascending=False, method='dense')

        # 투표 점수: importance * (1 / Rank)
        temp_df['Vote'] = temp_df['mean_abs_shap'] / temp_df['Rank']
        temp_df['Target'] = target_name

        # Append Temp df 
        long_frames.append(temp_df)
    
    # Concat Long Frames 
    long_df = pd.concat(long_frames, ignore_index=True) 

    # To Pivot 
    pivot_importance = long_df.pivot(
                                    index='Target',
                                    columns='feature',
                                    values='mean_abs_shap'
                                    )

    # Voting 집계 및 결과 DF 구성
    agg = (long_df
           .groupby('feature', as_index=False)
           .agg(
               Voting=('Vote', 'sum'),
               등장빈도=('Target', 'nunique'),
               평균중요도=('mean_abs_shap', 'mean'),
               관련타겟=('Target', lambda x: ', '.join(sorted(pd.unique(x))))
           ))

    # 사용자가 정의한 스키마로 맞추고, Voting으로 정렬하되 출력 컬럼은 틀 유지
    out = (agg
           .sort_values('Voting', ascending=False)
           .head(7)
           .rename(columns={
               'feature': 'Features',
               '등장빈도': '등장 빈도',
               '평균중요도': '평균 중요도',
               '관련타겟': '관련 타겟'
           })[['Features', '등장 빈도', '평균 중요도', '관련 타겟']]
           .reset_index(drop=True))

    return out

######################################################################## 제품별 클러스터 시각화  ######################################################################## 
def plot_cluster(df, product_code): 
    ''' 
    모든 제품에 대한 패턴 시각화 => 
        - 해당 제품의 배치 수 Count:
            - 20개가 넘지 않을 경우 전체 Print 
            - 20개가 넘을 경우, 전체 개수에서 비율 ratio % 랜덤 샘플링 출력 
            - 배치가 너무 많을 경우 => 연도별 상반기 하반기 나누어 그리기 

    Returns: [ 해당 배치의 시간(연-월-일) + 패턴 ] => n개 
    ''' 
    # Variables 
    sampling_ratio = 0.75

    # Copy 
    df = df[df['제품코드'].notna() & (df['제품코드'] != 0) & (df['제품코드'] != '0')] 
    df['시간'] = pd.to_datetime(df['시간'], errors='coerce')
    dataset = df.reset_index(drop=True).copy() 

    # YEAR, MONTH, 생성 
    dataset['YEAR'] = dataset['시간'].dt.year 
    dataset['MONTH'] = dataset['시간'].dt.month 
    dataset['HALF'] = np.where(dataset['MONTH'] <= 6, 'H1', 'H2') 

    # 해당 제품 필터링 
    dataset = dataset[dataset['제품코드']==product_code].reset_index(drop=True) 
    print(f'{product_code} 패턴 시각화 시작') 

    # 필요 컬럼 정의 
    need_cols = ['시간','YEAR','HALF','작업지시번호-배치','전력','전력량','Ram 압력','Rotor speed','mix온도','mix시간'] 
    dataset = dataset[need_cols].dropna(subset=['시간','작업지시번호-배치']).reset_index(drop=True)
    
    # 고유 배치 생성 및 Count 
    batches = dataset['작업지시번호-배치'].dropna().unique().tolist()
    num_batch = len(batches)

    if num_batch == 0:
        print(f"[INFO] {product_code} 유효한 배치가 없습니다.") 
        return None 
    
        # Print 
    print(f'{product_code} 전체 배치 수: {num_batch}') 
    
    ############################## 샘플링 1차 규칙 ##############################
    # 샘플링 규칙 적용 
    if num_batch > 20: 
        # Batch 랜덤 샘플링 
        whole_batches = max(1, ceil(num_batch * sampling_ratio))
        sample_batches = np.random.choice(batches, size=whole_batches, replace=False).tolist() 
        print(f"[INFO] 배치 {num_batch}개 중 {whole_batches}개 ({sampling_ratio*100}%) 랜덤 샘플링하여 시각화 시작")
    else:
        sample_batches = batches
        print(f"[INFO] 배치 {num_batch}개 전체를 시각화합니다.")

    ############################## 샘플링 2차 규칙 ##############################
    # 연도별, 상반기 하반기 대표 배치 추출 
    if len(sample_batches) > 40: 
        # 연도 x 반기별 고유 배치 샘플링 
        batch_collections = []  
        grouped = dataset.groupby(['YEAR','HALF'], dropna=False) 

        print('[INFO] 배치가 너무 많아, 연도별 상/하반기 기준으로 샘플링 시작')  
        for (yy, hh), sub in grouped: 
            unique_batches = sub['작업지시번호-배치'].dropna().unique().tolist() 
            n = len(unique_batches) 
            per_half_cap = 20
            minimum_count = min(per_half_cap, n) 

            # 해당 연도 상/하반기 Batch 수 Check 
            print(f'해당 {yy}-{hh} 배치 수: {n}') 
            if n==0: 
                print(f'해당 {yy}-{hh}에는 배치가 없습니다')
                continue 
                
            # 해당연동 상/하반기 배치 랜덤 샘플링 
            if n > minimum_count: 
                sampled = np.random.choice(unique_batches, size=minimum_count, replace=False).tolist() 
            else: 
                sampled = unique_batches 
                
            batch_collections.extend(sampled)  
            print(f"  - {yy}-{hh}: 배치 {n}개 중 {len(sampled)}개 선택")
        
        # Print 
        sample_batches = list(dict.fromkeys(batch_collections))
        print(f"[INFO] 반기별 재샘플링 후 최종 배치 수: {len(sample_batches)}")

    # Main 
    for batch in sample_batches: 
        # data 호출 
        data = dataset[dataset['작업지시번호-배치']==batch]

        # Sort Time 
        data = data.sort_values(by='시간', ascending=True) 
            # Set Time 
        data_time = data['시간'].iloc[0].strftime('%Y-%m-%d') 
        data = data.drop(columns=['시간']) 

        # 시각화 [전력량, Ram 압력, Rotor speed, mix온도, mix시간]
        plt.figure(figsize=(10,6)) 
            # Main Plot 
        plot_cols = ['전력','전력량', 'Ram 압력', 'Rotor speed', 'mix온도', 'mix시간']
        for col in plot_cols: 
            plt.plot(data[col], label=col, alpha=0.8) 
            # Title 
        plt.title(f'{data_time} | 배치 패턴: {batch}') 
        plt.xlabel('Time')
        plt.ylabel('Values') 
        plt.legend() 
        plt.grid() 
        plt.tight_layout()
        plt.show()     

######################################################################## 제품코드별 자재코드 추출 ######################################################################## 
def recipe_similarity(recipe_origin_df, p_type, union=True):    
    ''' 
    Inputs: log_df, recipe 원본 데이터셋 
    recipe_df 제품코드별 자재코드 모아서 mapping 
        - New_df 제품코드 생성  
        - 제품코드: [자재코드 1, 자재코드 2] 
        - New_df Cluster 생성 
        - New_df를 사용하여 제품코드별 유사도 측정 

    p_type = 'FMB', 'CMB' 

    Returns: Cluster별 유사도 DataFrame 
    ''' 
    # Copy  
    need_cols = ['작업지시번호-배치','작업지시번호','자재코드'] 
    dataset = recipe_origin_df[need_cols].copy()  

    # Print 
    if union: 
        print('동일 제품코드별 서로 다른 자재코드를 통합합니다') 
    else: 
        print('동일 제품코드별 서로 다른 자재코드를 분리합니다') 

    # Define p_type 
    if p_type=='FMB': 
        p_codes = [
                "FFWED70284", "FFWED70007", "FFWED70267", "FFWED70103", "FFWED70199", "FFSED70438",
                "FFWED70033", "FFWES60194", "FFSED70498", "FFSED70533", "FFWED70321",
                "FFWED70019", "FFWED70102", "FFWED70283", "FFHED70076", "FFWED70338",
                "FFHED70014", "FFSED70032", "FFHED70147", "FFHED60009", "FFHED60006"
                ]
                
    elif p_type=='CMB': 
        p_codes = [
                "HCSED50105", "HCSED60072", "HCWED60031", "HCSED50391", "HCSES60015", "HCWES60017",
                "HCSED70584", "HCSED60530", "HCSED50047", "HCSED70092", "HCSED60024", "HCSED40011",
                "HCWED70019", "FCHED60002", "FCWED70009", "HCSED60017", "HCSED70143"
                ]
    else: 
        raise ValueError('FMB, CMB 중 하나를 입력하세요') 

    # '작업지시번호-배치' 기준 자재코드 병합 
    material_df = (dataset
                .dropna(subset=['작업지시번호-배치','자재코드']) 
                .groupby('작업지시번호-배치')['자재코드']  
                .agg(lambda x: sorted(pd.unique(x))) 
                .to_frame(name='Material') 
                ).reset_index()

    # recipe_df 제품코드 생성 -> '작업지시번호-배치' Drop 
    material_df['제품코드'] = material_df['작업지시번호-배치'].astype(str).str[6:16]

        # Drop 
    material_df = material_df.drop(columns=['작업지시번호-배치']) 
    
    # p_codes 
    material_df = material_df[material_df['제품코드'].isin(p_codes)].reset_index(drop=True) 

    # 제품코드 기준 병합 -> 다른 게 있을 경우 Print 
    final_records = [] 
    conflict_codes = [] 

    # Main 
    for code, group in material_df.groupby('제품코드'): 
        # 제품코드별 고유 자재코드 
        material_sets = [set[Any](mats) for mats in group['Material']] 

        # 동일 제품코드에 Material 조합이 여러 개라면 경고 출력 
        unique_sets = [] 
        for s in material_sets: 
            if s not in unique_sets: 
                unique_sets.append(s) 

        # 동일 제품코드 안에 다른 자재코드 발견 - 하나로 합치기 
        if union: 
            if len(unique_sets) > 1: 
                conflict_codes.append(code) 
                print(f'동일 제품코드지만 서로 다른 Material 조합 존재: {code}')

                # 다른 조합의 자재코드 저장 
                merged_materials = sorted(set().union(*unique_sets))  

            else:
                # 단일 조합이면 그대로 저장
                merged_materials = sorted(list(unique_sets[0])) if unique_sets else []
            
        # 동일 제품코드 안에 다른 자재코드 발견 - 개별 분리 
        else: 
            if len(unique_sets) > 1: 
                conflict_codes.append(code) 
                print(f'동일 제품코드지만 서로 다른 Material 조합 존재: {code}')

                # 다른 조합의 자재코드 저장 
                merged_materials = unique_sets 
            
            else:
                merged_materials = unique_sets 

        # 최종 자재코드 DF 저장 
        final_records.append({'제품코드': code, 'Material': merged_materials}) 
    
    # 결과 저장 
    final_df = pd.DataFrame(final_records).sort_values('제품코드').reset_index(drop=True) 

    return final_df

######################################################################## 각 클러스터에 있는 제품코드별 자재코드를 기준으로 유사도 출력 ######################################################################## 
def cluster_simillarity(df, p_type, cluster): 
    ''' 
    Cluster 안에 각 제품코드를 기준으로 Material 유사도를 계산하는 상관관계 그래프를 작성 
        - Clsuter 컬럼 생성 
        - Clutser 안에서 제품코드 - Material 상관관계 그래프 작성 

    Return Simillarity Plot 
    ''' 
    # 함수 
    def which_cluster(code): 
        if code in cluster_1: 
            return 1 
        elif code in cluster_2: 
            return 2 
        elif code in clutser_3: 
            return 3 
        elif code in cluster_4: 
            return 4 
        else: 
            return np.nan 

    # Copy 
    dataset = df.reset_index(drop=True).copy() 

    # Clusters 
    if p_type=='FMB':
        cluster_1 = [
                    'FFWED70284','FFWED70007','FFWED70267','FFWED70338','FFHED70076','FFWES60194',
                    'FFWED70019','FFWED70321','FFHED60006','FFSED70533','FFHED60009','FFWED70103', 
                    ]
        cluster_2 = [
                    "FFWED70199","FFSED70438","FFSED70498",
                    "FFHED70147","FFWED70033","FFWED70283","FFWED70102"
                    ]
        clutser_3 = [ 
                    'FFHED70014','FFSED70032'
                    ]
        cluster_4 = [

                    ]
    elif p_type=='CMB': 
        cluster_1 = [
                    "HCWED70019", "FCHED60002", "FCWED70009", "HCSED60072",
                    "HCSED60530", "HCSED70584", "HCSED70092", "HCSED50391"
                    ]
        cluster_2 = [
                    "HCSED60024", "HCWED60031", "HCSED70143",
                    "HCSES60015", "HCWES60017", "HCSED60017"
                    ]
        clutser_3 = [ 
                    'HCSED50047','HCSED50105'
                    ]
        cluster_4 = [
                    'HCSED40011', 
                    ]
    else: 
        raise ValueError('P_type 오류: CMB, FMB 중 하나 입력')

    ############################## 클러스터 생성 ##############################
    # Clsuter Col 생성 
    dataset['Cluster'] = dataset['제품코드'].apply(which_cluster) 
        # Print 
    cluster_nas = dataset['Cluster'].isna().sum() 
    print(f'Cluster에 제외된 제품코드: {cluster_nas}') 
        # Cluster 
    dataset = dataset.loc[dataset['Cluster']==cluster, ['제품코드','Material']].reset_index(drop=True).copy() 
    cluster_df = dataset.copy()

    ############################## 유사도 그래프 생성 ##############################
    # Material 형태 보정 
    dataset['Material'] = dataset['Material'].apply(lambda x: x if isinstance(x, (list, set)) else []) 

    # Instances 
    products = dataset['제품코드'].tolist() 
    n = len(products) 
    sim_matrix = np.zeros((n,n)) 

    # 자카드유사도 계산 
    for i in range(n): 
        mats_i = set(dataset.iloc[i]['Material']) 
        for j in range(n): 
            mats_j = set(dataset.iloc[j]['Material']) 
            if len(mats_i.union(mats_j))==0: 
                sim=0 
            else:
                sim = len(mats_i.intersection(mats_j)) / len(mats_i.union(mats_j)) 
            sim_matrix[i,j] = round(sim*100, 1) 
    
    # 히트맵 시각화 
    fig, ax = plt.subplots(figsize=(0.6 * n + 4, 0.6 * n + 4)) 
    im = ax.imshow(sim_matrix, aspect='equal')  
    
    # 축 설정
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(products, rotation=45, ha="right")
    ax.set_yticklabels(products)
    ax.set_title(f"Cluster {cluster} - 제품 간 Material 유사도(%)", fontsize=20)    

    # 셀 내부에 % 표시
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i,j]:.1f}%", ha="center", va="center")

    # 컬러바
    cbar = fig.colorbar(im, shrink=0.7)
    cbar.set_label("Similarity (%)")

    plt.tight_layout()
    plt.show()

    return cluster_df, sim_matrix

######################################################################## 각 클러스터 안, 가장 빈번한 자재코드 계산 ######################################################################## 
def get_most_material(cluster_df): 
    ''' 
    모든 제품군에서 가장 많이 사용된 Material을 순위로 정리 

    Returns: DF 
    ''' 
    # Copy 
    dataset = cluster_df.copy() 

    # 모든 자재코드 펼치기 
    all_materials = [mat for sublist in dataset['Material'] for mat in sublist] 

    # 빈도 계산
    material_counts = Counter(all_materials)

    # DataFrame으로 변환 + 정렬
    result_df = (
        pd.DataFrame(material_counts.items(), columns=['Material', 'Count'])
        .sort_values(by='Count', ascending=False)
        .reset_index(drop=True)
    )

    return result_df

######################################################################## Cluster별 완전 공통 자재코드 추출 ######################################################################## 
def get_full_commons(cluster_df): 
    ''' 
    각 Cluster DF에서 모든 제품코드를 중심으로 공통된 Material 추출 

    Returns: cluster_full_commons 
    ''' 
    # Copy 
    dataset = cluster_df.reset_index(drop=True) 

    # Main 
    common_set = set(dataset.loc[0, 'Material']) 
    for i in range(1, len(dataset)): 
        materials = set(dataset.loc[i, 'Material']) 
        common_set = common_set.intersection(materials) 
        # 빈 값일 경우 
        if not common_set: 
            print('공통된 "자재코드" 없음') 
            break 

    # Results 
    cluster_full_commons = sorted(list(common_set)) 
    
    return cluster_full_commons

######################################################################## Cluster Column 생성 ######################################################################## 
def create_cluster_col(df, p_type):  
    ''' 
    Cluster Column 생성 
        - p_type = 

    Returns: DataFrame 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True).copy() 

    # Cluster Func 
    def which_cluster_fmb_bins(code): 
        if code in cluster_1: 
            return 1 
        elif code in cluster_2_1: 
            return 2.1 
        elif code in cluster_2_2: 
            return 2.2  
        elif code in clutser_3_1: 
            return 3.1
        elif code in cluster_3_2:
            return 3.2 
        else: 
            return np.nan 

    def which_cluster(code): 
        if code in cluster_1: 
            return 1 
        elif code in cluster_2: 
            return 2 
        elif code in clutser_3: 
            return 3 
        elif code in cluster_4: 
            return 4 
        else: 
            return np.nan 

    # Clusters 
    if p_type=='FMB_bins':
        cluster_1 = [
                    'FFHED60009', 
                    ]
        cluster_2_1 = [
                    "FFWES60194", "FFSED70533", "FFWED70007", "FFWED70103", "FFHED60006",
                    "FFWED70284", "FFWED70267", "FFWED70321", "FFWED70019"
                    ]
        cluster_2_2 = [
                    "FFHED70147", "FFWED70199", "FFWED70033", "FFSED70438", "FFSED70498"
                    ]
        clutser_3_1 = [ 
                    'FFHED70014', 'FFSED70032', 
                    ]
        cluster_3_2 = [
                     "FFHED70076", 'FFWED70283', 'FFWED70338', 'FFWED70102',
                    ]
    elif p_type=='FMB': 
        cluster_1 = [
                    'FFWED70284','FFWED70007','FFWED70267','FFWED70338','FFHED70076','FFWES60194',
                    'FFWED70019','FFWED70321','FFHED60006','FFSED70533','FFHED60009','FFWED70103', 
                    ]
        cluster_2 = [
                    'FFWED70199','FFSED70438','FFSED70498','FFHED70147','FFWED70033','FFWED70283','FFWED70102',
                    ]
        clutser_3 = [ 
                    'FFHED70014','FFSED70032'
                    ]
        cluster_4 = [
 
                    ]
    elif p_type=='CMB': 
        cluster_1 = [
                    "HCWED70019", "FCHED60002", "FCWED70009", "HCSED60072",
                    "HCSED60530", "HCSED70584", "HCSED70092", "HCSED50391"
                    ]
        cluster_2 = [
                    "HCSED60024", "HCWED60031", "HCSED70143",
                    "HCSES60015", "HCWES60017", "HCSED60017"
                    ]
        clutser_3 = [ 
                    'HCSED50047','HCSED50105'
                    ]
        cluster_4 = [
                    'HCSED40011', 
                    ]
    else: 
        raise ValueError('P_type 오류: CMB, FMB 중 하나 입력')

    # Get Cluster Col 
    dataset['제품코드'] = dataset['작업지시번호-배치'].map(lambda x: x[6:16]) 
    if p_type=='FMB': 
        dataset['Cluster'] = dataset['제품코드'].apply(which_cluster) 
    elif p_type=='FMB_bins':
        dataset['Cluster'] = dataset['제품코드'].apply(which_cluster_fmb_bins) 
    elif p_type=='CMB':
        dataset['Cluster'] = dataset['제품코드'].apply(which_cluster) 

    # Drop 
    dataset = dataset.drop(columns=['제품코드']) 
    dataset = dataset.dropna(subset=['Cluster']).reset_index(drop=True)\

    return dataset 

######################################################################## 모든 제품들의 Groupby 결과값 Median 산출 ######################################################################## 
def target_median(df, target_col):   
    ''' 
    결과값 기준 클러스터링을 위해 각 제품코드별 결과 중간값 산출 

    Returns: DF 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True).copy() 

    # 제품코드 
    dataset['제품코드'] = dataset['작업지시번호-배치'].map(lambda x: x[6:16]) 

    # 그룹 집계
    summary_df = (
        dataset.dropna(subset=['제품코드'])
               .groupby('제품코드', as_index=False)
               .agg(**{
                   f'{target_col}_median': (target_col, 'median'),
                   f'{target_col}_n': (target_col, lambda s: s.notna().sum())
               })
               .sort_values(by=f'{target_col}_median', ascending=True, na_position='last')
               .reset_index(drop=True)
    )
    return summary_df

######################################################################## Clustering - 1차 Vm ######################################################################## 
def cluster_vm(df, target_col):  
    ''' 
    BINS 기준에 따라 제품코드 분류 

    Returns: DF 
    ''' 
    # Copy 
    dataset = df.reset_index(drop=True).copy() 
    dataset = dataset.dropna(subset=[target_col]).reset_index(drop=True) 

    # Bins 
    bins = [15,35,48,65,np.inf] 
    labels = [1,2,3,4]     

    # Cluster Vm 생성 
    dataset['Cluster_Vm'] = pd.cut( 
                                    dataset[target_col], 
                                    bins=bins, 
                                    labels=labels, 
                                    right=True, 
                                    include_lowest=True, 
                                    ).astype('Int64') 

    return dataset 

######################################################################## Train Linear ######################################################################## 
def train_linear(train_df, test_df, target_col): 
    ''' 
    P_val, Elastic Net을 활용하여 Feature Importance 계산 및 Pred 도출 

    Returns: Model, Feature_Importance 
    ''' 
    ################################### 필요 함수 정의 ###################################
    def valid_r2_smape(y_true, y_pred): 
        ''' 
        R2, sMAPE 결과 반환 

        Returns: R2 ,sMAPE 
        ''' 
        # y asarray 
        y_true = np.asarray(y_true, dtype=float) 
        y_pred = np.asarray(y_pred, dtype=float) 

        # R2 계산
        r2 = r2_score(y_true, y_pred) 

        # sMAPE 계산
        # denom 
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 

        # sMAPE 반환 
        mask = denom != 0 
        if mask.sum() == 0: 
            return np.nan, np.nan   

        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0 

        return r2, smape 

    # Copy 
    train_dataset = train_df.reset_index(drop=True).copy() 
    test_dataset = test_df.reset_index(drop=True).copy() 

    # X_Cols 
    X_cols = [
            'step1_Ram 압력','step2_Ram 압력','step3_Ram 압력',
            'step1_Rotor speed','step2_Rotor speed','step3_Rotor speed',
            'step1_mix온도','step2_mix온도','step3_mix온도',	
            'step1_전력량','step2_전력량','step3_전력량', 
            'step1_time','step2_time','step3_time',
            '필팩터','TA_AVG','TA_MAX','TA_MIN',
            ]

    # X y 분리 
        # Train 
    X_train = train_dataset[X_cols] 
    y_train = train_dataset[target_col]  
        # Test 
    X_test = test_dataset[X_cols] 
    y_test = test_dataset[target_col] 

    # 상수항 추가 
    X_train_const = sm.add_constant(X_train, has_constant='add') 
    X_test_const = sm.add_constant(X_test, has_constant='add') 
    
    # Model Fitting 
    print(f'Linear Model 학습 시작, Features => {X_train_const.columns} | Target => {target_col}') 
    model = sm.OLS(y_train, X_train_const, missing='drop').fit() 
    
    # Pred 
    y_pred = model.predict(X_test_const) 

    # R2 & sMAPE 계산 
    r2, smape = valid_r2_smape(y_test, y_pred) 

    # Print 
    print(f'Model 평가: R2 = {r2} | sMAPE = {smape}')

    # Plot ACC 
    plot_acc(model, X_train_const, y_train, X_test_const, y_test)   


    ####################################### Feature Importance #######################################
    # P-value => Feature Selection 
    p_vals = model.pvalues.drop(labels=['const'], errors='ignore') 

    # P_value inf 제거 
    p_vals = p_vals.replace([np.inf, -np.inf], np.nan).dropna() 

    # P_value 기준 < 0.05 
    linear_selected_features = p_vals[p_vals < 0.05].index.tolist() 

    # P_value 안전망 적용 
    if len(linear_selected_features)==0:
        # Print 
        print('P_value < 0.05를 만족하는 Feature 없음 => 안전망 적용') 

        # Top 7 호출 
        top_ps = p_vals.sort_values().head(7).index.tolist()  
        linear_selected_features = X_train[top_ps].columns.tolist() 

    # Elastic Net (L1) 기준 => Feature Selection 
    elastic_model = make_pipeline(
                                ElasticNetCV(
                                    l1_ratio=[0.5, 0.8, 1.0],
                                    cv=5,
                                    random_state=22,
                                    n_alphas=100,
                                    max_iter=10000,
                                    )
                                ) 

        # Ealstic Fitting 
    elastic_model.fit(X_train[linear_selected_features], y_train)

        # Lasso 기준 Feature Selection 
    enet = elastic_model.named_steps['elasticnetcv'] 
    enet_coefs = pd.Series(enet.coef_, index=linear_selected_features) 
    enet_selected_features = enet_coefs[enet_coefs!=0].index.tolist() 

        # 없을 경우 -> 안전망 적용 
    # Elasticnet 안전망 적용
    if len(enet_selected_features) == 0:
        enet_selected_features = linear_selected_features
        print("ElasticNet에 충족한 Features 없음. P-value로 대체")

    return model, enet_selected_features 
