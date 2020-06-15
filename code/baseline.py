# coding=utf-8
'''
Created on 2020年6月5日

@author: LSH
'''
import os
import gc
import time
import psutil
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import entropy, pearsonr, stats
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.width',1000)
path = "../data/"
user_data = "../user_data/"

 
def get_app_feats(df):
    print(df.head())
    print(df["busi_name"].value_counts())
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')
    tmp = df.groupby("phone_no_m")["busi_name"].agg(busi_count="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    """使用的流量统计
    """
    tmp = df.groupby("phone_no_m")["flow"].agg(flow_mean="mean", 
                                               flow_median = "median", 
                                               flow_min  = "min", 
                                               flow_max = "max", 
                                               flow_var = "var",
                                               flow_sum = "sum")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["month_id"].agg(month_ids ="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    #月流量使用统计
    phones_app["flow_month"] = phones_app["flow_sum"] / phones_app["month_ids"]

    return phones_app


def get_voc_feat(df):
    df["start_datetime"] = pd.to_datetime(df['start_datetime'] )
    df["hour"] = df['start_datetime'].dt.hour
    df["day"] = df['start_datetime'].dt.day
    print(df.head())
    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    #对话人数和对话次数
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(opposite_count="count", opposite_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    """主叫通话
    """
    df_call = df[df["calltype_id"]==1].copy()
    tmp = df_call.groupby("phone_no_m")["imei_m"].agg(voccalltype1="count", imeis="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"] 
    tmp = df_call.groupby("phone_no_m")["city_name"].agg(city_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df_call.groupby("phone_no_m")["county_name"].agg(county_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    """和固定通话者的对话统计
    """
    tmp = df.groupby(["phone_no_m","opposite_no_m"])["call_dur"].agg(count="count", sum="sum")
    phone2opposite = tmp.groupby("phone_no_m")["count"].agg(phone2opposite_mean="mean", phone2opposite_median="median", phone2opposite_max="max")
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    phone2opposite = tmp.groupby("phone_no_m")["sum"].agg(phone2oppo_sum_mean="mean", phone2oppo_sum_median="median", phone2oppo_sum_max="max")
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    
    """通话时间长短统计
    """
    tmp = df.groupby("phone_no_m")["call_dur"].agg(call_dur_mean="mean", call_dur_median="median", call_dur_max="max", call_dur_min="min")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["city_name"].agg(city_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["county_name"].agg(county_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["calltype_id"].agg(calltype_id_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    """通话时间点偏好
    """
    tmp = df.groupby("phone_no_m")["hour"].agg(voc_hour_mode = lambda x:stats.mode(x)[0][0], 
                                               voc_hour_mode_count = lambda x:stats.mode(x)[1][0], 
                                               voc_hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["day"].agg(voc_day_mode = lambda x:stats.mode(x)[0][0], 
                                               voc_day_mode_count = lambda x:stats.mode(x)[1][0], 
                                               voc_day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    return phone_no_m


def get_sms_feats(df):
    print(df.head())
    df['request_datetime'] = pd.to_datetime(df['request_datetime'] )
    df["hour"] = df['request_datetime'].dt.hour
    df["day"] = df['request_datetime'].dt.day
#     df["month"] = df['request_datetime'].dt.month

    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    #对话人数和对话次数
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(sms_count="count", sms_nunique="nunique")
    tmp["sms_rate"] = tmp["sms_count"]/tmp["sms_nunique"]
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    """短信下行比例
    """
    calltype2 = df[df["calltype_id"]==2].copy()
    calltype2 = calltype2.groupby("phone_no_m")["calltype_id"].agg(calltype_2="count")
    phone_no_m = phone_no_m.merge(calltype2, on="phone_no_m", how="left")
    phone_no_m["calltype_rate"] = phone_no_m["calltype_2"] / phone_no_m["sms_count"]
    """短信时间
    """
    tmp = df.groupby("phone_no_m")["hour"].agg(hour_mode = lambda x:stats.mode(x)[0][0], 
                                               hour_mode_count = lambda x:stats.mode(x)[1][0], 
                                               hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["day"].agg(day_mode = lambda x:stats.mode(x)[0][0], 
                                               day_mode_count = lambda x:stats.mode(x)[1][0], 
                                               day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    return phone_no_m
    
    
    
    
def train_model():
    test_app_feat=pd.read_csv(user_data+'test_app_feat.csv')
    test_voc_feat=pd.read_csv(user_data+'test_voc_feat.csv')
    test_sms_feat=pd.read_csv(user_data + "test_sms_feat.csv")
    test_user=pd.read_csv(path+'test/test_user.csv')
    test_user = test_user.merge(test_app_feat, on="phone_no_m", how="left")
    test_user = test_user.merge(test_voc_feat, on="phone_no_m", how="left")
    test_user = test_user.merge(test_sms_feat, on="phone_no_m", how="left")
    test_user["city_name"] = LabelEncoder().fit_transform(test_user["city_name"].astype(np.str))
    test_user["county_name"] = LabelEncoder().fit_transform(test_user["county_name"].astype(np.str))
    
    train_app_feat = pd.read_csv(user_data + "train_app_feat.csv")
    train_voc_feat = pd.read_csv(user_data + "train_voc_feat.csv")
    train_sms_feat=pd.read_csv(user_data + "train_sms_feat.csv")
    train_user=pd.read_csv(path+'train/train_user.csv')
    drop_r = ["arpu_201908","arpu_201909","arpu_201910","arpu_201911","arpu_201912","arpu_202001","arpu_202002"]
    train_user.drop(drop_r, axis=1,inplace=True)
    train_user.rename(columns={"arpu_202003":"arpu_202004"},inplace=True)
    train_user = train_user.merge(train_app_feat, on="phone_no_m", how="left")
    train_user = train_user.merge(train_voc_feat, on="phone_no_m", how="left")
    train_user = train_user.merge(train_sms_feat, on="phone_no_m", how="left")
    train_user["city_name"] = LabelEncoder().fit_transform(train_user["city_name"].astype(np.str))
    train_user["county_name"] = LabelEncoder().fit_transform(train_user["county_name"].astype(np.str))

    sub = test_user[["phone_no_m"]].copy()
    
    params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_bin': 50,
    'max_depth': 6,
    "learning_rate":0.02,
    "colsample_bytree":0.8,#每次迭代中随机选择特征的比例
    "bagging_fraction":0.8, #每次迭代时用的数据比例
    'min_child_samples': 25,
    'n_jobs': -1,
    'silent': True,    #信息输出设置成1则没有信息输出
    'seed':1000,
    }
    train_label = train_user[["label"]].copy()
    print(np.sum(train_label), train_label.shape, train_label.head())
    
    test_user.drop(["phone_no_m"], axis=1,inplace=True)
    train_user.drop(["phone_no_m", "label"], axis=1,inplace=True)
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1024)
    test_p = []
    threshold = 0
    scores = 0
    for i, (train_index, vaild_index) in enumerate(kf.split(train_user, train_label["label"])):
#         print('第{}次训练...'.format(i+1))
        train_x = train_user.iloc[train_index]
        train_y = train_label.iloc[train_index]
        valid_x = train_user.iloc[vaild_index]
        valid_y = train_label.iloc[vaild_index]
        
        cat = ["city_name", "county_name"]
        lgb_train = lgb.Dataset(train_x, train_y, categorical_feature=cat, silent=True)
        lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train, categorical_feature=cat, silent=True)
        
        gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train,lgb_eval], verbose_eval=100, early_stopping_rounds=200)
    
        vaild_preds = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
        bp, bestf1= score_vail(vaild_preds, valid_y)
        print("score: ", bestf1 , bp)
        scores += bestf1
        threshold += bp
#         sauc = myFevalAuc(vaild_preds, valid_y)
#         importance = gbm.feature_importance()
#         feature_name = gbm.feature_name()
#         feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by='importance',ascending=False)
#         feature_importance.to_csv('feature_importance_{}.csv'.format(idx+1),index=None,encoding='utf-8-sig')
        test_pre = gbm.predict(test_user, num_iteration=gbm.best_iteration)
        test_pre = MinMaxScaler().fit_transform(test_pre.reshape(-1, 1))
        test_pre = test_pre.reshape(-1, )
        test_p.append(test_pre)
    threshold = threshold/len(test_p)
    print(threshold)
    print("五折平均分数: ", scores/len(test_p))
    test_p = np.array(test_p)
    test_p = test_p.mean(axis=0)
    sub["prob"] = test_p
    sub["label"] = sub["prob"] > round(np.percentile(sub["prob"], threshold), 4)
    sub[["phone_no_m", "label"]].to_csv('submission.csv', index=False,encoding='utf-8') 


def score_vail(vaild_preds, real):
    """f1阈值搜索
    """
#     import matplotlib.pylab as plt
#     plt.figure(figsize=(16,5*10))
    best = 0
    bp = 0
    score = []
    for i in range(600):
        p = 32+i*0.08
        threshold_test = round(np.percentile(vaild_preds, p), 4)
        pred_int = vaild_preds>threshold_test
        ff = f1_score(pred_int,real)
        score.append(ff)
        
        if ff>=best:
            best = ff
            bp = p
#     plt.plot(range(len(score)), score)
#     plt.show()
    return bp, best


def feats():
    test_voc=pd.read_csv(path+'test/test_voc.csv',)
    test_voc_feat = get_voc_feat(test_voc)
    test_voc_feat.to_csv(user_data + "test_voc_feat.csv", index=False)

    test_app=pd.read_csv(path+'test/test_app.csv',)
    test_app_feat = get_app_feats(test_app)
    test_app_feat.to_csv(user_data + "test_app_feat.csv", index=False)
    
    test_sms=pd.read_csv(path+'test/test_sms.csv',)
    test_sms_feat = get_sms_feats(test_sms)
    test_sms_feat.to_csv(user_data + "test_sms_feat.csv", index=False)
     
    train_voc=pd.read_csv(path+'train/train_voc.csv',)
    train_voc_feat = get_voc_feat(train_voc)
    train_voc_feat.to_csv(user_data + "train_voc_feat.csv", index=False)

    train_app=pd.read_csv(path+'train/train_app.csv',)
    train_app_feat = get_app_feats(train_app)
    train_app_feat.to_csv(user_data + "train_app_feat.csv", index=False)

    train_sms=pd.read_csv(path+'train/train_sms.csv',)
    train_sms_feat = get_sms_feats(train_sms)
    train_sms_feat.to_csv(user_data + "train_sms_feat.csv", index=False)


if __name__ == "__main__":
    print("start")
    
    feats()
    train_model()
    





