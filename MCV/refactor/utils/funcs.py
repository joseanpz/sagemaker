from itertools import chain, zip_longest
from functools import reduce
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
import xgboost as xgb


pd.options.display.max_rows = 4000


def predict(probs, threshold):
    preds = []
    for prob in probs:
        if prob > threshold:
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

# fval for hyperparameters
def val_func(pred_probs, dmat):
    thrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ret = []
    for thr in thrs:
        preds = predict(pred_probs, thr)
        score = f1_score(preds, dmat.get_float_info('label'))
        ret.append(('f1_score_thr{}'.format(thr), score))
    return ret

def val_func_thr(bst, dtest, test_y):
    f1_sc = 0
    max_step = 0
    thr_sample = (dtest, test_y)
    _score_preds = bst.predict(thr_sample[0])
    for thr_step in np.linspace(0, 1, 101):
        _preds = predict(_score_preds, thr_step)
        f1_sc_step = f1_score(thr_sample[1], _preds)        
        if f1_sc_step >= f1_sc:
            f1_sc = f1_sc_step
            max_step = thr_step
    # threshold = max_step
    return max_step, f1_sc  # threshold, f1score max value

def pretty_table(matrix):
    matrix_aux = chain.from_iterable(
        zip_longest(
            *(x.splitlines() for x in y),
            fillvalue='')
        for y in [[str(e) for e in row] for row in matrix])

    s = [[str(e) for e in row] for row in matrix_aux]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    
def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()

    feat_importances = []
    for ft, score in fscore.items():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    print(feat_importances.head())
    return feat_importances
    
    
def sampler1(data, periods, sample, period_samples, sample_excludes, month_size=500, seed=0):
    # print('init month size', month_size)
    if periods:
        per = periods.pop()        
        if isinstance(per, list):
            # print('in per list size', month_size)
            sample, period_samples, sample_excludes = sampler1(
                data, 
                per, 
                sample, 
                period_samples, 
                sample_excludes, 
                month_size, 
                seed
            )
            # print('month size', month_size)
            sample_excludes = [pd.Series([False]*sample.shape[0])]*len(sample_excludes)
            
            return sampler1(data, periods, sample, period_samples, sample_excludes, month_size, seed)
        
        # exclude constrain
        period = (data['FECHA'] == per) & reduce(lambda s1, s2: ~s1 & ~s2, sample_excludes)       
        local_universe = data[period].drop_duplicates('RFC')
        
        # local sample
        # print(local_universe.shape)
        local_month_size = month_size.pop() if isinstance(month_size, list) else month_size 
        # print('local month size', local_month_size)
        local_sample = local_universe.sample(local_month_size, random_state=seed)
        period_samples.append(local_sample)
        
        if not month_size:
            month_size = local_month_size
        
        sample = pd.Series(data.index.isin(local_sample.index)) | sample
        
        # set exclude
        sample_excludes.append(data['RFC'].isin(local_sample['RFC']))
        # print(len(sample_excludes))
        
        sample_excludes.pop(0)
            
        return sampler1(data, periods, sample, period_samples, sample_excludes, month_size, seed)
    else:
        return sample, period_samples, sample_excludes

    

    
def sampler(data, periods, sample, period_samples, sample_exclude, month_size=500, seed=0):
    # print('init month size', month_size)
    if periods:
        per = periods.pop()        
        if isinstance(per, list):
            # print('in per list size', month_size)
            sample, period_samples, sample_exclude = sampler(
                data, 
                per, 
                sample, 
                period_samples, 
                sample_exclude, 
                month_size, 
                seed
            )
            # print('month size', month_size)
            sample_exclude = pd.Series([False]*sample_exclude.shape[0])
            return sampler(data, periods, sample, period_samples, sample_exclude, month_size, seed)
        
        # exclude constrain
        period = (data['FECHA'] == per) & (~sample_exclude)       
        local_universe = data[period].drop_duplicates('RFC')
        
        # local sample
        # print(local_universe.shape)
        local_month_size = month_size.pop() if isinstance(month_size, list) else month_size        
        # print('local month size', local_month_size)
        local_sample = local_universe.sample(local_month_size, random_state=seed)
        period_samples.append(local_sample)
        
        if not month_size:
            month_size = local_month_size
        
        sample = pd.Series(data.index.isin(local_sample.index)) | sample
        sample_exclude = data['RFC'].isin(local_sample['RFC']) | sample_exclude
        return sampler(data, periods, sample, period_samples, sample_exclude, month_size, seed)
    else:
        return sample, period_samples, sample_exclude