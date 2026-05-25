#!/usr/bin/env python3
"""Phase 3 survival validation for sheaf regulatory inconsistency.

Outputs Cox PH models, five-fold out-of-fold C-index, time-horizon AUROC/AUPRC,
likelihood-ratio tests, and diagnostics. Age is not part of SRIS; it is used only
as an external adjustment covariate.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from statsmodels.duration.hazard_regression import PHReg
warnings.filterwarnings('ignore')


def harrell_c_index(time, event, risk):
    time=np.asarray(time,float); event=np.asarray(event,int); risk=np.asarray(risk,float)
    permissible=0.0; concordant=0.0
    n=len(time)
    # O(n^2) is fine for n≈420 and avoids dependency on lifelines.
    for i in range(n):
        ti=time[i]; ei=event[i]; ri=risk[i]
        for j in range(i+1,n):
            if ti==time[j]:
                continue
            if ti<time[j] and ei==1:
                permissible += 1
                if ri>risk[j]: concordant += 1
                elif ri==risk[j]: concordant += 0.5
            elif time[j]<ti and event[j]==1:
                permissible += 1
                if risk[j]>ri: concordant += 1
                elif risk[j]==ri: concordant += 0.5
    return float(concordant/permissible) if permissible else np.nan


def bootstrap_ci(time,event,risk,n_boot=120,seed=123):
    rng=np.random.default_rng(seed); n=len(time); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,size=n)
        v=harrell_c_index(time[idx],event[idx],risk[idx])
        if np.isfinite(v): vals.append(v)
    return (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))) if vals else (np.nan,np.nan)


def clean_features(df, features):
    out=df.copy()
    keep=[]
    for f in features:
        if f not in out.columns: continue
        out[f]=pd.to_numeric(out[f], errors='coerce')
        if out[f].notna().sum()>5 and out[f].nunique(dropna=True)>1:
            keep.append(f)
    return out, keep


def prep_train_test(df, features, train=None, test=None):
    X=df[features].copy()
    if train is None:
        med=X.median(numeric_only=True).fillna(0.0)
        X=X.fillna(med)
        sc=StandardScaler()
        return sc.fit_transform(X), sc, med
    Xtr=X.iloc[train].copy(); Xte=X.iloc[test].copy()
    med=Xtr.median(numeric_only=True).fillna(0.0)
    Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
    sc=StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), sc, med


def fit_cox(time,event,X):
    try:
        res=PHReg(time,X,status=event,ties='breslow').fit(disp=0)
        return res, False
    except Exception:
        res=PHReg(time,X,status=event,ties='breslow').fit_regularized(alpha=0.05,refit=False)
        return res, True


def params_from(res,p):
    arr=np.asarray(getattr(res,'params',np.zeros(p)),float)
    if len(arr)!=p: arr=np.resize(arr,p)
    arr=np.nan_to_num(arr,nan=0,posinf=0,neginf=0)
    return arr


def fit_full(df, features, name, family):
    use=df[['os_months','deceased']+features].copy()
    use['os_months']=pd.to_numeric(use['os_months'],errors='coerce')
    use['deceased']=pd.to_numeric(use['deceased'],errors='coerce')
    use=use.dropna(subset=['os_months','deceased'])
    use=use[use['os_months']>0].reset_index(drop=True)
    use, features=clean_features(use,features)
    X,sc,med=prep_train_test(use,features)
    t=use['os_months'].values.astype(float); e=use['deceased'].values.astype(int)
    res,reg=fit_cox(t,e,X); beta=params_from(res,len(features)); risk=X@beta
    pvals=np.asarray(getattr(res,'pvalues',[np.nan]*len(features)),float) if not reg else np.array([np.nan]*len(features))
    try: conf=res.conf_int() if not reg else None
    except Exception: conf=None
    coef=[]
    for k,f in enumerate(features):
        lo=hi=np.nan
        if conf is not None:
            lo=float(np.exp(conf[k,0])); hi=float(np.exp(conf[k,1]))
        coef.append({'model':name,'model_family':family,'feature':f,'beta_per_sd':float(beta[k]),'hazard_ratio_per_sd':float(np.exp(beta[k])),'hr_ci_low':lo,'hr_ci_high':hi,'p_value':float(pvals[k]) if k<len(pvals) else np.nan,'regularized_fallback':bool(reg)})
    return pd.DataFrame(coef), {'model':name,'model_family':family,'n':int(len(use)),'events':int(e.sum()),'n_features':len(features),'log_likelihood':float(getattr(res,'llf',np.nan)) if not reg else np.nan,'in_sample_c_index':harrell_c_index(t,e,risk),'regularized_fallback':bool(reg)}, use, risk


def cv_risk(df,features,seed=43,n_splits=5):
    use=df[['patient_id','os_months','deceased']+features].copy() if 'patient_id' in df.columns else df[['os_months','deceased']+features].copy()
    use['os_months']=pd.to_numeric(use['os_months'],errors='coerce')
    use['deceased']=pd.to_numeric(use['deceased'],errors='coerce')
    use=use.dropna(subset=['os_months','deceased'])
    use=use[use['os_months']>0].reset_index(drop=True)
    use,features=clean_features(use,features)
    t=use['os_months'].values.astype(float); e=use['deceased'].values.astype(int)
    risk=np.full(len(use),np.nan)
    splitter=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed) if len(np.unique(e))>1 and min(np.bincount(e))>=n_splits else KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    splits=splitter.split(np.zeros(len(use)),e) if isinstance(splitter,StratifiedKFold) else splitter.split(np.zeros(len(use)))
    fold_rows=[]
    for fold,(tr,te) in enumerate(splits):
        Xtr,Xte,sc,med=prep_train_test(use,features,tr,te)
        res,reg=fit_cox(t[tr],e[tr],Xtr); beta=params_from(res,len(features))
        risk[te]=Xte@beta
        fold_rows.append({'fold':fold,'n_test':int(len(te)),'events_test':int(e[te].sum()),'fold_c_index':harrell_c_index(t[te],e[te],risk[te]),'regularized_fallback':bool(reg)})
    c=harrell_c_index(t,e,risk); lo,hi=(float('nan'),float('nan'))
    return use,risk,pd.DataFrame(fold_rows),{'cv_c_index':c,'cv_c_index_ci_low':lo,'cv_c_index_ci_high':hi,'mean_fold_c_index':float(pd.DataFrame(fold_rows)['fold_c_index'].mean()),'std_fold_c_index':float(pd.DataFrame(fold_rows)['fold_c_index'].std())}


def horizon_metrics(t,e,risk,h):
    t=np.asarray(t,float); e=np.asarray(e,int); risk=np.asarray(risk,float)
    cases=(e==1)&(t<=h); controls=t>h; mask=cases|controls
    y=cases[mask].astype(int); s=risk[mask]
    out={'horizon_months':h,'n_usable':int(mask.sum()),'n_cases':int(y.sum())}
    if len(np.unique(y))<2:
        out.update({'auroc':np.nan,'auprc':np.nan,'balanced_accuracy_median_threshold':np.nan,'f1_median_threshold':np.nan}); return out
    pred=(s>=np.median(s)).astype(int)
    out.update({'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)),'balanced_accuracy_median_threshold':float(balanced_accuracy_score(y,pred)),'f1_median_threshold':float(f1_score(y,pred))})
    return out


def likelihood_ratio(base,full):
    if not np.isfinite(base.get('log_likelihood',np.nan)) or not np.isfinite(full.get('log_likelihood',np.nan)): return (np.nan,np.nan,np.nan)
    df=full['n_features']-base['n_features']
    if df<=0: return (np.nan,np.nan,np.nan)
    stat=2*(full['log_likelihood']-base['log_likelihood'])
    return float(stat),int(df),float(chi2.sf(stat,df))


def km_curve(t,e):
    order=np.argsort(t); t=t[order]; e=e[order]
    times=np.unique(t[e==1]); s=1.0; xs=[0.0]; ys=[1.0]
    for tt in times:
        at=np.sum(t>=tt); d=np.sum((t==tt)&(e==1))
        if at>0: s*=1-d/at
        xs.append(float(tt)); ys.append(float(s))
    return np.array(xs),np.array(ys)


def logrank(t,e,g):
    t=np.asarray(t,float); e=np.asarray(e,int); g=np.asarray(g,int)
    O=E=V=0.0
    for tt in np.unique(t[e==1]):
        at=t>=tt; d=(t==tt)&(e==1); n=at.sum(); n1=(at&(g==1)).sum(); n0=n-n1; dd=d.sum(); d1=(d&(g==1)).sum()
        if n<=1: continue
        exp=dd*n1/n; var=n1*n0*dd*(n-dd)/(n*n*(n-1)) if n>1 else 0
        O+=d1; E+=exp; V+=var
    if V<=0: return np.nan,np.nan
    stat=(O-E)**2/V
    return float(stat),float(chi2.sf(stat,1))


def run(data_dir,results_dir,figures_dir):
    import matplotlib.pyplot as plt
    data_dir=Path(data_dir); results_dir=Path(results_dir); figures_dir=Path(figures_dir)
    results_dir.mkdir(parents=True,exist_ok=True); figures_dir.mkdir(parents=True,exist_ok=True)
    p1=pd.read_csv(data_dir/'phase1_sris_results.csv')
    p2=pd.read_csv(data_dir/'phase2_sris_all_models.csv')
    p1=p1.rename(columns={'SRIS':'SRIS_phase1','E_D_to_R':'P1_E_D_to_R','E_D_to_C':'P1_E_D_to_C','E_R_to_C':'P1_E_R_to_C'})
    cols=['patient_id','sample_id','os_months','deceased','age','grade','grade_risk','purity','kps','idh_mutant','mgmt_methylated','egfr_amp','SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']
    base=p1[cols].copy()
    for c in cols[2:]: base[c]=pd.to_numeric(base[c],errors='coerce')
    base['kps_low']=-base['kps']; base['purity_low']=-base['purity']
    p2=p2.rename(columns={'SRIS':'SRIS_phase2','E_D_to_R':'P2_E_D_to_R','E_D_to_C':'P2_E_D_to_C','E_R_to_C':'P2_E_R_to_C'})
    full=p2[['patient_id','model','variant','SRIS_phase2','P2_E_D_to_R','P2_E_D_to_C','P2_E_R_to_C']].merge(base,on='patient_id',how='left')
    cohort=base.dropna(subset=['os_months','deceased']); cohort=cohort[cohort['os_months']>0]
    cohort_summary={'n_patients':int(len(cohort)),'n_events':int(cohort['deceased'].sum()),'n_censored':int((1-cohort['deceased']).sum()),'median_os_months':float(cohort['os_months'].median()),'mean_os_months':float(cohort['os_months'].mean()),'median_age':float(cohort['age'].median()),'events_fraction':float(cohort['deceased'].mean())}
    clinical=['age','grade','purity_low','kps_low']
    molecular=clinical+['idh_mutant','mgmt_methylated','egfr_amp']
    strict=['age','purity_low','kps_low','mgmt_methylated','egfr_amp']
    nograde=['age','purity_low','kps_low','idh_mutant','mgmt_methylated','egfr_amp']
    noidh=['age','grade','purity_low','kps_low','mgmt_methylated','egfr_amp']
    model_defs=[
        ('Clinical',base,clinical,'baseline'),('Clinical + molecular',base,molecular,'baseline'),
        ('Clinical + molecular + Phase1 SRIS',base,molecular+['SRIS_phase1'],'phase1_sris'),
        ('Clinical + molecular + Phase1 edges',base,molecular+['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'],'phase1_edges'),
        ('Clinical + molecular + Phase1 SRIS+edges',base,molecular+['SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'],'phase1_all'),
        ('No-grade baseline',base,nograde,'leakage_control'),('No-grade + Phase1 edges',base,nograde+['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'],'leakage_control'),
        ('No-IDH baseline',base,noidh,'leakage_control'),('No-IDH + Phase1 edges',base,noidh+['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'],'leakage_control'),
        ('Strict no-grade/no-IDH baseline',base,strict,'strict'),('Strict no-grade/no-IDH + Phase1 edges',base,strict+['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'],'strict')]
    selected_phase2=['reference_bio_constrained_lowrisk','reference_ridge_lowrisk','identity_projection']
    for m in selected_phase2:
        if m in set(full['model']):
            sub=full[full['model']==m].drop_duplicates('patient_id').copy()
            model_defs += [(f'Clinical + molecular + Phase2 edges [{m}]',sub,molecular+['P2_E_D_to_R','P2_E_D_to_C','P2_E_R_to_C'],'phase2_edges')]
    coefs=[]; summaries=[]; folds=[]; horizons=[]; risks=[]
    for name,df,features,family in model_defs:
        c,s,use_full,r_full=fit_full(df,features,name,family)
        use_cv,r_cv,fold,cv=cv_risk(df,features)
        s.update(cv)
        summaries.append(s); coefs.append(c); fold['model']=name; folds.append(fold)
        for h in (24.0,60.0):
            hm=horizon_metrics(use_cv['os_months'].values,use_cv['deceased'].values.astype(int),r_cv,h); hm.update({'model':name,'model_family':family}); horizons.append(hm)
        tmp=use_cv[['patient_id','os_months','deceased']].copy() if 'patient_id' in use_cv.columns else use_cv[['os_months','deceased']].copy()
        tmp['model']=name; tmp['risk_oof']=r_cv; risks.append(tmp)
    coef_df=pd.concat(coefs,ignore_index=True); summary_df=pd.DataFrame(summaries); fold_df=pd.concat(folds,ignore_index=True); horizon_df=pd.DataFrame(horizons); risk_df=pd.concat(risks,ignore_index=True)
    by={r['model']:r for _,r in summary_df.iterrows()}
    pairs=[('Clinical + molecular','Clinical + molecular + Phase1 SRIS'),('Clinical + molecular','Clinical + molecular + Phase1 edges'),('Clinical + molecular','Clinical + molecular + Phase1 SRIS+edges'),('No-grade baseline','No-grade + Phase1 edges'),('No-IDH baseline','No-IDH + Phase1 edges'),('Strict no-grade/no-IDH baseline','Strict no-grade/no-IDH + Phase1 edges')]
    for m in selected_phase2:
        pairs += [('Clinical + molecular',f'Clinical + molecular + Phase2 edges [{m}]')]
    lr=[]
    for b,f in pairs:
        if b in by and f in by:
            stat,df_,p=likelihood_ratio(by[b],by[f])
            lr.append({'base_model':b,'full_model':f,'lr_statistic':stat,'df':df_,'p_value':p,'delta_cv_c_index':float(by[f]['cv_c_index']-by[b]['cv_c_index'])})
    lr_df=pd.DataFrame(lr)
    diag=[]
    for x in ['SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']:
        for y in ['age','grade','os_months']:
            mask=base[x].notna()&base[y].notna()
            rho,p=stats.spearmanr(base.loc[mask,x],base.loc[mask,y])
            diag.append({'x':x,'y':y,'spearman_rho':float(rho),'p_value':float(p),'n':int(mask.sum())})
    diag_df=pd.DataFrame(diag)
    summary_df.sort_values('cv_c_index',ascending=False).to_csv(results_dir/'phase3_survival_model_summary.csv',index=False)
    coef_df.to_csv(results_dir/'phase3_cox_coefficients.csv',index=False)
    fold_df.to_csv(results_dir/'phase3_cv_fold_metrics.csv',index=False)
    horizon_df.to_csv(results_dir/'phase3_time_horizon_accuracy.csv',index=False)
    lr_df.to_csv(results_dir/'phase3_likelihood_ratio_tests.csv',index=False)
    diag_df.to_csv(results_dir/'phase3_diagnostic_correlations.csv',index=False)
    risk_df.to_csv(results_dir/'phase3_out_of_fold_risks.csv',index=False)
    (results_dir/'phase3_cohort_summary.json').write_text(json.dumps(cohort_summary,indent=2))
    # Figures
    top=summary_df.sort_values('cv_c_index',ascending=False).head(12)
    plt.figure(figsize=(10,6)); y=np.arange(len(top)); plt.barh(y,top['cv_c_index']); plt.yticks(y,top['model'].str.replace('Clinical + molecular + ','',regex=False).str.slice(0,56)); plt.xlabel('5-fold out-of-fold Harrell C-index'); plt.title('Phase 3 survival model comparison'); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(figures_dir/'phase3_cindex_comparison.png',dpi=220); plt.close()
    subcoef=coef_df[coef_df['model']=='Clinical + molecular + Phase1 edges'].sort_values('hazard_ratio_per_sd')
    plt.figure(figsize=(8,max(4,0.35*len(subcoef)))); y=np.arange(len(subcoef)); xerr=[subcoef['hazard_ratio_per_sd']-subcoef['hr_ci_low'],subcoef['hr_ci_high']-subcoef['hazard_ratio_per_sd']]; plt.errorbar(subcoef['hazard_ratio_per_sd'],y,xerr=xerr,fmt='o'); plt.axvline(1,ls='--',lw=1); plt.yticks(y,subcoef['feature']); plt.xlabel('Hazard ratio per 1 SD'); plt.title('Adjusted Cox model: molecular baseline + Phase 1 edge energies'); plt.tight_layout(); plt.savefig(figures_dir/'phase3_phase1_edge_hazard_ratios.png',dpi=220); plt.close()
    best=summary_df.sort_values('cv_c_index',ascending=False).iloc[0]['model']; br=risk_df[risk_df['model']==best].dropna(); br['high_risk']=(br['risk_oof']>=br['risk_oof'].median()).astype(int); stat,p=logrank(br['os_months'].values,br['deceased'].values.astype(int),br['high_risk'].values)
    plt.figure(figsize=(7,5))
    for g,label in [(0,'Low predicted risk'),(1,'High predicted risk')]:
        ss=br[br['high_risk']==g]; tx,sy=km_curve(ss['os_months'].values,ss['deceased'].values.astype(int)); plt.step(tx,sy,where='post',label=f'{label} (n={len(ss)})')
    plt.xlabel('Overall survival time (months)'); plt.ylabel('Estimated survival probability'); plt.title(f'OOF risk stratification: {best[:55]}\nlog-rank p={p:.3g}'); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir/'phase3_oof_kaplan_meier_best_model.png',dpi=220); plt.close()
    sel=['Clinical + molecular','Clinical + molecular + Phase1 edges','Strict no-grade/no-IDH baseline','Strict no-grade/no-IDH + Phase1 edges']; hs=horizon_df[(horizon_df['model'].isin(sel))&(horizon_df['horizon_months'].isin([24.0,60.0]))]
    plt.figure(figsize=(10,5)); labels=[f"{r['model']} @ {int(r['horizon_months'])}m" for _,r in hs.iterrows()]; plt.bar(np.arange(len(hs)),hs['auroc']); plt.xticks(np.arange(len(hs)),labels,rotation=55,ha='right'); plt.ylabel('Out-of-fold horizon AUROC'); plt.title('Time-horizon classification from Cox risk'); plt.tight_layout(); plt.savefig(figures_dir/'phase3_horizon_auroc.png',dpi=220); plt.close()
    report=f"""# Phase 3 Survival Validation Report\n\nPatients: {cohort_summary['n_patients']}\nEvents: {cohort_summary['n_events']}\nMedian OS: {cohort_summary['median_os_months']:.2f} months\n\nBest CV model: {summary_df.sort_values('cv_c_index',ascending=False).iloc[0]['model']}\nBest CV C-index: {summary_df.sort_values('cv_c_index',ascending=False).iloc[0]['cv_c_index']:.4f}\n\nClinical + molecular CV C-index: {by['Clinical + molecular']['cv_c_index']:.4f}\nClinical + molecular + Phase 1 edges CV C-index: {by['Clinical + molecular + Phase1 edges']['cv_c_index']:.4f}\nDelta: {by['Clinical + molecular + Phase1 edges']['cv_c_index']-by['Clinical + molecular']['cv_c_index']:.4f}\n\nThis is internal survival validation. External cohort validation remains required before publication-level claims.\n"""
    (results_dir/'phase3_survival_report.md').write_text(report)
    summary={'cohort':cohort_summary,'best_model':summary_df.sort_values('cv_c_index',ascending=False).iloc[0].to_dict(),'clinical_molecular_cv_c_index':float(by['Clinical + molecular']['cv_c_index']),'phase1_edge_cv_c_index':float(by['Clinical + molecular + Phase1 edges']['cv_c_index']),'phase1_edge_delta_cv_c_index':float(by['Clinical + molecular + Phase1 edges']['cv_c_index']-by['Clinical + molecular']['cv_c_index'])}
    (results_dir/'phase3_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='data'); ap.add_argument('--results_dir',default='results'); ap.add_argument('--figures_dir',default='figures'); args=ap.parse_args()
    print(json.dumps(run(args.data_dir,args.results_dir,args.figures_dir),indent=2))
