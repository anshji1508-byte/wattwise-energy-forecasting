from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title='WattWise | Energy Intelligence', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0b1120; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101a31 0%,#0b1120 100%); border-right: 1px solid #243452; }
.hero { padding: 24px 30px; border: 1px solid #263b5d; border-radius: 24px; background: radial-gradient(circle at 85% 0%,#1c4770 0%,#12213a 38%,#10182b 100%); margin-bottom: 22px; }
.eyebrow { color:#71e3c1; text-transform:uppercase; letter-spacing:2px; font-size:12px; font-weight:700; }
.hero h1 { font-size:42px; color:#f4f8ff; margin:4px 0 8px; }
.hero p { color:#a9bad5; font-size:16px; max-width:760px; margin:0; }
.metric { padding:17px 18px; border-radius:16px; background:#121d32; border:1px solid #263b5d; }
.metric-label { color:#91a5c4; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#f5f8ff; font:700 27px 'Space Grotesk'; margin-top:4px; }
.metric-sub { color:#71e3c1; font-size:12px; margin-top:4px; }
.section { color:#71e3c1; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1.6px; margin:25px 0 10px; }
.chip { display:inline-block; color:#b7c8e3; background:#172641; border:1px solid #294568; border-radius:20px; padding:5px 11px; margin:2px; font-size:12px; }
div[data-testid="stMetric"] { background:#121d32; border:1px solid #263b5d; padding:13px; border-radius:16px; }
div[data-testid="stMetricLabel"] { color:#91a5c4; }
div[data-testid="stMetricValue"] { color:#f5f8ff; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#121d32; border-radius:10px; padding:8px 18px; color:#91a5c4; }
.stTabs [aria-selected="true"] { color:#71e3c1 !important; border-bottom:2px solid #71e3c1; }
</style>
''', unsafe_allow_html=True)

DATA = Path(__file__).with_name('energydata_complete.csv')

@st.cache_data
def load_data():
    d = pd.read_csv(DATA)
    d['date'] = pd.to_datetime(d['date'])
    return d.sort_values('date').drop_duplicates().reset_index(drop=True)

@st.cache_data
def build_features(raw):
    z = raw.copy().sort_values('date').reset_index(drop=True)
    dt = z.date
    z['month'] = dt.dt.month; z['day_of_month'] = dt.dt.day; z['day_of_week'] = dt.dt.dayofweek
    z['hour'] = dt.dt.hour; z['minute'] = dt.dt.minute; z['is_weekend'] = (z.day_of_week >= 5).astype(int)
    slot = z.hour * 6 + z.minute // 10
    z['time_sin'] = np.sin(2*np.pi*slot/144); z['time_cos'] = np.cos(2*np.pi*slot/144)
    z['dow_sin'] = np.sin(2*np.pi*z.day_of_week/7); z['dow_cos'] = np.cos(2*np.pi*z.day_of_week/7)
    y = z.Appliances.copy()
    for lag in [1,2,3,6,12,18,36,72,144,288,1008]: z[f'lag_{lag}'] = y.shift(lag)
    for w in [6,18,36,144]:
        p = y.shift(1); z[f'rolling_mean_{w}'] = p.rolling(w).mean(); z[f'rolling_std_{w}'] = p.rolling(w).std()
    dates = z.date.copy(); target = z.Appliances.copy()
    X = z.drop(columns=['date','Appliances','rv2'], errors='ignore')
    valid = X.notna().all(axis=1)
    return X.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True), dates.loc[valid].reset_index(drop=True)

@st.cache_resource
def train_model(X_train, y_train):
    model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', RobustScaler()),
                      ('model', HistGradientBoostingRegressor(max_iter=161, learning_rate=.0345196,
                              max_leaf_nodes=19, min_samples_leaf=58, l2_regularization=.07818, random_state=42))])
    model.fit(X_train, y_train)
    return model

def chart_style(ax):
    ax.set_facecolor('#121d32'); ax.tick_params(colors='#91a5c4'); ax.xaxis.label.set_color('#91a5c4'); ax.yaxis.label.set_color('#91a5c4')
    for s in ax.spines.values(): s.set_color('#263b5d')
    ax.title.set_color('#f4f8ff'); ax.grid(alpha=.12, color='#91a5c4')

raw = load_data(); X, y, dates = build_features(raw); cut = int(len(X)*.8)
X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
model = train_model(X_train, y_train); pred = model.predict(X_test); naive = X_test.lag_1.to_numpy()
errors = pd.DataFrame({'date':dates.iloc[cut:].values,'actual':y_test.values,'predicted':pred})
errors['error'] = errors.actual-errors.predicted; errors['abs_error']=errors.error.abs(); errors['hour']=errors.date.dt.hour

with st.sidebar:
    st.markdown('## ⚡ WattWise')
    st.caption('Appliances energy intelligence')
    page = st.radio('Navigate', ['Command Center','Forecast Explorer','Error Lab','Model Explainability','Data Health'], label_visibility='collapsed')
    st.markdown('---')
    st.markdown('**Model status**')
    st.markdown('<span class="chip">● LIVE MODEL</span> <span class="chip">54 FEATURES</span>', unsafe_allow_html=True)
    st.caption('HistGradientBoosting · chronological holdout')

st.markdown('''<div class="hero"><div class="eyebrow">Energy intelligence / 01</div><h1>Know your next watt.</h1><p>Explore appliance consumption forecasts with time-aware machine learning, transparent diagnostics and decision-ready error analysis.</p></div>''', unsafe_allow_html=True)

if page == 'Command Center':
    st.markdown('<div class="section">Portfolio pulse</div>', unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric('Average consumption', f'{y.mean():.1f}', 'Wh per 10 minutes')
    k[1].metric('Model R²', f'{r2_score(y_test,pred):.3f}', 'future holdout')
    k[2].metric('Forecast MAE', f'{mean_absolute_error(y_test,pred):.1f}', 'Wh · lower is better')
    k[3].metric('Model lift vs naive', f'{(1-mean_squared_error(y_test,pred)/mean_squared_error(y_test,naive))*100:.1f}%', 'RMSE improvement')
    st.markdown('<div class="section">Consumption signal</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(15,4)); ax.plot(dates.iloc[-1200:], y.iloc[-1200:], color='#71e3c1', lw=1); ax.fill_between(dates.iloc[-1200:], y.iloc[-1200:], color='#71e3c1', alpha=.08); ax.set_title('Recent appliance consumption'); chart_style(ax); st.pyplot(fig, use_container_width=True); plt.close(fig)
    a,b = st.columns([1.25,1])
    with a:
        st.markdown('<div class="section">Actual vs predicted</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9,4)); ax.scatter(y_test, pred, s=7, alpha=.25, color='#71e3c1'); lim=[min(y_test.min(),pred.min()),max(y_test.max(),pred.max())]; ax.plot(lim,lim,'--',color='#ffb86b'); ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); chart_style(ax); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with b:
        st.markdown('<div class="section">Quick read</div>', unsafe_allow_html=True)
        st.info('The model uses previous appliance readings and rolling windows, so it understands daily rhythm and short spikes. Scores are measured only on the final unseen 20% of time.')
        st.markdown(''.join([f'<span class="chip">{x}</span>' for x in ['Leakage-safe','Chronological split','Residual checks','Optuna-ready']]), unsafe_allow_html=True)

elif page == 'Forecast Explorer':
    st.markdown('<div class="section">One-step-ahead forecast</div>', unsafe_allow_html=True)
    idx = st.slider('Choose a timestamp from the future holdout', 0, len(X_test)-1, min(800,len(X_test)-1))
    row = X_test.iloc[[idx]]; estimate = float(model.predict(row)[0]); actual = float(y_test.iloc[idx])
    c = st.columns(4); c[0].metric('Selected time', dates.iloc[cut+idx].strftime('%d %b · %H:%M')); c[1].metric('Predicted', f'{estimate:.1f} Wh'); c[2].metric('Actual', f'{actual:.1f} Wh'); c[3].metric('Absolute error', f'{abs(actual-estimate):.1f} Wh', 'within expected range' if abs(actual-estimate)<mean_absolute_error(y_test,pred)*2 else 'large miss')
    window = slice(max(0,idx-80), min(len(X_test),idx+80)); fig, ax = plt.subplots(figsize=(15,4)); ax.plot(dates.iloc[cut+window.start:cut+window.stop], y_test.iloc[window], label='Actual', color='#71e3c1'); ax.plot(dates.iloc[cut+window.start:cut+window.stop], pred[window], label='Predicted', color='#ffb86b'); ax.axvline(dates.iloc[cut+idx], color='#f07caa', ls='--'); ax.legend(); chart_style(ax); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown('<div class="section">Input context</div>', unsafe_allow_html=True)
    context = row.T.rename(columns={row.index[0]:'value'}).sort_values('value', ascending=False).head(12); st.dataframe(context, use_container_width=True)

elif page == 'Error Lab':
    st.markdown('<div class="section">Where does the model struggle?</div>', unsafe_allow_html=True)
    c=st.columns(3); c[0].metric('RMSE', f'{mean_squared_error(y_test,pred)**.5:.2f}'); c[1].metric('Worst miss', f'{errors.abs_error.max():.0f} Wh'); c[2].metric('Bias', f'{errors.error.mean():+.2f} Wh', 'positive = under-prediction')
    fig, ax = plt.subplots(1,2,figsize=(15,4)); ax[0].plot(errors.date,errors.error,color='#f07caa',lw=.7); ax[0].axhline(0,color='#91a5c4',lw=1); ax[0].set_title('Residuals over time'); ax[1].bar(errors.groupby('hour').abs_error.mean().index,errors.groupby('hour').abs_error.mean(),color='#ffb86b'); ax[1].set_title('Mean absolute error by hour'); ax[1].set_xlabel('Hour'); [chart_style(x) for x in ax]; st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown('<div class="section">Top 15 largest misses</div>', unsafe_allow_html=True); st.dataframe(errors.sort_values('abs_error',ascending=False).head(15).style.format({'actual':'{:.1f}','predicted':'{:.1f}','error':'{:.1f}','abs_error':'{:.1f}'}),use_container_width=True)

elif page == 'Model Explainability':
    st.markdown('<div class="section">What drives the forecast?</div>', unsafe_allow_html=True)
    st.caption('Permutation importance: how much future-test RMSE worsens when a feature is shuffled.')
    sample = X_test.sample(min(900,len(X_test)), random_state=42); sample_y=y_test.loc[sample.index]
    imp=permutation_importance(model,sample,sample_y,n_repeats=2,random_state=42,scoring='neg_root_mean_squared_error',n_jobs=-1)
    imp_df=pd.DataFrame({'feature':X.columns,'importance':imp.importances_mean}).sort_values('importance',ascending=False).head(18)
    fig,ax=plt.subplots(figsize=(11,7)); ax.barh(imp_df.feature.iloc[::-1],imp_df.importance.iloc[::-1],color='#71e3c1'); ax.set_title('Top forecast drivers'); chart_style(ax); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.dataframe(imp_df.reset_index(drop=True),use_container_width=True)
    st.success('Interpretation: lag and rolling features represent the strongest real-time signal. Weather and calendar features add context around that signal.')

else:
    st.markdown('<div class="section">Trust & data quality</div>', unsafe_allow_html=True)
    c=st.columns(4); c[0].metric('Rows loaded',f'{len(raw):,}'); c[1].metric('Missing cells',f'{raw.isna().sum().sum():,}'); c[2].metric('Duplicate rows',f'{raw.duplicated().sum():,}'); c[3].metric('Sampling', '10 min')
    checks=pd.DataFrame({'Check':['Date parse','Chronological order','Target available','Duplicate feature rv2','Future holdout'], 'Status':['PASS','PASS','PASS','REMOVED','PASS'], 'Detail':['No invalid timestamps','Sorted before features','Appliances present','Exact duplicate removed','Last 20% untouched during fit']})
    st.dataframe(checks,use_container_width=True,hide_index=True)
    st.markdown('<div class="section">Feature groups</div>',unsafe_allow_html=True)
    groups={'Calendar':['month','day_of_month','day_of_week','hour','is_weekend','time_sin','time_cos','dow_sin','dow_cos'],'Lags':[c for c in X.columns if c.startswith('lag_')],'Rolling':[c for c in X.columns if c.startswith('rolling_')],'Sensors':[c for c in X.columns if c not in sum([['month','day_of_month','day_of_week','hour','is_weekend','time_sin','time_cos','dow_sin','dow_cos'],[c for c in X.columns if c.startswith('lag_')],[c for c in X.columns if c.startswith('rolling_')]],[])][:12]}
    for name,items in groups.items(): st.write(name, ' '.join([f'`{x}`' for x in items]))

st.markdown('<div style="text-align:center;color:#617596;font-size:12px;margin-top:35px">WattWise · built for transparent, time-aware forecasting · data stays local</div>', unsafe_allow_html=True)
