import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import onnxruntime as ort
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Match Predictor Pro", page_icon="⚽", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');
.stApp{background:linear-gradient(135deg,#4a0e1e 0%,#2d0a12 50%,#1a0609 100%);font-family:'Roboto',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.main-header{background:linear-gradient(135deg,#6b1529,#4a0e1e);padding:1.5rem 2rem;border-radius:12px;margin-bottom:2rem;border:2px solid #8b1e3c;text-align:center}
.main-header h1{color:white;font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:500;margin:0;text-transform:uppercase;letter-spacing:2px}
.main-header p{color:rgba(255,255,255,0.8);font-size:1rem;margin-top:0.5rem}
.stSelectbox>div>div{background:#1e293b!important;border:1px solid #475569!important;border-radius:8px!important;color:white!important}
.stSelectbox>div>div:hover{border-color:#6b1529!important}
.stTextInput>div>div>input{background:#1e293b!important;border:1px solid #475569!important;border-radius:8px!important;color:white!important;padding:0.75rem 1rem!important}
.stTextInput>div>div>input:focus{border-color:#6b1529!important;box-shadow:0 0 0 2px rgba(107,21,41,0.3)!important}
.stDateInput>div>div{background:#1e293b!important;border-radius:8px!important}
.stDateInput>div>div>input{background:#1e293b!important;border:1px solid #475569!important;border-radius:8px!important;color:white!important}
.stDateInput label{color:#94a3b8!important;font-weight:500!important;font-size:0.85rem!important;text-transform:uppercase!important;letter-spacing:0.5px!important}
.stButton>button{background:linear-gradient(135deg,#6b1529,#4a0e1e)!important;color:white!important;border:1px solid #8b1e3c!important;border-radius:5px!important;padding:0.75rem 2rem!important;text-transform:uppercase!important;letter-spacing:1px!important}
.stButton>button:hover{background:linear-gradient(135deg,#8b1e3c,#6b1529)!important;box-shadow:0 4px 15px rgba(107,21,41,0.5)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#dc2626,#b91c1c)!important;border:1px solid #ef4444!important;font-size:1rem!important;padding:1rem 2.5rem!important}
.stSelectbox label,.stTextInput label{color:#94a3b8!important;font-weight:500!important;font-size:0.85rem!important;text-transform:uppercase!important;letter-spacing:0.5px!important}
h1,h2,h3,h4,h5,h6{color:white!important;font-family:'Oswald',sans-serif!important}
p,span,div{color:#e2e8f0}
.streamlit-expanderHeader{background:#1e293b!important;border:1px solid #475569!important;border-radius:8px!important;color:white!important}
.streamlit-expanderContent{background:#0f172a!important;border:1px solid #475569!important;border-top:none!important;border-radius:0 0 8px 8px!important}
.vs-badge{background:linear-gradient(135deg,#dc2626,#b91c1c);color:white;font-family:'Oswald',sans-serif;font-size:1.8rem;font-weight:500;padding:1rem 1.5rem;border-radius:8px;display:inline-block;box-shadow:0 8px 25px rgba(220,38,38,0.4);text-transform:uppercase;letter-spacing:2px}
.match-info-card{background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #475569;border-radius:12px;padding:1.25rem;text-align:center}
.match-date-display{font-family:'Oswald',sans-serif;font-size:1.1rem;color:#dc2626;text-transform:uppercase;letter-spacing:2px}
.match-day-display{font-size:0.85rem;color:#94a3b8;margin-top:0.25rem}
.analysis-card{background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #475569;border-radius:8px;padding:1.5rem;margin:0.5rem 0}
.analysis-title{font-family:'Oswald',sans-serif;font-size:1.1rem;font-weight:600;color:#dc2626;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem}
.analysis-item{display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05)}
.analysis-label{color:#94a3b8}.analysis-value{color:white;font-weight:600}
[data-testid="column"]{padding:0.5rem}
hr{border-color:rgba(255,255,255,0.1)!important;margin:1.5rem 0!important}
.xi-box{background:linear-gradient(145deg,#4a0e1e,#2d0a12);border:2px solid #6b1529;border-radius:12px;padding:1.25rem;margin-bottom:0.5rem}
.xi-title{font-family:'Oswald',sans-serif;font-size:1.3rem;font-weight:700;color:white;text-transform:uppercase;letter-spacing:3px;margin-bottom:0.75rem}
.xi-team{display:inline-block;background:linear-gradient(135deg,#1e293b,#334155);border:2px solid #475569;border-radius:30px 0 0 30px;padding:0.4rem 1.2rem;font-family:'Oswald',sans-serif;font-size:1rem;font-weight:600;color:white;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem}
.pitch{background:linear-gradient(180deg,#1a6b32 0%,#1f7a3a 10%,#1a6b32 20%,#1f7a3a 30%,#1a6b32 40%,#1f7a3a 50%,#1a6b32 60%,#1f7a3a 70%,#1a6b32 80%,#1f7a3a 90%,#1a6b32 100%);border-radius:8px;position:relative;height:280px;border:3px solid rgba(255,255,255,0.3);box-shadow:inset 0 0 30px rgba(0,0,0,0.4)}
.pitch::before{content:'';position:absolute;top:50%;left:5%;right:5%;height:2px;background:rgba(255,255,255,0.4)}
.pitch::after{content:'';position:absolute;top:calc(50% - 35px);left:50%;transform:translateX(-50%);width:70px;height:70px;border:2px solid rgba(255,255,255,0.4);border-radius:50%}
.pj{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:3px}
.pj-s{width:30px;height:30px;clip-path:polygon(20% 0%,80% 0%,100% 15%,100% 100%,0% 100%,0% 15%)}
.pj-s.of{background:linear-gradient(180deg,#2563eb,#1d4ed8)}.pj-s.gk{background:linear-gradient(180deg,#eab308,#ca8a04)}
.pj-n{font-size:0.55rem;font-weight:600;color:white;text-shadow:1px 1px 2px rgba(0,0,0,0.8);max-width:60px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-transform:uppercase}
.pl-r{display:flex;align-items:center;margin:2px 0}
.pl-num{background:linear-gradient(135deg,#1e293b,#0f172a);color:#94a3b8;font-family:'Oswald',sans-serif;font-size:0.8rem;font-weight:600;min-width:26px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:3px 0 0 3px}
.pl-bar{flex:1;background:linear-gradient(135deg,#475569,#334155 40%,#1e293b);height:24px;display:flex;align-items:center;padding:0 8px;clip-path:polygon(0 0,100% 0,95% 100%,0% 100%)}
.pl-nm{font-size:0.7rem;font-weight:500;color:white;text-transform:uppercase;letter-spacing:0.5px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_prediction_data():
    ps = pd.read_csv('dataset/epl_player_stats_24_25.csv'); ps.columns = ps.columns.str.strip()
    if 'Rating' in ps.columns: ps['Rating'] = pd.to_numeric(ps['Rating'], errors='coerce').fillna(6.5)
    return ps, pd.read_csv('dataset/player_injuries_impact.csv')

@st.cache_data
def load_app_data():
    with open("streamlit_app_data.pkl","rb") as f: return pickle.load(f)

@st.cache_resource
def load_model():
    p = "match_predictor_model.onnx"
    return ort.InferenceSession(p) if os.path.exists(p) else None

def predict_with_model(s, d):
    return s.run([s.get_outputs()[0].name], {s.get_inputs()[0].name: d.astype(np.float32)})[0]

PLAYER_STATS, INJURY_DATA = load_prediction_data()

def get_player_features(pn, club):
    f = {'rating':6.5,'goals':0,'assists':0,'is_injured':0,'injury_severity':0}
    if not pn or pd.isna(pn): return f
    pr = PLAYER_STATS[(PLAYER_STATS['Player Name'].str.lower()==pn.lower())|(PLAYER_STATS['Player Name'].str.lower().str.contains(pn.lower(),na=False))]
    if len(pr)>0:
        r=pr.iloc[0]; f['rating']=float(r.get('Rating',6.5)) if pd.notna(r.get('Rating')) else 6.5
        f['goals']=int(r.get('Goals',0)) if pd.notna(r.get('Goals')) else 0; f['assists']=int(r.get('Assists',0)) if pd.notna(r.get('Assists')) else 0
    inj = INJURY_DATA[INJURY_DATA['Name'].str.lower().str.contains(pn.lower(),na=False)]
    if len(inj)>0:
        f['is_injured']=1
        for c in ['Match1_missed_match_Result','Match2_missed_match_Result','Match3_missed_match_Result']:
            if c in inj.columns: f['injury_severity']+=inj[c].notna().sum()
    return f

def calc_strength(lineup, tn):
    tr=tg=ta=ic=iv=pc=0
    for _,p in lineup.items():
        if p:
            f=get_player_features(p,tn); tr+=f['rating']; tg+=f['goals']; ta+=f['assists']; ic+=f['is_injured']; iv+=f['injury_severity']; pc+=1
    if pc==0: pc=1
    return {'avg_rating':tr/pc,'total_goals':tg,'total_assists':ta,'injured_count':ic,'injury_severity':iv,'player_count':pc}

def gen_features(h,a):
    f=np.zeros(20,dtype=np.float32)
    f[0]=(h['avg_rating']-a['avg_rating'])/2.0; f[1]=h['total_goals']/50.0; f[2]=a['total_goals']/50.0
    f[3]=h['total_assists']/30.0; f[4]=a['total_assists']/30.0; f[5]=-h['injured_count']*0.1; f[6]=-a['injured_count']*0.1
    f[7]=-h['injury_severity']*0.05; f[8]=-a['injury_severity']*0.05; f[9]=0.1; f[10]=h['avg_rating']/10.0; f[11]=a['avg_rating']/10.0
    f[12]=(h['total_goals']+h['total_assists'])/60.0; f[13]=(a['total_goals']+a['total_assists'])/60.0; f[14:]=0.5
    return f.reshape(1,-1)

def sched_factor(md):
    dw=md.weekday(); mw=dw in[1,2,3]; ff=0.03 if mw else 0.0; m=md.month
    cf=0.05 if m==12 else(0.03 if m in[1,2] else(0.02 if m in[4,5] else 0.0))
    ss=date(md.year if md.month>=8 else md.year-1,8,1); sp=min(1.0,max(0.0,(md-ss).days/300.0))
    return {'is_midweek':mw,'fatigue_factor':ff,'congestion_factor':cf,'season_progress':sp,'day_name':md.strftime('%A'),'date_display':md.strftime('%d %B %Y')}

def pred_scorers(lineup,tn,ng):
    import random; random.seed(hash(tn)%(2**32))
    if ng<=0: return []
    fw=['LW','ST','RW','ST1','ST2']; md=['CM1','CM2','CM3','LM','RM','DM1','DM2','AM']
    ps=[]
    for pos,p in lineup.items():
        if not p: continue
        f=get_player_features(p,tn); pw=2.0 if pos in fw else(0.3 if pos in md else 0.1)
        ps.append((p,min(0.8,(f['goals']/20.0)*pw*(f['rating']/7.0))))
    ps.sort(key=lambda x:x[1],reverse=True); sc=[]; rem=ng
    for p,pr in ps:
        if rem<=0: break
        if random.random()<pr*1.5 or(rem>0 and pr>0.3): sc.append(p); rem-=1
    while rem>0 and ps:
        p,_=ps[min(rem,len(ps)-1)]
        if p not in sc: sc.append(p); rem-=1
    return sc[:ng]

FORMATION_POSITIONS={"4-3-3":[["GK"],["LB","CB1","CB2","RB"],["CM1","CM2","CM3"],["LW","ST","RW"]],"4-4-2":[["GK"],["LB","CB1","CB2","RB"],["LM","CM1","CM2","RM"],["ST1","ST2"]],"4-2-3-1":[["GK"],["LB","CB1","CB2","RB"],["DM1","DM2"],["LW","AM","RW"],["ST"]]}
POS_MAP={"GK":"GKP","LB":"DEF","CB1":"DEF","CB2":"DEF","RB":"DEF","CM1":"MID","CM2":"MID","CM3":"MID","LM":"MID","RM":"MID","DM1":"MID","DM2":"MID","AM":"MID","LW":"FWD","ST":"FWD","RW":"FWD","ST1":"FWD","ST2":"FWD"}
PITCH_COORDS={"4-3-3":{"GK":(50,90),"LB":(15,70),"CB1":(35,72),"CB2":(65,72),"RB":(85,70),"CM1":(25,45),"CM2":(50,50),"CM3":(75,45),"LW":(15,20),"ST":(50,15),"RW":(85,20)},"4-4-2":{"GK":(50,90),"LB":(15,70),"CB1":(35,72),"CB2":(65,72),"RB":(85,70),"LM":(15,45),"CM1":(38,48),"CM2":(62,48),"RM":(85,45),"ST1":(35,18),"ST2":(65,18)},"4-2-3-1":{"GK":(50,90),"LB":(15,70),"CB1":(35,72),"CB2":(65,72),"RB":(85,70),"DM1":(35,52),"DM2":(65,52),"LW":(15,32),"AM":(50,35),"RW":(85,32),"ST":(50,12)}}

def sn(fn):
    if not fn: return ""
    p=fn.split(); return p[-1].upper()[:10] if len(p)>=2 else fn.upper()[:10]

def auto_fill(fm, tpd):
    sel,used={},set(); vp={k:v for k,v in tpd.items() if k and k.strip()}
    for pk in[p for r in FORMATION_POSITIONS[fm] for p in r]:
        rp=POS_MAP.get(pk,""); el=[(p,pos) for p,pos in vp.items() if pos==rp and p not in used]
        if el: pl=sorted(el,key=lambda x:x[0])[0][0]; sel[pk]=pl; used.add(pl)
        else:
            av=[p for p in vp.keys() if p not in used]
            if av: pl=sorted(av)[0]; sel[pk]=pl; used.add(pl)
            else: sel[pk]=""
    return sel

def render_xi(tn, fm, lineup):
    coords=PITCH_COORDS.get(fm,PITCH_COORDS["4-3-3"])
    ph=""
    for pos,(x,y) in coords.items():
        s="gk" if pos=="GK" else "of"
        ph+=f'<div class="pj" style="left:{x}%;top:{y}%;"><div class="pj-s {s}"></div><div class="pj-n">{sn(lineup.get(pos,""))}</div></div>'
    op=[(pos,lineup[pos]) for r in FORMATION_POSITIONS[fm] for pos in r if pos in lineup and lineup[pos]]
    lh="".join(f'<div class="pl-r"><div class="pl-num">{i:02d}</div><div class="pl-bar"><span class="pl-nm">{sn(p)}</span></div></div>' for i,(pos,p) in enumerate(op,1))
    st.markdown(f'<div class="xi-box"><div style="display:flex;gap:1rem;flex-wrap:wrap;"><div style="flex:1.2;min-width:240px;"><div class="xi-title">Starting XI</div><div class="xi-team">{tn}</div><div class="pitch">{ph}</div></div><div style="flex:0.8;min-width:160px;display:flex;flex-direction:column;justify-content:center;">{lh}</div></div></div>', unsafe_allow_html=True)

def render_panel(side, teams, pdata):
    sk=side.lower(); team=st.selectbox(f"Select {side} Team",teams,key=f"{sk}_t")
    fm=st.selectbox(f"{side} Formation",list(FORMATION_POSITIONS.keys()),key=f"{sk}_f")
    tpd=pdata.get(team,{}); roster=sorted([p for p in tpd.keys() if p])
    lk=f"{sk}_lineup"; pt=st.session_state.get(f"{sk}_pt"); pf=st.session_state.get(f"{sk}_pf")
    if lk not in st.session_state or pt!=team or pf!=fm: st.session_state[lk]=auto_fill(fm,tpd)
    st.session_state[f"{sk}_pt"]=team; st.session_state[f"{sk}_pf"]=fm
    if st.button("Auto-fill Lineup",key=f"{sk}_af"): st.session_state[lk]=auto_fill(fm,tpd); st.rerun()
    with st.expander("Edit Lineup",expanded=False):
        sel,used={},set()
        for row in FORMATION_POSITIONS[fm]:
            cols=st.columns(len(row))
            for i,pk in enumerate(row):
                with cols[i]:
                    rp=POS_MAP.get(pk,""); el=[p for p,pos in tpd.items() if pos==rp and p not in used]
                    opts=sorted(el) if el else roster; cur=st.session_state[lk]
                    di=opts.index(cur[pk]) if pk in cur and cur[pk] in opts else 0
                    sel[pk]=st.selectbox(pk,opts,index=di,key=f"{sk}_{pk}"); used.add(sel[pk])
        st.session_state[lk]=sel
    lineup=st.session_state[lk]; render_xi(team,fm,lineup); return team,lineup


st.markdown('<div class="main-header"><h1>⚽ Match Predictor Pro</h1><p>Football Match Predictions with Player Analysis</p></div>', unsafe_allow_html=True)
data=load_app_data(); model=load_model()
st.markdown("### 📅 Match Details")
dc1,dc2,dc3=st.columns([2,2,1])
with dc1: match_date=st.date_input("Match Date",value=date.today(),min_value=date(2008,8,1),max_value=date(2026,12,31),key="md",help="Select the match date.")
sc=sched_factor(match_date)
with dc2:
    cg=" · ⚠️ Congestion period" if sc['congestion_factor']>0.03 else ""
    st.markdown(f'<div class="match-info-card"><div class="match-date-display">{sc["date_display"]}</div><div class="match-day-display">{sc["day_name"]} {"· Midweek Fixture" if sc["is_midweek"] else "· Weekend Fixture"}</div><div style="margin-top:0.5rem;font-size:0.8rem;color:#64748b;">Season progress: {sc["season_progress"]*100:.0f}%{cg}</div></div>', unsafe_allow_html=True)
with dc3:
    fv=sc['fatigue_factor']+sc['congestion_factor']
    st.markdown(f'<div style="background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #475569;border-radius:12px;padding:1rem;text-align:center;margin-top:0.5rem;"><div style="font-family:Oswald,sans-serif;color:#dc2626;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Fatigue</div><div style="color:white;font-size:1.5rem;font-weight:700;font-family:Oswald,sans-serif;">{"+" if fv>0 else ""}{fv*100:.0f}%</div><div style="font-size:0.7rem;color:#64748b;">Schedule impact</div></div>', unsafe_allow_html=True)
st.divider()
left,right=st.columns(2)
with left: h_team,h_lineup=render_panel("Home",data['teams'],data['players_info'])
with right: a_team,a_lineup=render_panel("Away",[t for t in data['teams'] if t!=h_team],data['players_info'])
st.markdown('<div style="text-align:center;margin:2rem 0;"><span class="vs-badge">VS</span></div>', unsafe_allow_html=True)

if st.button("PREDICT MATCH RESULT",type="primary",use_container_width=True):
    if model is None: st.error("Model not found!")
    else:
        hs=calc_strength(h_lineup,h_team); ast2=calc_strength(a_lineup,a_team)
        with st.expander("📊 Team Analysis",expanded=True):
            c1,c2=st.columns(2)
            for col,tn,s,ic in[(c1,h_team,hs,"🏠"),(c2,a_team,ast2,"✈️")]:
                with col:
                    ijc='#ef4444' if s['injured_count']>0 else '#10b981'
                    st.markdown(f'<div class="analysis-card"><div class="analysis-title">{ic} {tn}</div><div class="analysis-item"><span class="analysis-label">Average Rating</span><span class="analysis-value">{s["avg_rating"]:.2f}</span></div><div class="analysis-item"><span class="analysis-label">Total Goals</span><span class="analysis-value">{s["total_goals"]}</span></div><div class="analysis-item"><span class="analysis-label">Total Assists</span><span class="analysis-value">{s["total_assists"]}</span></div><div class="analysis-item"><span class="analysis-label">Injured Players</span><span class="analysis-value" style="color:{ijc};">{s["injured_count"]}</span></div></div>', unsafe_allow_html=True)
        mf=gen_features(hs,ast2); df=data['default_features']
        if df.shape[1]>mf.shape[1]:
            pd2=np.zeros((1,df.shape[1]),dtype=np.float32); pd2[0,:mf.shape[1]]=mf[0]; pd2[0,mf.shape[1]:]=df[0,mf.shape[1]:]
        else: pd2=mf
        bp=float(predict_with_model(model,pd2).flatten()[0])
        rd=(hs['avg_rating']-ast2['avg_rating'])/10.0; ha=(hs['total_goals']+hs['total_assists'])/60.0; aa=(ast2['total_goals']+ast2['total_assists'])/60.0
        ii=(ast2['injured_count']*0.1+ast2['injury_severity']*0.05)-(hs['injured_count']*0.1+hs['injury_severity']*0.05)
        prob=max(0.05,min(0.95,bp*0.3+0.35+rd*0.2+(ha-aa)*0.1+ii+0.08-sc['fatigue_factor']*0.5-sc['congestion_factor']*0.3))
        if prob>=0.75: oc="Home Win"; hg=min(4,int(2+ha*3)); ag=min(3,int(1+aa*2+(1 if prob<0.85 else 0)))
        elif prob>=0.5: oc="Home Win" if prob>0.6 else "Draw"; hg=min(3,int(1+ha*2)); ag=min(2,int(1+aa*2))
        elif prob>=0.35: oc="Draw"; hg=1; ag=1
        elif prob>=0.2: oc="Away Win"; hg=min(2,int(1+ha)); ag=min(3,int(2+aa*2))
        else: oc="Away Win"; hg=min(1,int(ha*2)); ag=min(4,int(2+aa*3))
        hsc=pred_scorers(h_lineup,h_team,hg); asr=pred_scorers(a_lineup,a_team,ag)
        import random; random.seed(hash(h_team+a_team)%(2**32))
        def gt(n):
            ts=sorted([random.randint(1,90)+(random.choice([0,1,2,3,4]) if random.random()<0.15 else 0) for _ in range(n)])
            return[f"{t}'" if t<=90 else f"90'+{t-90}'" for t in ts]
        htx,atx=gt(len(hsc)) if hsc else [],gt(len(asr)) if asr else []
        def fs(sc2,tx):
            if not sc2: return ""
            t=[f"{s.split()[-1]} {tx[i] if i<len(tx) else ''}" for i,s in enumerate(sc2)]
            return(f"{', '.join(t[:2])} <span style='color:#3b82f6;'>+{len(t)-2} more</span>" if len(t)>3 else ", ".join(t))
        hst,astt=fs(hsc,htx),fs(asr,atx)
        hwp=prob*100 if oc=="Home Win" else(100-prob*100)/2 if oc=="Draw" else(1-prob)*30
        awp=(1-prob)*100 if oc=="Away Win" else(100-prob*100)/2 if oc=="Draw" else prob*30
        dwp=100-hwp-awp; tot=hwp+dwp+awp; hwp,dwp,awp=(hwp/tot)*100,(dwp/tot)*100,(awp/tot)*100
        mb='<span style="display:inline-block;background:#475569;color:#e2e8f0;padding:0.2rem 0.6rem;border-radius:4px;font-size:0.7rem;margin-left:0.5rem;">MIDWEEK</span>' if sc['is_midweek'] else ''
        st.markdown(f"""<div style="background:linear-gradient(145deg,#f8fafc,#e2e8f0);border-radius:16px;padding:2rem;margin:2rem 0;box-shadow:0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align:center;margin-bottom:1rem;"><span style="font-size:0.85rem;color:#6b7280;">{sc['date_display']} · {sc['day_name']}{mb}</span></div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div style="display:flex;align-items:center;gap:1rem;flex:1;"><div style="display:flex;align-items:center;gap:0.5rem;"><span style="color:#374151;font-size:0.7rem;">▶</span><div style="width:50px;height:50px;background:linear-gradient(135deg,#1e3a8a,#3b82f6);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:0.8rem;">{h_team[:3].upper()}</div></div><span style="font-size:1.4rem;font-weight:600;color:#1f2937;">{h_team}</span></div>
                <div style="text-align:center;padding:0 2rem;"><div style="font-size:4rem;font-weight:800;color:#111827;letter-spacing:-2px;">{hg} <span style="color:#9ca3af;font-weight:400;">-</span> {ag}</div><div style="font-size:0.85rem;color:#6b7280;margin-top:-0.5rem;">Match Prediction</div></div>
                <div style="display:flex;align-items:center;gap:1rem;flex:1;justify-content:flex-end;"><span style="font-size:1.4rem;font-weight:600;color:#1f2937;">{a_team}</span><div style="width:50px;height:50px;background:linear-gradient(135deg,#7c3aed,#a78bfa);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:0.8rem;">{a_team[:3].upper()}</div></div></div>
            <div style="display:flex;justify-content:space-between;padding:1rem 0;border-top:1px solid #e5e7eb;margin-top:0.5rem;"><div style="flex:1;text-align:left;"><span style="font-size:0.9rem;color:#4b5563;">{hst}</span></div><div style="text-align:center;padding:0 1rem;"><span style="font-size:1.2rem;color:#9ca3af;">⚽</span></div><div style="flex:1;text-align:right;"><span style="font-size:0.9rem;color:#4b5563;">{astt}</span></div></div>
            <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e5e7eb;"><div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;"><span style="font-size:1rem;font-weight:600;color:#3b82f6;">{hwp:.1f}%</span><span style="font-size:0.85rem;color:#6b7280;">Win probability</span><span style="font-size:1rem;font-weight:600;color:#6b7280;">{awp:.1f}%</span></div><div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#e5e7eb;"><div style="width:{hwp}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);"></div><div style="width:{dwp}%;background:#d1d5db;"></div><div style="width:{awp}%;background:linear-gradient(90deg,#9ca3af,#6b7280);"></div></div></div>
            <div style="text-align:center;margin-top:1.5rem;"><span style="display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:white;padding:0.6rem 1.5rem;border-radius:30px;font-weight:600;font-size:0.95rem;box-shadow:0 4px 15px rgba(16,185,129,0.4);">Predicted: {oc}</span></div></div>""", unsafe_allow_html=True)
