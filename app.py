import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

st.set_page_config(page_title="AI Smart Slum Upgradation Planner", layout="wide")
FEATURES = ["population_density","service_access","infrastructure_gap",
            "environmental_risk","basic_service_distance",
            "economic_vulnerability","health_risk"]

@st.cache_data
def demo_data(n=500, seed=42):
    rng=np.random.default_rng(seed)
    d=pd.DataFrame({
        "area_id":[f"AREA_{i+1:04d}" for i in range(n)],
        "latitude":rng.uniform(30.60,30.78,n),
        "longitude":rng.uniform(76.65,76.90,n),
        "population_density":rng.uniform(100,1500,n),
        "service_access":rng.uniform(20,95,n),
        "infrastructure_gap":rng.uniform(10,95,n),
        "environmental_risk":rng.uniform(5,95,n),
        "basic_service_distance":rng.uniform(.1,5,n),
        "economic_vulnerability":rng.uniform(10,95,n),
        "health_risk":rng.uniform(5,95,n)})
    s=(.28*d.infrastructure_gap/100+.20*d.environmental_risk/100+
       .18*d.economic_vulnerability/100+.16*d.health_risk/100+
       .12*d.population_density/1500+.10*(1-d.service_access/100))
    d["priority"]=(s>=s.quantile(.68)).astype(int)
    d["priority_score"]=(s*100).round(2)
    return d

def train(d):
    X=d[FEATURES]; y=d["priority"]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    scaler=StandardScaler(); XtS=scaler.fit_transform(Xt); XvS=scaler.transform(Xv)
    models={
      "Random Forest":RandomForestClassifier(n_estimators=250,max_depth=10,
          random_state=42,class_weight="balanced"),
      "Gradient Boosting":GradientBoostingClassifier(n_estimators=180,
          learning_rate=.05,max_depth=3,random_state=42),
      "SVM":SVC(kernel="rbf",C=2,probability=True,class_weight="balanced",
          random_state=42)}
    rows=[]; fitted={}
    for name,m in models.items():
        if name=="SVM":
            m.fit(XtS,yt); p=m.predict(XvS)
        else:
            m.fit(Xt,yt); p=m.predict(Xv)
        rows.append({"Model":name,
          "Accuracy":round(accuracy_score(yv,p)*100,2),
          "Precision":round(precision_score(yv,p,zero_division=0)*100,2),
          "Recall":round(recall_score(yv,p,zero_division=0)*100,2),
          "F1":round(f1_score(yv,p,zero_division=0)*100,2)})
        fitted[name]=m
    return pd.DataFrame(rows),fitted,scaler

def scenarios(d):
    out=[]
    for name,gap,access,risk in [
        ("Basic infrastructure",.20,5,0),
        ("Comprehensive service improvement",.35,15,.10),
        ("Mixed targeted intervention",.45,20,.20)]:
        x=d.copy()
        x["infrastructure_gap"]*=1-gap
        x["service_access"]=np.minimum(100,x.service_access+access)
        x["environmental_risk"]*=1-risk
        x["health_risk"]*=1-risk
        s=(.28*x.infrastructure_gap/100+.20*x.environmental_risk/100+
           .18*x.economic_vulnerability/100+.16*x.health_risk/100+
           .12*x.population_density/1500+.10*(1-x.service_access/100))
        out.append({"Scenario":name,
          "Average service access (%)":round(x.service_access.mean(),2),
          "High-priority/risk share (%)":round((s>=.50).mean()*100,2)})
    return pd.DataFrame(out)

st.title("AI-Driven Smart Slum Upgradation Planning Tool")
st.caption("Working prototype based on the methodology in the supplied research paper.")

with st.sidebar:
    upload=st.file_uploader("Upload real CSV (optional)",type="csv")
    k=st.slider("Planning clusters",2,8,5)

if upload:
    d=pd.read_csv(upload)
    missing=[c for c in FEATURES if c not in d.columns]
    if missing:
        st.error("Missing columns: "+", ".join(missing)); st.stop()
    for c in FEATURES: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=FEATURES).copy()
    if "area_id" not in d: d["area_id"]=[f"AREA_{i+1:04d}" for i in range(len(d))]
    if "latitude" not in d: d["latitude"]=30.68
    if "longitude" not in d: d["longitude"]=76.75
    if "priority" not in d:
        s=(.28*d.infrastructure_gap/100+.20*d.environmental_risk/100+
           .18*d.economic_vulnerability/100+.16*d.health_risk/100+
           .12*d.population_density/max(d.population_density.max(),1)+
           .10*(1-d.service_access/100))
        d["priority"]=(s>=s.quantile(.68)).astype(int)
else:
    d=demo_data()

cs=StandardScaler().fit_transform(d[FEATURES])
d["cluster"]=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(cs)+1
results,models,scaler=train(d)

t1,t2,t3,t4,t5=st.tabs(["Dashboard","Model Evaluation","Priority Map","Clusters","Scenarios"])

with t1:
    a,b,c,e=st.columns(4)
    a.metric("Areas analysed",len(d))
    b.metric("High-priority areas",int(d.priority.sum()))
    c.metric("High-risk share",f"{d.priority.mean()*100:.1f}%")
    e.metric("Avg. service access",f"{d.service_access.mean():.1f}%")
    st.plotly_chart(px.histogram(d,x="priority_score",nbins=20,
        title="Area priority score distribution"),use_container_width=True)

with t2:
    st.subheader("Random Forest, Gradient Boosting and SVM")
    st.dataframe(results,use_container_width=True)
    st.info("The paper reports RF 92%, Gradient Boosting 94%, and SVM 89%. "
            "These are paper-reported results; the values above are recalculated "
            "from the data used by this prototype.")
    long=results.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1"])
    st.plotly_chart(px.bar(long,x="Model",y="value",color="variable",
        barmode="group",labels={"value":"Score (%)"}),use_container_width=True)

with t3:
    best=results.sort_values("F1",ascending=False).iloc[0]["Model"]
    m=models[best]
    if best=="SVM": prob=m.predict_proba(scaler.transform(d[FEATURES]))[:,1]
    else: prob=m.predict_proba(d[FEATURES])[:,1]
    md=d.copy(); md["model_priority_probability"]=(prob*100).round(2)
    st.subheader(f"Predicted intervention priority — {best}")
    fig=px.scatter_map(md,lat="latitude",lon="longitude",
        color="model_priority_probability",hover_name="area_id",
        hover_data=["cluster","service_access","infrastructure_gap",
                   "environmental_risk"],zoom=11,height=600,
        title="Priority map")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(md.sort_values("model_priority_probability",ascending=False)
        [["area_id","cluster","model_priority_probability","service_access",
          "infrastructure_gap","environmental_risk","health_risk"]].head(20),
        use_container_width=True)

with t4:
    st.subheader("K-Means neighbourhood clusters")
    st.dataframe(d.groupby("cluster")[FEATURES].mean().round(2),use_container_width=True)
    st.plotly_chart(px.scatter(d,x="service_access",y="infrastructure_gap",
        color=d.cluster.astype(str),size="population_density",
        hover_name="area_id",labels={"color":"Cluster"}),
        use_container_width=True)

with t5:
    st.subheader("Intervention scenario simulation")
    s=scenarios(d)
    st.dataframe(s,use_container_width=True)
    st.plotly_chart(px.bar(s,x="Scenario",y="Average service access (%)"),
                    use_container_width=True)

st.divider()
st.write("Workflow: data collection → preprocessing → RF/Gradient Boosting/SVM "
         "prediction → K-Means clustering → priority ranking → geospatial map → "
         "scenario comparison → update with new data.")
