# AI Smart Slum Upgradation Working Model

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`
5. Open the local URL shown by Streamlit.

## Included
- app.py: complete Streamlit implementation
- demo_slum_data.csv: runnable demonstration data
- requirements.txt: Python dependencies
- README.md: setup and research notes

## Model pipeline
The implementation follows the paper's methodology:
demographic/household/infrastructure/environment/service data →
preprocessing/normalization → Random Forest + Gradient Boosting + SVM →
K-Means clustering → multi-criteria priority ranking → spatial visualization →
intervention scenario simulation.

## Important limitation
The paper reports RF=92%, Gradient Boosting=94%, SVM=89%, five clusters,
32% high-risk population, 78% mixed-scenario coverage, 45% risk reduction,
22% cost efficiency and 30% planning-time reduction. The supplied paper does
not include the original training dataset, exact labels, split, or full
hyperparameter configuration. Therefore the included CSV is synthetic
demonstration data and the application does NOT claim to reproduce those
reported results. Replace it with the actual project dataset for a valid
experimental reproduction.

## CSV columns
Required:
population_density, service_access, infrastructure_gap,
environmental_risk, basic_service_distance, economic_vulnerability, health_risk

Recommended:
area_id, latitude, longitude, priority
# AI-Based-Smart-Slum-Upgradation-Planning-Tool
