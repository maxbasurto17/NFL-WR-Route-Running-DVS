# NFL Wide Receiver Route Running: Dangerous Viable Space (DVS)
## Max Basurto | Suhas Narra

## Abstract
In the NFL, route-running is one of a wide receiver’s (WR) most important skills. While current NFL wide receiver route-running metrics are often rudimentary and lack information about the threat of a given route, our research quantifies route-running ability by combining spatial control and expected threat. By applying Spearman’s soccer pitch control methodology (2017) to NFL tracking data, we created a physics-based, machine-learning-less model to identify WR and defensive back (DB) spatial control over the course of a pass play. To better understand the value of the space WRs control relative to DBs, we weight it by an xEPA model that assigns a threat value to each field location. This product yields the WR's Dangerous Viable Space (DVS) at each point in time. By taking the average and maximum of the wide receiver’s DVS, we can calculate both the receiver’s Average Route Threat Efficiency (aRTE) and Maximum Route Threat Efficiency (mRTE), respectively. By isolating wide receiver route-running ability from target share and throw outcome, aRTE and mRTE offer clearer metrics for evaluating downfield threat, scheme fit, and receiver archetypes.

## Repository Info
**Methodology:**
Contains folders with the following items:

* **Data Cleaning and Prep:** Combines inputs, outputs, and supplementary data into one file. Standardizes field coordinates so that all offenses are oriented in the same direction.
* **Field Control Model:** Contains files for the baseline soccer model, adapted football field control, and a calculation for the reasonable range of passing velocity that's used in the control model.
* **Field Value Model:** Establishes xEPA as our field value metric.
* **Final Metrics Computation:** Combines field control and value into a single pipeline to evaluate wide receivers using the metrics derived from control and value.
* **Reasonable Pass Bounds:** Optimizes a boundary for which a Quarterback would reasonably throw the football to a receiver at any given moment in the receiver's route, controlling the area considered for DVS.
