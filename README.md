# Horizontal rapid-mCDR Model to Simulate OAE Perturbation 

## Project Context

This project is part of an internship aimed at gaining a deeper understanding of the physical-biogeochemical mechanisms controlling carbon exchange between the ocean and the atmosphere, in particular through the Ocean Alkalinity Enhancement (OAE) method. OAE is a Marine Carbon Dioxide Removal (mCDR) technique that involves increasing the alkalinity of the ocean to enhance its capacity to sequester atmospheric $CO_2$. This method alters the equilibrium of the ocean carbonate system, promoting the dissolution and long-term storage of carbon dioxide ($CO_2^{aq}$) in the ocean.

The project builds on the reference paper produced by Suselj et al. (2024) presenting Rapid-mCDR, a simplified 1D vertical model that simulates carbon dynamics in the water column after injection of an alkalinity flux. The aim is to improve this model by explicitly integrating horizontal transport (U and V currents) to develop a simplified 3D model, while maintaining computational simplicity and efficiency.

---

## Article Summary

### Rapid-mCDR vertical (1D) model

- Modelizes the vertical dynamics of alkalinity (ALK) and dissolved inorganic carbon (DIC) in the water column by integrating vertical transport processes (advection and diffusion) and using conservation equations solved with the Eulerian formulation,
- Simulates continuous injection of alkalinity flux at the surface, and calculates net flux of $CO_2$ between ocean and atmosphere,
- Compares the results with the global ECCO-Darwin model to validate the relevance of the results.

### Limitations of the initial vertical model

- The 1D model integrates horizontal transport only on average (spatially integrated), which limits realistic modeling of spatial variations in DIC and ALK,
- Need to explicitly incorporate horizontal dynamics (U and V currents) to better represent the dispersion of injected alkalinity and net $CO_2$ flux to the surface.

---

## Project Objectives

- Develop a 2D horizontal model to simulate $CO_2$ dynamics related to alkalinity injection at the surface,
- Explicitly take into account horizontal transport by advection, without horizontal diffusion (dominance of horizontal currents),
- Simulate the following processes:
  - Continuous/daily local injection of alkalinity flux into the initial surface cell,
  - Horizontal transport of alkalinity and dissolved inorganic carbon,
  - Air-sea exchange of $CO_2$ at the surface according to ALK and DIC perturbations.
- Use a Lagrangian method (OceanParcels tool) to simulate surface particle trajectories as a function of velocity fields U and V.

---

## Description of Codes

### DatasLLC270:
- `read-create_LLC270data.ipynb`: First Jupyter notebook reads LLC270 output data from the ECCO Data Portal. You select the date of the required monthly fields—DIC, ALK, SST, SALTanom, and the velocity components uVel_C, vVel_C, and wVel. It also requires the XC, YC, XG, and YG grid files associated with the LLC270 dataset. The notebook then generates NetCDF files on Arakawa C grids for use in OceanParcels.
- `create_forcing_data.ipynb`: Second Jupyter notebook used to create the forcing data by computing and extracting it from the datasets used in the rapid-mCDR vertical model (Suselj et al., 2024).
- `data_fonctions.py`: Python code containing all functions used in the notebooks for creating, formatting, and visualizing data, including the routines that generate NetCDF files and forcing datasets.

### RapidmCDR:
- `LLC270_OceanParcels_rapidmCDR.ipynb`: Final Jupyter notebook used to run the rapid-mCDR 2D horizontal model, combining LLC270 output data from ECCO-Darwin, forcing data derived from the rapid-mCDR vertical model (Suselj et al., 2024), and the Lagrangian particle-tracking method provided by OceanParcels.
  
---

## Dependencies

- Scientific libraries (numpy, scipy, xarray, pandas, matplotlib, cartopy, MITgcmutils, netCDF4, parcels, PyCO2SYS)
- OceanParcels (for Lagrangian particle-tracking simulation): https://docs.parcels-code.org/en/latest/
- ECCO-Darwin data from ECCO Data Portal (for ocean fields and grids):
  - https://data.nas.nasa.gov/ecco/llc_270/ecco_darwin_v5/output/monthly/
  - https://data.nas.nasa.gov/ecco/llc_270/grid/

---

## Main Reference 

Kay Suselj, Dustin Carroll, Daniel Whitt, Bridget Samuels, Dimitris Menemenlis, Hong Zhang, Nate Beatty, Anna Savage (2024)

Quantifying Regional Efficiency of Marine Carbon Dioxide Removal (mCDR) via Alkalinity Enhancement using the ECCO-Darwin Ocean Biogeochemistry State Estimate and an Idealized Vertical 1-D Model.

https://doi.org/10.1029/2024MS004847
