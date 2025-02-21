# Yaulab_pipelines_templates
YAU-Lab[^1] pipelines to process electrophysiologycal data using SpikeInterface[^2] (raw data is NWB[^3] formatted[^4]):

  a) Create NWB file from ripple & behavioral files
  
  b) Run spikeinterface preprocessing from raw-NWB
  
  c) Run spikeinterface sorting on the preprocessing results
  
  d) Process raw-NWB to export LFPs & re-sampling of behavioral signals into a new prepro-NWB file 
  
  e) Export the curated results from PHY or spikeinterface-SortingAnalyzer into a preproNWB


NOTE: it will make use of the yaulab_processing toolbox

[^1]: https://yaulab.com/
[^2]: https://github.com/SpikeInterface
[^3]: https://www.nwb.org/<br />https://github.com/NeurodataWithoutBorders/pynwb
[^4]: https://github.com/borrepp/createNWB
