# Flat airport forecast deployment

Remote layout:

```text
/data2/share/chenxi/PuTS/tianji/forecast/
  vis_forecast.slurm
  forecast_config.json
  mlp_test3.py
  xiahang_forecast_system.py
  airport_visibility_common.py
  logs/
  model/
    airport25_main_to_metar_ft_testsplit_20260610_S2_PhaseB_best_score.pt
    robust_scaler_airport25_main_to_metar_ft_testsplit_20260610_s2_w12_dyn25_nopm.pkl
    dataset_metadata.json
    dataset_build_config.json
    airport_s2_backup_best.pt
    airport_s2_backup_scaler.pkl
    dataset_metadata_s2.json
    dataset_build_config_s2.json
```

The Slurm launcher reads `model_path`, `scaler_path`, and
`dataset_metadata_path` from `forecast_config.json`. To switch from the
fine-tuned model to the S2 backup, only change those three paths in the config.
