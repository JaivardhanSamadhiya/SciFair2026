# 2. From VHRStaph.xlsx
pos_vhr = vhr_df[
    vhr_df["host"] == target_host_norm
].copy()
pos_vhr = pos_vhr[["phage", "host"]]
pos_vhr["label"] = 1