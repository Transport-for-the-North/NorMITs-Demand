# All the functions to run:


# 1) Preprocess MND long dataset ------------------------------------------


preprocess_mnd(mdd_path = "Y:/Mobile Data/LOT 2/Data/MPD Mode-Purpose-TypeOfDay-TimeOfDay/Backup/FullYear_Expanded.csv",
               tpp_path = "Y:/Mobile Data/LOT 2/Data/tpp lookup.csv",
               output_path = "Y:/Mobile Data/LOT 2/Data/MPD Mode-Purpose-TypeOfDay-TimeOfDay/")


# 2) privacy masking segmenting tool --------------------------------------

segmentator_privacy_masker(mdd_path = "C:/Users/Pluto/Documents/MPD/lot2_long_od.csv",
                           output_path = "C:/Users/Pluto/Documents/MPD/",
                           segments = c("m", "tpp", "p", "d"),
                           add_privacy = FALSE,
                           flatten_wide = TRUE,
                           long_segments = TRUE,
                           long_unsegment = FALSE)

# TODO: Add functionality to bind rows and save privacy masked big long format so not in segments!