# All the functions to run:


# 1) Preprocess MND long dataset ------------------------------------------


preprocess_mnd(mdd_path = "Y:/Mobile Data/LOT 2/Data/MPD Mode-Purpose-TypeOfDay-TimeOfDay/Backup/FullYear_Expanded.csv",
               tpp_path = "Y:/Mobile Data/LOT 2/Data/tpp lookup.csv",
               output_path = "Y:/Mobile Data/LOT 2/Data/MPD Mode-Purpose-TypeOfDay-TimeOfDay/")



segmentator_privacy_masker(mdd_path = "C:/Users/Pluto/Documents/MPD/lot2_long_od.csv",
                           output_path = "C:/Users/Pluto/Documents/MPD/",
                           add_privacy = TRUE, # TRUE if privacy infill
                           flatten_wide = TRUE) # TRUE if flatten to wide

segmentator_privacy_masker(mdd_path = "Y:/Mobile Data/LOT 2/Data/MPD Mode-Purpose-TypeOfDay-TimeOfDay/lot2_long_od.csv",
                           output_path = "C:/Users/HumzaAli/Documents/MPD/",
                           add_privacy = TRUE,
                           flatten_wide = TRUE)


# TODO: Add functionality to bind rows and save privacy masked big long format so not in segments!

