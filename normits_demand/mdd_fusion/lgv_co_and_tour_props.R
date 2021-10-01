library(readr)
library(tidyr)
library(dplyr)
library(purrr)

user <- Sys.info()[[6]]
repo_dir <- paste0("C:/Users/", user, "/Documents/GitHub/NTS-Processing/")

# Load custom functions
source(paste0(repo_dir, "utils/utils.R"))
source(paste0(repo_dir, "utils/lookups.R"))

nts <- read_csv("C:/Users/Pluto/Documents/NTS_C/classified builds/cb_vehicle_type.csv")
# nts <- read_csv("Y:/NTS/unclassified builds/unclassified builds/ub_vehicle_type.csv")

# Filter for diary sample
nts <- filter(nts, W1 == 1)

# Add in aggregated GOR's
nts <- nts %>% 
  lu_agg_gor_from() %>%
  lu_agg_gor_to()

# select wanted columns
nts <- nts %>% 
  select(agg_gor_from, 
         agg_gor_to, 
         p, 
         TripPurpFrom_B01ID, 
         TripPurpTo_B01ID, 
         start_time, 
         is_north,
         weighted_trips,
         MainMode_B03ID,
         VehType_B01ID,
         TripDisIncSW)

# Define trip direction
nts <- nts %>%
  filter(!(TripPurpFrom_B01ID == 17 & TripPurpTo_B01ID == 23),
         !(TripPurpFrom_B01ID == 23 & TripPurpTo_B01ID == 17),
         !(TripPurpFrom_B01ID == 17 & TripPurpFrom_B01ID == 17)) %>%
  mutate(trip_direction = ifelse(TripPurpFrom_B01ID %in% c(17,23), "FH",
                                 ifelse(TripPurpTo_B01ID %in% c(17,23), "TH", "NHB")))

# Redefine the trip purpose of TH trips as a HB trip
tohometrips <- nts %>%
  filter(TripPurpTo_B01ID %in% c(17,23)) %>%
  filter(!(TripPurpFrom_B01ID %in% c(16,22) & TripPurpTo_B01ID %in% c(17,23))) %>%
  mutate(TripPurpFrom_B01ID = case_when(
    TripPurpFrom_B01ID %in% c(1, 18) ~ 1,
    TripPurpFrom_B01ID %in% c(2, 19) ~ 2,
    TripPurpFrom_B01ID %in% c(3, 20) ~ 3,
    TripPurpFrom_B01ID %in% c(4, 5, 21) ~ 4,
    TripPurpFrom_B01ID %in% c(6, 7, 8) ~ 5,
    TripPurpFrom_B01ID %in% c(9, 11, 12, 13, 22, 16) ~ 6,
    TripPurpFrom_B01ID %in% c(10) ~ 7,
    TripPurpFrom_B01ID %in% c(14) ~ 8,
    TripPurpFrom_B01ID %in% c(15) ~ 8,
  )) %>%
  mutate(p = TripPurpFrom_B01ID)

nts <- nts %>%
  filter(!TripPurpTo_B01ID %in% c(17,23)) %>%
  bind_rows(tohometrips)

# Redefine p to match mnd
nts <- nts %>% 
  filter(!is.na(p)) %>% 
  mutate(p = case_when(
    p == 1 ~ "HBW",
    p %in% 2:8 ~ "HBO",
    p %in% 11:18 ~ "NHB"
  ))

# filter time period
nts <- filter(nts, start_time %in% 1:4)

# filter for van
nts <- nts %>%
  filter(MainMode_B03ID %in% 13:16, VehType_B01ID %in% c(8,9))

# aggregate
tour_props <- nts %>% 
  group_by(agg_gor_from, p, trip_direction, start_time) %>% 
  summarise(trips = sum(weighted_trips)) %>% 
  ungroup()

tour_props_output <- tour_props %>% 
  group_by(agg_gor_from, p) %>% 
  mutate(total_trips = sum(trips),
         prop = trips/total_trips) %>% 
  ungroup()

write_csv(tour_props_output, "Y:/NTS/outputs/van_tour_props.csv")

# TLD ---------------------------------------------------------------------

# Aggregate
tlds <- nts %>%
  group_by(agg_gor_from, p, TripDisIncSW) %>% 
  summarise(trips = sum(weighted_trips)) %>% 
  ungroup()

# bands
tlds <- tlds %>% 
  mutate(bands = cut(TripDisIncSW, c(0,15,25,50,75,100,125,150,200,250,300,350,400,500), 
                     right = FALSE,
                     labels = FALSE,
                     ordered_result = TRUE)) %>% 
  mutate(bands = factor(bands))

# Aggregate by band
tlds <- tlds %>%
  group_by(agg_gor_from, p, bands) %>% 
  summarise(trips = sum(trips)) %>% 
  ungroup()

# Calculate band share
tlds <- tlds %>% 
  group_by(agg_gor_from, p) %>% 
  mutate(total_trips = sum(trips),
         band_share = trips/total_trips) 

write_csv(tlds, "Y:/NTS/outputs/van_tld.csv")

# CO Occupancy by TLD ------------------------------------------------------

### new

van_co <- nts %>% 
  mutate(passenger = ifelse(MainMode_B03ID %in% 15:16, 1, 0),
         driver = ifelse(MainMode_B03ID %in% 13:14, 1, 0)) %>% 
  group_by(p) %>% 
  summarise(passenger = sum(weighted_trips * passenger),
            driver = sum(weighted_trips * driver)) %>% 
  ungroup()

# Calculate car occupancy
van_co <- mutate(van_co, occupancy = 1 + passenger/driver)

###


# aggregate 
van_co <- nts %>% 
  mutate(passenger = ifelse(MainMode_B03ID %in% 15:16, 1, 0),
         driver = ifelse(MainMode_B03ID %in% 13:14, 1, 0)) %>% 
  group_by(agg_gor_from, p, TripDisIncSW) %>% 
  summarise(passenger = weighted_trips * passenger,
            driver = weighted_trips * driver) %>% 
  ungroup()

# bands
van_co <- van_co %>% 
  mutate(bands = cut(TripDisIncSW, c(0,15,25,50,75,100,125,150,200,250,300,350,400,500), 
                     right = FALSE,
                     labels = FALSE,
                     ordered_result = TRUE)) %>% 
  mutate(bands = factor(bands)) 

# aggregate by band
van_co <- van_co %>% 
  group_by(agg_gor_from, p, bands) %>%
  summarise(passenger = sum(passenger),
            driver = sum(driver)) %>% 
  ungroup()

# Calculate car occupancy
van_co <- mutate(van_co, occupancy = 1 + passenger/driver)
  
# Sanity check
van_co %>%
  group_by(agg_gor_from, p) %>%
  summarise(mean(occupancy, na.rm = TRUE))

write_csv(van_co, "Y:/NTS/outputs/van_co.csv")

van_co %>% View()
