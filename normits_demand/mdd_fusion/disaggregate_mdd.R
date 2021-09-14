# Main wrapper functions: -------------------------------------------------

preprocess_mnd <- function(mdd_path, tpp_path, output_path){
  
  "
  Description
  ----------
  - Add column names and tp,tpp to mdd dataset
  - Only use as last resort. i.e. if processed csv goes missing etc.
  
  Parameters
  ----------
  - mdd_path:
      Path to mdd backup 
  
  - tpp_path:
      tpp_path for lookup 
  
  - output_path:
      output path for processed mdd
  
  Return
  ----------
  Save mdd processed dataset in ouput path
  "
  
  tpp_in <- read_csv(tpp_in)
  mdd_in <- data.table::fread(mdd_path)
  
  mdd_in %>% 
    as_tibble() %>% 
    rename_all(~ c('o_zone', 'd_zone', 'm', 't', 'p', 'd', 'demand')) %>% 
    left_join(tpp_in, by = 't') %>% 
    select(o_zone, d_zone, m, p, d, t, tp, tpp, demand) %>% 
    write_csv(output_path)
  
}

segmentator_privacy_masker <- function(mdd_path,
                                       output_path,
                                       segments,
                                       add_privacy,
                                       flatten_wide,
                                       long_segments,
                                       long_unsegment){
  
  "
  Description
  ----------
  - Apply privacy to data by probabilistic infilling (optional)
  - Save in either wide or long format depending on  segments (optional)
  
  Parameters
  ----------
  - mdd_path:
      string - Path to mdd backup 
  
  - output_path:
      string - output path to save segments
    
  - segments:
      Vector - Segments to aggregate privacy with - generally this is done for tpp
    
  - add_privacy:
      Logical - Enable privacy mask
      
  - flatten_wide:
      Logical - Save wide
      
  - long_segments:
      Logical - Save long segments
  
  - long_unsegment:
      Logical - Save privacy masked dataset in long unsegmented format
      
  Return
  ----------
  Segmented or unsegmented with optional privacy masking csv's
  "
  
  sanity_checks(add_privacy, long_unsegment)
  
  mdd_in <- data.table::fread(mdd_path)
  mdd_in <- as_tibble(mdd_in)
  
  mdd_split <- split_segments(mdd_in, segments)
  split_data <- mdd_split[[1]]
  split_keys <- mdd_split[[2]]
  
  if(add_privacy){
    
    print(str_c("Adding Privacy masking via probabilistic infilling"))
    
    split_data <- lapply(split_data, probabilistic_infill, segments, output_path)
    
    print(str_c("Finished privacy masking"))
    
  }
  
  if(flatten_wide){
    
    print(str_c("Beginning to Flatten to wide"))
    
    output_wide <- mapply(long_to_wide, split_data, split_keys, MoreArgs = list(output_path, add_privacy))
    
    print(str_c("Finished Flattening"))
    
  }
  
  if(long_segments){
    
    print(str_c("Saving as segmented long format"))
    
    output_privacy_wide <- mapply(save_long, split_data, split_keys, MoreArgs = list(output_path, add_privacy))
    
    print(str_c("Finshed saving as segmented long format"))
    
  }
  
  if(long_unsegment){
    
    print(str_c("Saving privacy masked dataset in unsegmented od format"))
    
    output_path <- str_c(output_path, "privacy_long_unsegmented/")
    check_path <- dir.create(output_path, showWarnings = FALSE)
    
    output_privacy_long <- split_data %>% 
      bind_rows() %>% 
      write_csv(str_c(output_path, "lot2_od_privacy.csv"))
    
  }
  
}


flexible_filtering_aggregation <- function(mdd_path){
  
  mdd_in <- data.table::fread(mdd_path, nrows = 10000000)
  mdd_in <- as_tibble(mdd_in)
  
}

# Support functions -------------------------------------------------------

split_segments <- function(df, segments){
  
  "
  Description
  ----------
  - Split data into lists by segments
  
  Parameters
  ----------
  - df:
      dataframe - long format od matrix
  
  - segments:
      vector - segments to split data by
      
  Return
  ----------
  - list of dataframe splits
  - vector of splitting keys
  "
  
  split_data <- group_split(df, !!!syms(segments))
  
  split_keys <- df %>% 
    group_by(!!!syms(segments)) %>% 
    group_keys() %>% 
    imap(~ paste0(.y, .x)) %>% 
    as_tibble() %>% 
    unite("temp_name") %>% 
    pull() 
  
  list(split_data, split_keys)
  
}

probabilistic_infill <- function(df, segments, output_path){
  
  "
  Description
  ----------
  - Probabilistically mask data by randomly assinging a probability to each value below 10 and
  if demand is greater than 10 * probability, upscale demand to 10 otherwise 0
  
  Parameters
  ----------
  - df:
      list - list of dataframes which have been split by 'split segments' function
  
  - segments:
      vector - segments to aggregate by
      
  Return
  ----------
  list of dataframes split by segments and privacy masked
  "
  
  df_grouped <- df %>% 
    group_by(o_zone, d_zone, !!!syms(segments)) %>% 
    summarise(demand = sum(demand), .groups = 'drop') %>% 
    filter(demand != 0) %>% 
    ungroup()
  
  probabilities <- round(runif(nrow(df_grouped)),1)
  
  df_grouped %>% 
    mutate(probs = probabilities * 10,
           demand = as.double(demand), 
           demand = if_else(demand < probs, 0,
                           if_else(demand >= 10, demand, 10))) %>% 
    select(-probs)
  
}

long_to_wide <- function(df, key, output_path, add_privacy){
  
  "
  Description
  ----------
  - Flatten long segmented dataframes to wide format
  - Save as csv
  
  Parameters
  ----------
  - df:
      list - list of dataframes which have been split by 'split segments' function
      
  - key:
      vector - vector of keys assigned to each segmented df
  
  - output_path:
      string - output path to save segments
      
  - add privacy:
      logical - from main wrapper function
      
  Return
  ----------
  Save wide formatted segments as csv's
  "
  
  if(add_privacy == TRUE){
    
    output_path <- str_c(output_path, "privacy_wide/")
    
  } else {
    
    output_path <- str_c(output_path, "wide/")
    
  }
  
  check_path <- dir.create(output_path, showWarnings = FALSE)
  
  long_total <- df %>% 
    summarise(demand = sum(demand), .groups = 'drop') %>%
    pull(demand)
  
  df_wide <- df %>% 
    group_by(o_zone, d_zone) %>% 
    summarise(demand = sum(demand), .groups = 'drop') %>% 
    ungroup() %>% 
    complete(o_zone = 1:2771,
             d_zone = 1:2771,
             fill = list(demand = 0)) %>% 
    arrange(o_zone, d_zone) %>%
    spread(d_zone, demand)
  
  wide_total <- df_wide %>% 
    select(-o_zone) %>% 
    as.matrix() %>% 
    sum()
  
  if(wide_total == long_total){
    
    print(str_c(str_split(Sys.time(), " ")[[1]][2], " - ", key, ": Done"))
    write_csv(df_wide, str_c(output_path, key, ".csv"))
    
  } else {
    
    stop("Long and Wide Totals do not match for: ", key)
    
  }
  
}

save_long <- function(df, key, output_path, add_privacy){
  
  "
  Description
  ----------
  - Save as segmented long format
  
  Parameters
  ----------
  - df:
      list - list of dataframes which have been split by 'split segments' function
      
  - key:
      vector - vector of keys assigned to each segmented df
  
  - output_path:
      string - output path to save segments
      
  - add privacy:
      logical - from main wrapper function
      
  Return
  ----------
  Save long formatted segments as csv's
  "
  
  if(add_privacy){
    
    output_path <- str_c(output_path, "privacy_long_segments/")
    
  } else {
    
    output_path <- str_c(output_path, "long_segments/")
    
  }
  
  check_path <- dir.create(output_path, showWarnings = FALSE)
  
  write_csv(df, str_c(output_path, key, ".csv"))
  
}

sanity_checks <- function(add_privacy, long_unsegment){
  
  "
  Description
  ----------
  - Check for contradictions in logicals
  
  Parameters
  ----------
  - add privacy:
      logical - add probabillistic masking?
      
  - long_unsegment:
      logical - save as one long od matrix?
      
  Return
  ----------
  If TRUE then stop function
  "
  
  if(!add_privacy & long_unsegment){
    
    print("If long_unsegment == TRUE then add_privacy == TRUE")
    print("Otherwise change long_unsegment to FALSE")
    
    stop("See statements above")
    
  }
}

