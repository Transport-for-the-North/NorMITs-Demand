# Main wrapper functions: -------------------------------------------------

preprocess_mnd <- function(mdd_path, tpp_path, output_path){
  
  "Description
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
  Save mdd processed dataset in ouput path"
  
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
                                       segments = c("m", "tpp", "p", "d"),
                                       add_privacy,
                                       flatten_wide){
  
  mdd_in <- data.table::fread(mdd_path, nrows = 50000)
  mdd_in <- as_tibble(mdd_in)
  
  mdd_split <- split_segments(mdd_in, segments)
  split_data <- mdd_split[[1]]
  split_keys <- mdd_split[[2]]
  
  if(add_privacy == TRUE){
    
    # mapply(probabilistic_infill, split_data[1:3], split_keys[1:3], MoreArgs = list(segments, output_path))
    
    print(str_c("Adding Privacy masking via probabilistic infilling"))
    
    split_data <- lapply(split_data[1:3], probabilistic_infill, segments, output_path)
    
    print(str_c("Finished privacy masking"))
    
  }
  
  if(flatten_wide){
    
    print(str_c("Beginning to Flatten to wide"))
    
    output <- mapply(long_to_wide, split_data[1:3], split_keys[1:3], MoreArgs = list(output_path, add_privacy))
    
    print(str_c("Finished Flattening"))
    
  }
  
  if(add_privacy & flatten_wide){
    
    print(str_c("Saving privacy masked long format"))
    
    output <- mapply(save_privacy_long, split_data[1:3], split_keys[1:3], MoreArgs = list(output_path))
    
    print(str_c("Finshed saving privacy masked long format"))
    
  }
  
  if(add_privacy & long_od){
    
    print(str_c("Saving privacy masked dataset in unsegmented od format"))
    
    
    
  }
  
  
  
}

# Support functions -------------------------------------------------------

split_segments <- function(df, segments){
  
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
  
  df_grouped <- df %>% 
    group_by(o_zone, d_zone, !!!syms(segments)) %>% 
    summarise(demand = sum(demand)) %>% 
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
  
  if(add_privacy == TRUE){
    
    output_path <- str_c(output_path, "Flat_test_privacy/")
    
  } else {
    
    output_path <- str_c(output_path, "Flat_test/")
    
  }
  
  check_path <- dir.create(output_path, showWarnings = FALSE)
  
  long_total <- df %>% 
    summarise(demand = sum(demand)) %>%
    pull(demand)
  
  df_wide <- df %>% 
    group_by(o_zone, d_zone) %>% 
    summarise(demand = sum(demand)) %>% 
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

save_privacy_long <- function(df, key, output_path){
  
  output_path <- str_c(output_path, "Long_privacy/")
  check_path <- dir.create(output_path, showWarnings = FALSE)
  
  write_csv(df, str_c(output_path, key, ".csv"))
  
}

