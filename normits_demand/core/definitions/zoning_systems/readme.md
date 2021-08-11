If you want to define a new zoning system, take the following steps:

1. Create a new folder, at this level, with the name of the zoning system.
   This will be the name used to refer to this zoning system in all areas of
   NorMITs Demand.
   
2. In this folder, only one file needs to be defined:
    * **zones.pbz2** - A compressed pandas dataframe, compressed 
    using `normits_demand.utils.fileops.write_df()`.
    Only one columns is needed: **zone_name**.
    This column should contain every single zone name for this zoning system
    If an example is needed, `normits_demand.utils.fileops.read_df()`
    can be used to decompress any zones files that already exist.
