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
    
3. If the new zoning system will need to be translated into other zoning
   systems, translation definition(s) need to be added into `_translations`.
   In this folder add a file with the following naming structure
   `{zone1}_to_{zone2}_{weighting}.pbz2`, where:
   * _zone1_ is the name of the new zoning system
   * _zone2_ is the name of the zoning system to translate to. (_zone1_ and
     _zone2_ can be swapped, and will still be found.)
   * _weighting_ is the type of weighting used in the translation. So far
     'correspondence' (for spatial translation), 'population_weight' (for 
     population weighted translation), and 'employment weight' (for 
     employment weighted translations) are supported.
    
   The file should have the following columns:
   * _{zone1}\_zone_id_ - The zone numbers in _zone1_.
   * _{zone2}\_zone_id_ - The zone numbers in _zone2_.
   * _{zone1}\_zone_id_to\_{zone2}\_zone_id_ - The factor (between 0 and 1)
     of _zone1_ zone data that should end up in _zone2_ zone.
   * _{zone2}\_zone_id_to\_{zone1}\_zone_id_ - The factor (between 0 and 1)
     of _zone2_ zone data that should end up in _zone1_ zone.
   
