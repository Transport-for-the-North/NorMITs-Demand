
import os

# Paths
# Export
EXPORT = 'I:/NorMITs Synthesiser/Zone Translation/Export/'
SHP_PATH = 'Y:/Data Strategy/GIS Shapefiles'

# Existing Shapefiles
ZNELUM1 = os.path.join(SHP_PATH, 'NELUM_Zones_150518.shp')
ZNELUM2 = os.path.join(SHP_PATH, 'NELUM zones/NELUM_Zones_2.12_upd.shp')
ZNORMS15 = os.path.join(SHP_PATH, 'Norms zones/TfN_Zones_Combined.shp')
ZNORMS18 = 'Y:/NoRMS/Zoning/Norms/Norms_Zoning-2.11/norms_zoning_freeze_2.11.shp'
ZNTEM = os.path.join(SHP_PATH, 'NTEM/GB_70_region.shp')
ZTFGM = os.path.join(SHP_PATH, 'TfGM_PT')
ZNTMv5 = os.path.join(SHP_PATH, 'NTM/NTM_Polygons_v6_5_region.shp')
ZTFNSECTORS = os.path.join(SHP_PATH, 'Analytical Sectors 38/analytical_sectors_v2_3_region.shp')
ZNOHAMS_NORTH = os.path.join(SHP_PATH, 'NoHAMS North/NoHAM_North_zone.shp')
ZNOHAMS_SOUTH = os.path.join(SHP_PATH, 'NoHAMS South - simplified for smaller file size/Simplified NoHAM South zones.shp')
ZNOHAM = 'Y:/NoHAM/Data/600 Zoning/200 Zone System/North_Zones_v2.10/noham_zones_freeze_2.10.shp'
ZIZ2001 = os.path.join(SHP_PATH, 'Scottish_Intermediate_Zones_2001/SG_IntermediateZone_Bdry_2001.shp')
ZIZ2011 = os.path.join(SHP_PATH, 'Scottish_Intermediate_Zones_2011/SG_IntermediateZone_Bdry_2011.shp')
ZUKMSOAHYBRID = os.path.join(SHP_PATH, 'UK MSOA 2011 IZ 2001 Hybrid/UK MSOA 2011 IZ 2001 Hybrid.shp')
ZMSOA = os.path.join(SHP_PATH, 'UK MSOA and Intermediate Zone Clipped 2011/uk_ew_msoa_s_iz.shp')
ZLSOA = os.path.join(SHP_PATH, 'UK LSOA and Data Zone Clipped 2011/uk_ew_lsoa_s_dz.shp')
ZMERGEDLA =  os.path.join(SHP_PATH, 'Merged_LAD_December_2011_Clipped_GB/Census_Merged_Local_Authority_Districts_December_2011_Generalised_Clipped_Boundaries_in_Great_Britain.shp')
ZLAD = os.path.join(SHP_PATH, 'LAD GB 2017/Local_Authority_Districts_December_2017_Full_Clipped_Boundaries_in_Great_Britain.shp')
ZSTOCKTON = os.path.join(SHP_PATH, 'Stockton_Central_Model_Zones_LSOA_EID/Stockton_Central_Model_Zones_LSOA_EID.shp')
ZPLD = os.path.join(SHP_PATH, 'PLD zones/PLDZones_OSGB.shp')

# Existing Zone Translations
LSOATFNSECTORS = (EXPORT + 'tfn_sector_to_lsoa/tfn_sector_to_lsoa.csv')
LSOANORMS18 = (EXPORT + '/norms_to_lsoa/norms_to_lsoa.csv')
LSOAHYBRIDMSOA =  (EXPORT + '/lsoa_to_hybrid_msoa/hybrid_msoa_to_lsoa.csv')
LSOANORMS15 = (EXPORT + '/old_norms_to_lsoa/old_norms_to_lsoa.csv')
LSOANTEM = (EXPORT + 'ntem_to_lsoa/ntem_to_lsoa.csv')
LSOAMSOA = (EXPORT + 'msoa_to_lsoa/msoa_to_lsoa.csv')
LSOANOHAM = (EXPORT + 'noham_to_lsoa/noham_to_lsoa.csv')
LSOAPLD = (EXPORT + 'pld_to_lsoa/pld_to_lsoa.csv')
LSOALAD = (EXPORT + 'lad_to_lsoa/lad_to_lsoa.csv')
LSOASTOCKTON = (EXPORT + 'stockton_to_lsoa/stockton_to_lsoa.csv')
LSOANELUM2 = (EXPORT + 'nelum_to_lsoa.csv')

# CRS System for osgb
OSGB_CRS = {'proj': 'tmerc',
            'lat_0': 49,
            'lon_0': -2,
            'k': 0.9996012717,
            'x_0': 400000,
            'y_0': -100000,
            'datum': 'OSGB36',
            'units': 'm',
            'no_defs': True}
