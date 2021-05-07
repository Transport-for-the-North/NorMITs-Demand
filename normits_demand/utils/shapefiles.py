"""
Shapefile handling utils
"""
import geopandas as gpd


def count_list_shp(shp,
                   id_col=None):
    """
    Go and fetch a shape and return a count and a list of unq values
    """

    shp = gpd.read_file(shp)
    if id_col is None:
        id_col = list(shp)[0]
    shp = shp.loc[:, id_col]

    return len(shp), shp


def hybrid_zone_counts(hybrid_msoa_col,
                       unq_hybrid):
    """
    Count hybrid zones
    """
    # TODO: Test if this works or not
    print('Comparing hybrid MSOA zones to census journey to work')
    unq_zones = hybrid_msoa_col.drop_duplicates()
    hybrid_msoa_residence = unq_hybrid[unq_hybrid.isin(unq_zones)]
    if len(hybrid_msoa_residence) == len(unq_hybrid):
        print('All MSOAs accounted for')
        return True
    else:
        hybrid_msoa_non_residence = unq_hybrid[~unq_hybrid.isin(unq_zones)]
        print(len(hybrid_msoa_non_residence), 'residential MSOAs not accounted for')
        return False