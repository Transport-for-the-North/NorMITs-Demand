# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:27:28 2020

@author: Sneezy
"""

from typing import List, Union, Tuple

import pandas as pd


class SectorReporter:
    """
    Sector reporting class for use in NorMITs Demand framework.
    """
    # defaults
    default_zone_system_name = str
    default_zone_system = pd.DataFrame
    default_sector_grouping = pd.DataFrame
    
    # TODO: TMS Merge: Update NorMITs Synthesiser paths on merge
    def __init__(self,
                 default_zone_system: str = "MSOA",
                 default_zone_file: str = "Y:/NorMITs Synthesiser/Repo/Normits-Utils/zone_groupings/msoa_zones.csv",
                 default_sector_grouping_file: str = "Y:/NorMITs Synthesiser/Repo/Normits-Utils/zone_groupings/lad_msoa_grouping.csv"
                 ):
        """
        Initialises the sector reporting class.
        
        Parameters
        ----------
        default_zone_system:
            The default zoning system title. Unused but necessary for potential
            future improvements or changes.
            Default input is: "MSOA".
            Possible input is any string.
            
        default_zone_file:
            The default zone file which is used to read in the full list of
            model zones. Used for zone checking and calculation of any
            missing sectors.
            Possible input is any file location.
            This file is required to have the column: model_zone_id
            
        default_sector_grouping_file:
            The default zone file which is used to read in the full list of
            sector groups and model zones. Used to determine zone groupings
            for sector total calculation.
            Possible input is any file location.
            This file is required to have the columns: grouping_id, model_zone_id
        """
        self.default_zone_system_name = default_zone_system
        self.default_zone_system = pd.read_csv(default_zone_file)[[
                "model_zone_id"
                ]]
        self.default_sector_grouping = pd.read_csv(default_sector_grouping_file)[[
                "grouping_id",
                "model_zone_id"
                ]]
        
    def calculate_sector_totals(self,
                                calculating_dataframe: pd.DataFrame,
                                grouping_metric_columns: List[str] = None,
                                sector_grouping_file: str = None,
                                zone_col: str = 'model_zone_id',
                                verbose: bool = True
                                ) -> pd.DataFrame:
        """
        Calculates the sector totals off the given parameters.
        
        Parameters
        ----------
        calculating_dataframe:
            This is the dataframe from which the sector totals will be
            calculated. This dataframe needs at least a "model_zone_id"
            column but any additional non-categoric columns can be included.
            No default input.
            Possible input is any Pandas Dataframe with a "model_zone_id"
            column.
            
        grouping_metric_columns:
            The columns to be grouped into sector totals.
            Default input is: None. If this default input is used then
            all columns (except "model_zone_id") are selected.
            Possible input is any list of strings.
            THESE ARE THE COLUMNS WE KEEP

        zone_system_name:
            The name of the zone system for this data set.
            Default input is: None. If this default input is used then
            the zone system name is set to the default for this object.
            Possible input is any file location.
            
        zone_system_file:
            The  zone file which is used to read in the full list of
            model zones. Used for zone checking and calculation of any
            missing sectors.
            Default input is: None. If this default input is used then the
            zone system is set to the default for this boject.
            Possible input is any file location.
            This file is required to have the columns: model_zone_id
            
        sector_grouping_file:
            The default zone file which is used to read in the full list of
            sector groups and model zones. Used to determine zone groupings
            for sector total calculation.
            Default input is: None. If this default input is used then
            the sector grouping is set to the default for this object.
            Possible input is any file location.
            This file is required to have the columns: grouping_id, model_zone_id
            
        Return
        ----------
        sector_totals:
            The returned sector totals in Pandas Dataframe form. This will
            include a summed total of each column in either grouping_metric_columns
            or all the non-model zone ID columns depending on what is input in
            grouping_metric_columns.
            
        Future Improvements
        ----------
        Pass different total values. Preserve, as an example, purposes or modes.
        Include zone checks to ensure everything matches up.
        Include zone conversion from default zone to this zone system.
        Different aggregation / total methods. Average or median are examples.
        """
        calculating_dataframe = calculating_dataframe.copy()
            
        if (sector_grouping_file != None):
            # read in file
            print("Changing sector grouping...")
            sector_grouping = pd.read_csv(sector_grouping_file)
        else:
            print("Not changing sector grouping...")
            sector_grouping = self.default_sector_grouping.copy()

        sector_grouping = sector_grouping.rename(columns={'model_zone_id': zone_col})
            
        if (grouping_metric_columns == None):
            grouping_metric_columns = calculating_dataframe.columns[1:]
            
        groupings = sector_grouping["grouping_id"].unique()
        grouping_dataframe_columns = ["grouping_id"]
        grouping_dataframe_columns.extend(grouping_metric_columns)
        sector_totals = pd.DataFrame(
                columns = grouping_dataframe_columns
                )
        
        for group in groupings:
            zones = sector_grouping[
                sector_grouping["grouping_id"] == group
            ][zone_col].values
            calculating_dataframe_mask = calculating_dataframe[zone_col].isin(zones)
            new_grouping_dataframe = pd.DataFrame({"grouping_id": [group]})
    
            for metric in grouping_metric_columns:
                new_grouping_dataframe[metric] = calculating_dataframe[
                    calculating_dataframe_mask
                ].sum()[metric]
                
            if verbose:
                print(new_grouping_dataframe)
            sector_totals = sector_totals.append(new_grouping_dataframe)
            
        return sector_totals

    def get_parameters(self,
                       zone_system_name: str = None,
                       zone_system_file: str = None,
                       sector_grouping_file: str = None,
                       verbose: bool = False
                       ) -> Tuple[str, str, str]:
        """If non-default parameters are supplied, return the correct paths

        Parameters
        ----------
        zone_system_name : str, optional
            Non-default zone system name, by default None
        zone_system_file : str, optional
            Path to the non-default zone definition file, by default None
        sector_grouping_file : str, optional
            Path to the non-default sector definition file, by default None
        verbose : bool, optional
            If True, prints messages when using non-default values,
            by default False

        Returns
        -------
        Tuple[str, str, str]
            Returns the values for zone_system_name, zone_system_file,
            and sector_grouping_file that should be used
        """
        messages = []

        if zone_system_name is not None:
            # do nothing
            messages.append("Changing zone system name...")
        else:
            messages.append("Not changing zone system name...")
            zone_system_name = self.default_zone_system_name

        if zone_system_file is not None:
            # read in file
            messages.append("Changing zone system...")
            zone_system = pd.read_csv(zone_system_file)
        else:
            messages.append("Not changing zone system...")
            zone_system = self.default_zone_system.copy()

        if sector_grouping_file is not None:
            # read in file
            messages.append("Changing sector grouping...")
            sector_grouping = pd.read_csv(sector_grouping_file)
        else:
            messages.append("Not changing sector grouping...")
            sector_grouping = self.default_sector_grouping.copy()

        if verbose:
            print(*messages, sep="\n")

        return zone_system_name, zone_system, sector_grouping

    def calculate_sector_totals_v2(self,
                                   calculating_dataframe: pd.DataFrame,
                                   metric_columns: List[str] = None,
                                   segment_columns: List[str] = None,
                                   zone_system_name: str = "model",
                                   zone_system_file: str = None,
                                   sector_grouping_file: str = None,
                                   sector_system_name: str = "grouping",
                                   aggregation_method: Union[List[str], str] = "sum",
                                   verbose: bool = False
                                   ) -> pd.DataFrame:
        """Groups a dataframe by the desired sector system.
        Aggregates metric_columns by the desired method and retains any
        column names given as segment_columns.

        Parameters
        ----------
        calculating_dataframe : pd.DataFrame
            The dataframe to apply the sectoring to.
            Requires the column model_zone_id to be present and the zoning
            should match the sector_grouping_file used.
        metric_columns : List[str], optional
            List of columns to aggregate, by default None
        segment_columns : List[str], optional
            List of column names to keep dissaggregated. Any columns not
            specified in metric_columns or segment_columns will be lost,
            by default None
        zone_system_name : str, optional
            Name of the zone system if not using the default system. Will be
            used to as the original zone column name
            e.g. "model" -> "model_zone_id, by default "model"
        zone_system_file : str, optional
            Path to the non-default zone definition file, by default None
        sector_grouping_file : str, optional
            Path to the non-default sector grouping file, by default None
        sector_system_name : str, optional
            Name of the sector system. Will be used as the sector column
            name. e.g. "grouping" -> "grouping_id", by default "grouping"
        aggregation_method : Union[List[str], str], optional
            Aggregation method to use for the metric columns. The same method
            is applied to all matric columns. Can be a list of methods to
            produce a column for each method for each metric or None to keep
            the original zone system but add the sector ids as a new column,
             by default "sum"
        verbose : bool, optional
            If True, will print progress messages, by default False

        Returns
        -------
        pd.DataFrame
            The dataframe with sectoring applied.

        Raises
        ------
        ValueError
            Raises a ValueError is any provided column name is not available.
        """
        # Init
        calculating_dataframe = calculating_dataframe.copy()
        metric_columns = metric_columns or []
        segment_columns = segment_columns or []

        # Validate Inputs
        missing_metrics = list()
        for metric in metric_columns:
            if metric not in calculating_dataframe.columns:
                missing_metrics.append(metric)

        missing_segments = list()
        for segment in segment_columns:
            if segment not in calculating_dataframe.columns:
                missing_segments.append(segment)

        if len(missing_metrics) + len(missing_segments) > 0:
            raise ValueError(
                "Dataframe missing the following:\n"
                "Segments: %s\nMetrics: %s"
                % (str(missing_segments), str(missing_metrics))
            )

        zone_system_name, zone_system, sector_grouping = self.get_parameters(
            zone_system_name,
            zone_system_file,
            sector_grouping_file,
            verbose=verbose
        )

        # Use the sector system name as the column name
        sector_grouping_column = f"{sector_system_name}_id"
        grouping_dataframe_columns = [sector_grouping_column]
        grouping_dataframe_columns.extend(segment_columns)

        zone_id_column = f"{zone_system_name}_zone_id"
        sector_grouping.rename(
            {zone_id_column: "model_zone_id"},
            axis=1,
            inplace=True
        )
        res = calculating_dataframe.merge(
            sector_grouping,
            on="model_zone_id"
        )

        if aggregation_method is None:
            # Return the non-aggregated dataframe
            return res

        # Otherwise use the given aggregation method
        res = res.groupby(grouping_dataframe_columns, as_index=False).agg(
            {col: aggregation_method for col in metric_columns}
        )

        return res

    def aggregate_matrix_sectors(self,
                                 matrix_path: str,
                                 out_path: str = None,
                                 zone_system_name: str = "model",
                                 zone_system_file: str = None,
                                 sector_grouping_file: str = None,
                                 sector_system_name: str = "grouping",
                                 aggregation_method: str = "sum",
                                 verbose: bool = False
                                 ) -> None:
        """Aggregates a wide format matrix file to the given sector system.

        Parameters
        ----------
        matrix_path : str
            Path to a matrix file. Must be wide format and contain the zone
            id system within the sector_grouping_file

        out_path : str, optional
            Path to save the new matrix file. If None, the original file is
            overwritten, by default None

        zone_system_name : str, optional
            Name of the zone system if not using the default system. Will be
            used to as the original zone column name
            e.g. "model" -> "model_zone_id, by default "model"

        zone_system_file : str, optional
            Path to the zone definition file, by default None

        sector_grouping_file : str, optional
            Path to the non-default sector grouping file. Must contain
            model_zone_d and grouping_id columns, by default None

        sector_system_name : str, optional
            Name of the sector system. Will be used as the sector column
            name. e.g. "grouping" -> "grouping_id", by default "grouping"

        aggregation_method : str, optional
            Aggregation method to use for the metric columns. The same method
            is applied to all matric columns, by default "sum"

        verbose : bool, optional
            If True, log messages will be printed, by default False
        """

        if out_path is None:
            out_path = matrix_path

        # Fetch any overridden parameters
        zone_system_name, zone_system, sector_grouping = self.get_parameters(
            zone_system_name,
            zone_system_file,
            sector_grouping_file,
            verbose=verbose
        )

        zone_id_column = f"{zone_system_name}_zone_id"
        sector_grouping.rename(
            {zone_id_column: "model_zone_id"},
            axis=1,
            inplace=True
        )
        sector_grouping.set_index("model_zone_id", inplace=True)

        # Replace with du.safe_read_csv when possible
        df = pd.read_csv(matrix_path, index_col=0)

        # Convert to a stacked matrix
        df = df.stack().reset_index()
        df.columns = ["i", "j", "v"]
        df["i"] = df["i"].astype("int64")
        df["j"] = df["j"].astype("int64")

        # Get the initial matrix total
        pre_agg_total = df["v"].sum()

        # Aggregate Origin Zones
        df = df.set_index("i").join(sector_grouping)
        df.columns = ["j", "v", "i_sec"]
        df = df.groupby(["i_sec", "j"], as_index=False).agg(
            {"v": aggregation_method}
        )

        # Aggregate Destination Zones
        df = df.set_index("j").join(sector_grouping)
        df.columns = ["i_sec", "v", "j_sec"]
        df = df.groupby(["i_sec", "j_sec"]).agg(
            {"v": aggregation_method}
        )

        # Get the post aggregation total
        post_agg_total = df["v"].sum()
        if abs(pre_agg_total - post_agg_total) > 1e6:
            print(f"Warning: Starting total {pre_agg_total} does not "
                  f"equal final total {post_agg_total}")

        # Convert back to wide matrix form for file
        df = df.unstack()
        df.index.name = sector_system_name
        df.columns = df.columns.droplevel()

        # Replace with safe write csv
        df.to_csv(out_path)
