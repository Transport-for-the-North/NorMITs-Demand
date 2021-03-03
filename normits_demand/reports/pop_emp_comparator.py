# -*- coding: utf-8 -*-
"""
    Module to compare the inputs and outputs of the MSOA population and employment data.
"""

##### IMPORTS #####
# Standard imports
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

# Third party imports
import pandas as pd
import numpy as np
from openpyxl.worksheet.worksheet import Worksheet

# Local imports
import normits_demand as nd
from normits_demand import efs_constants as consts
from normits_demand.utils import general as du, sector_reporter_v2 as sr_v2
from normits_demand.models import efs_production_model as pm
from normits_demand.models import efs_attraction_model as am


##### CONSTANTS #####
EXCEL_MAX = (10000, 1000) # Maximum size of dataframe to be written to excel

##### CLASSES #####
# TODO: Split into two classes inheriting from a common base
class PopEmpComparator:
    """Class to read the input MSOA population (or employment) data and compare it to the output.

    Reads the input values, growth and constraints and compares these to the
    output population (or employment) data at various levels of aggregation.
    Excel file or CSVs are produced with the comparisons for checking.
    """
    ZONE_COL = 'model_zone_id'

    ZONE_COL = "msoa_zone_id"

    def __init__(self,
                 import_home: str,
                 growth_csv: str,
                 constraint_csv: str,
                 output_csv: str,
                 data_type: str,
                 base_year: str,
                 sector_grouping_file: str = None,
                 sector_system_name: str = "tfn_sectors_zone",
                 verbose: bool = True,
                 ):
        """Reads and checks the csv files required for the comparisons.

        Parameters
        ----------
        import_home : str
            Base path for the input location, used for finding the base population
            and employment data.

        growth_csv : str
            Path to the csv containing the input growth values for population (or employment)
            at MSOA level for all output years.

        constraint_csv : str
            Path to the csv containing the input constraints for population (or employment)
            at MSOA level for all output years.

        output_csv : str
            Path to the output population (or employment) data produced by the
            EFSProductionGenerator (or EFSAttractionGenerator) class at MSOA level.

        data_type : {'population', 'employment'}
            Whether 'population' or 'employment' data is given for the comparison.

        base_year : str
            Base year of the model.

        sector_grouping_file : str, optional
            Path to the sector grouping file to use, if None (default) uses the default
            one present in the SectorReporter class.

        sector_system_name : str, optional
            Name of the sector system being used, defaul 'tfn_sectors_zone'.

        Raises
        ------
        ValueError
            If the data_type parameter isn't 'population' or 'employment' or the base_year
            isn't found as a column in the output csv.
        """
        # Init
        self.verbose = verbose

        # Check data_type is allowed
        self.data_type = data_type.lower()
        if self.data_type not in ("population", "employment"):
            raise ValueError(
                'data_type parameter should be "population" or "employment" '
                f'not "{data_type}"'
            )

        du.print_w_toggle("Initialising %s comparisons:"
                          % self.data_type.capitalize(),
                          echo=verbose)

        # Read the output data and extract years columns
        self.output, self.years = self._read_output(output_csv)
        # Check base year is present in output
        self.base_year = base_year
        if str(self.base_year) not in self.years:
            raise ValueError(f'Base year ({self.base_year}) not found in the '
                             f'{self.data_type} output DataFrame.')

        # Read the required columns for input csv files
        self.input_data, self.constraint_data, self.growth_data = self._read_inputs(
            import_home, constraint_csv, growth_csv
        )

        # Create dictionary of sector reporter parameters
        self.sector_params = {
            "sector_grouping_file": str(sector_grouping_file),
            "sector_system_name": str(sector_system_name),
            "zone_system_name": "msoa",
        }

    def _read_print(self,
                    path: Path,
                    group_cols: List[str] = None,
                    **kwargs
                    ) -> pd.DataFrame:
        """Reads given CSV file and outputs time taken.

        Parameters
        ----------
        path : Path
            Path to the CSV file to read.
        kwargs :
            Keyword arguments to pass to `du.safe_read_csv`.

        Returns
        -------
        pd.DataFrame
            The data read from the CSV.
        """
        df = du.safe_read_csv(
            path,
            print_time=self.verbose,
            **kwargs
        )

        if group_cols is not None:
            df = df.groupby(group_cols).sum().reset_index()

        return df

    def _read_output(self, output_csv: str) -> Tuple[pd.DataFrame, List[str]]:
        """Reads the output CSV file and gets the year columns present.

        Parameters
        ----------
        output_csv : str
            Path to the output CSV file.

        Returns
        -------
        pd.DataFrame
            The output data.
        List[str]
            List of the year columns present in the output.
        """
        output = self._read_print(output_csv)
        pat = re.compile(r"\d{4}")
        years = [i for i in output.columns if pat.match(i.strip())]
        return output, years

    def _read_inputs(self,
                     import_home: str,
                     constraint_csv: str,
                     growth_csv: str,
                     base_year: str = '2018',
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Reads the input, constraint and growth CSV files.

        Performs some checks on the inputs and normalises the growth.

        Parameters
        ----------
        import_home : str
            Base path for the input location, used for finding the base population
            and employment data.

        constraint_csv : str
            Path to the constraint CSV file.

        growth_csv : str
            Path to the growth CSV file.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            The input, constraint and growth data from the
            given files.
        """
        # Init
        start = time.perf_counter()

        # BACKLOG: Update pop_emp_comparator path building
        #  labels: EFS, demand merge

        # Big old cludge!
        # Half of these don't matter. We just want the imports!!
        imports, *_ = du.build_efs_io_paths(
            import_location=import_home,
            export_location=import_home,
            model_name=consts.MODEL_NAME,
            iter_name="iter0",
            scenario_name=consts.SC00_NTEM,
            demand_version=nd.ExternalForecastSystem.__version__
        )

        # Read input data
        if self.data_type == "population":
            # Set up args
            base_yr_col = "people"
            index_cols = [self.ZONE_COL, base_yr_col]
            path = os.path.join(imports["land_use"], consts.LU_POP_FNAME % base_year)

            # Read base year data
            du.print_w_toggle(f'\tReading "{path}"', end="", echo=self.verbose)

            input_data = pm.get_pop_data_from_land_use(path, years=[base_year])
            input_data = input_data.reindex(columns=index_cols)
            input_data = input_data.groupby(self.ZONE_COL, as_index=False).sum()

        elif self.data_type == "employment":
            # Set up args
            base_yr_col = "E01"
            index_cols = [self.ZONE_COL, base_yr_col]
            path = os.path.join(imports["land_use"], consts.LU_EMP_FNAME % base_year)

            # Read in base year data
            du.print_w_toggle(f'\tReading "{path}"', end="", echo=self.verbose)
            input_data = am.get_emp_data_from_land_use(path, years=[base_year])
            input_data = input_data.reindex(columns=index_cols)

        else:
            # We shouldn't be able to get here
            raise ValueError("%s is not a valid data_type. We shouldn't be "
                             "able to get here!" % str(self.data_type))

        input_data.rename(columns={base_yr_col: str(self.base_year)}, inplace=True)
        du.print_w_toggle(f" - Done in {time.perf_counter() - start:,.1f}s",
                          echo=self.verbose)

        # Read constraint data for required years
        cols = [self.ZONE_COL, *self.years]
        constraint_data = self._read_print(
            constraint_csv,
            group_cols=[self.ZONE_COL],
            skipinitialspace=True,
            usecols=cols
        )

        # Read the growth data and normalise against the base year
        growth_data = self._read_print(
            growth_csv,
            group_cols=[self.ZONE_COL],
            skipinitialspace=True,
            usecols=cols
        )
        growth_data[self.years] = growth_data[self.years].div(
            growth_data[str(self.base_year)],
            axis=0
        )
        return input_data, constraint_data, growth_data

    def compare_totals(self) -> pd.DataFrame:
        """Compares the input and output column totals and produces a summary of the differences.

        Returns
        -------
        pd.DataFrame
            DataFrame containing differences between the input and output column totals.
        """
        # Calculate totals for each input and output for each year column
        totals = pd.DataFrame({'input total': self.input_data[self.base_year].sum(),
                               'constraint total': self.constraint_data[self.years].sum(),
                               'mean growth input': self.growth_data[self.years].mean(),
                               'output total': self.output[self.years].sum()})
        totals.index.name = 'year'

        # Calculate comparison columns
        con_diff = totals['output total'] - totals['constraint total']
        con_div = (totals['output total'] / totals['constraint total']) - 1
        growth = totals['output total'] / totals.loc[self.base_year, 'output total']
        out_growth = growth - totals['mean growth input']

        # Assign
        totals['constraint difference'] = con_diff
        totals['constraint % difference'] = con_div
        totals['output growth'] = growth
        totals['growth difference'] = out_growth

        return totals

    def _compare_dataframes(self, index_col: str,
                            input_: pd.DataFrame,
                            constraint: pd.DataFrame,
                            growth: pd.DataFrame,
                            output: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenates the given dataframes and calculates comparison columns.

        All dataframes should include a column with the name `self.ZONE_COL`
        which will be set as the index for the concatenation. An additional
        level will be added to the column names to distinguish the source of
        the data.

        Parameters
        ----------
        index_col : str
            Name of the column to be used as the index for concatenation.

        input_ : pd.DataFrame
            Input base year data.


        constraint : pd.DataFrame
            Input constraint data with all output year columns.

        growth : pd.DataFrame
            Input growth data with all output year columns.

        output : pd.DataFrame
            Output data.

        Returns
        -------
        pd.DataFrame
            Concatenation of all given dataframes with additional comparison columns.
        """
        # Init
        nm_dfs = [
            ('input', input_),
            ('constraint', constraint),
            ('growth', growth),
            ('output', output)
        ]

        # Set index of dataframes to index_col for concat and rename columns
        # with source
        concat = list()
        for nm, df in nm_dfs:
            df = df.set_index(index_col)
            df.columns = pd.MultiIndex.from_tuples([(i, nm) for i in df.columns])
            concat.append(df)
        comp = pd.concat(concat, axis=1)
        comp.index.name = index_col

        # Calculate comparison columns
        for yr in self.years:
            con_diff = comp[(yr, 'output')] - comp[(yr, 'constraint')]
            con_div = (comp[(yr, 'output')] / comp[(yr, 'constraint')]) - 1
            growth_calc = comp[(yr, 'output')] / comp[(self.base_year, 'output')]
            out_growth = growth_calc - comp[(yr, 'growth')]

            comp[(yr, 'constraint difference')] = con_diff
            comp[(yr, 'constraint % difference')] = con_div
            comp[(yr, 'output growth')] = growth_calc
            comp[(yr, 'growth difference')] = out_growth

        # Sort columns and return comparison
        return comp.sort_index(axis=1, level=0, sort_remaining=False)

    def compare_msoa_totals(self) -> pd.DataFrame:
        """Compares the input and output values at MSOA level and produces a summary.

        Returns
        -------
        pd.DataFrame
            Differences between the input and output values at MSOA level.
        """
        # Calculate totals for each MSOA on the outputs
        cols = [self.ZONE_COL, *self.years]
        output_msoa = self.output[cols].groupby(self.ZONE_COL, as_index=False).sum()

        # Concatentate dataframes and create comparison columns
        return self._compare_dataframes(self.ZONE_COL, self.input_data, self.constraint_data,
                                        self.growth_data, output_msoa)

    def compare_sector_totals(self) -> pd.DataFrame:
        """Compares the input and output values at sector level and provides a summary.

        Returns
        -------
        pd.DataFrame
            Differences between the input and output values at sector level.
        """
        # Init
        SPLIT_COL = 'overlap_msoa_split_factor'
        sector_rep = sr_v2.SectorReporter()
        cols = [self.ZONE_COL, *self.years]
        msoa_coded_data = {
            'input': self.input_data[[self.ZONE_COL, self.base_year]],
            'constraint': self.constraint_data[cols],
            'output': self.output[cols],
            'growth': self.growth_data[cols]
        }

        # Init Loop
        sector_data = {}
        metric_cols = {}
        loop_gen = zip(msoa_coded_data.items(),
                       [[self.base_year]] + [self.years] * 3)

        # Calculate sectors totals (or means) for the comparison data
        for (nm, df), met_cols in loop_gen:
            # Rename column for sector totals method
            df = df.rename(columns={self.ZONE_COL: "model_zone_id"})
            original_cols = [c for c in df.columns if c != 'model_zone_id']
            df = sector_rep.calculate_sector_totals_v2(
                df, met_cols, **self.sector_params, aggregation_method=None
            )
            df = df.rename(columns={self.sector_params['sector_system_name'] + '_id':
                                    self.sector_params['sector_system_name']})
            sector_data[nm] = df[[self.sector_params['sector_system_name'],
                                  SPLIT_COL,
                                  *original_cols]]
            metric_cols[nm] = met_cols
        del msoa_coded_data

        # Aggregate the sectors together, accounting for the split column
        grouped = {}
        grouping_cols = [self.sector_params['sector_system_name']]
        for (nm, df), agg in zip(sector_data.items(), 3 * ['sum'] + ['mean']):
            if agg == 'sum':
                # Mutliply metric columns by split factor
                for c in metric_cols[nm]:
                    df[c] = df[c] * df[SPLIT_COL]
                df = df.drop(columns=SPLIT_COL)
                grouped[nm] = df.groupby(grouping_cols, as_index=False).sum()
            else:
                # Calculate the weighted average
                weighted_avg = lambda x: pd.Series(
                    np.average(x[metric_cols[nm]], axis=0, weights=x[SPLIT_COL]), metric_cols[nm]
                    )
                df = df.groupby(grouping_cols).apply(weighted_avg).reset_index()
                grouped[nm] = df
        del sector_data

        # Concatentate dataframes and create comparison columns
        sector_comp = self._compare_dataframes(
            self.sector_params['sector_system_name'],
            grouped['input'], grouped['constraint'], grouped['growth'], grouped['output']
        )
        sector_comp.index.name = 'sector'
        return sector_comp

    def write_comparisons(self,
                          output_loc: str,
                          output_as: str = 'excel',
                          year_col: bool = False
                          ) -> None:
        """Runs each comparison method and writes the output to a csv.

        Parameters
        ----------
        output_loc : str
            Path to the output folder.
        output_as : str, optional
            What type of output file(s) should be created options are 'excel' (default)
            or 'csv', if 'excel' is selected any outputs that are too large for a sheet
            will be saved as CSVs instead.
        year_col : bool, optional
            Whether or not to include the year as an index or column header, default is False.
            If True the year will be an index value so there will be multiple rows for each ID,
            if False (default) the year will be an column header so there will be multiple
            columns for each value.
        """
        start = time.perf_counter()
        # Check output type
        output_as = output_as.lower()
        accepted_values = ('excel', 'csv')
        if output_as not in accepted_values:
            raise ValueError(f'output_as should be one of {accepted_values} not "{output_as}"')

        # Create output folder, if it doesn't already exist
        output_dir = Path(output_loc) / f"{self.data_type} comparisons".title()
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f'Producing {self.data_type.capitalize()} comparisons:')
        # Comparison methods to run with tuple containing names for return DataFrames
        comparisons = ((self.compare_totals, ('Totals Comparison',)),
                       (self.compare_msoa_totals, ('MSOA Totals Comparison',)),
                       (self.compare_sector_totals, ('Sector Totals Comparison',)))
        # Run all comparisons and save each dataframe
        outputs = {}
        for func, names in comparisons:
            # Create tuple for dataframes if func only returns one, so it can be looped through
            dataframes = (func(),) if len(names) == 1 else func()
            for nm, df in zip(names, dataframes):
                if year_col and nm.lower() != 'totals comparison':
                    # Move year column to index level and set name to year, keep column order
                    col_order = df.loc[:, df.columns.get_level_values(0)[0]].columns.tolist()
                    df = df.stack(level=0)
                    df.index.names = df.index.names[:-1] + ['year']

                    # Make sure no columns are missing from column order
                    missing = [c for c in df.columns if c not in col_order]
                    df = df[col_order + missing]
                # Write to csv if df too large for excel sheet, or csv output type selected
                if output_as == 'csv' or len(df) > EXCEL_MAX[0] or len(df.columns) > EXCEL_MAX[1]:
                    du.safe_dataframe_to_csv(df, output_dir / f'{nm}.csv', flatten_header=True)
                else: # Save to dict ready to write to spreadsheet
                    outputs[nm] = df

        if output_as == 'excel':
            # Check path can be written to
            out_path = output_dir / f'{self.data_type.capitalize()}_Comparisons.xlsx'
            out_path = du.file_write_check(out_path)

            with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                # Write all the outputs dataframes to a separate sheet in the spreadsheet
                for nm, df in outputs.items():
                    df.to_excel(writer, sheet_name=nm)

                    # Generate list of number formats based on column names
                    # Use different pattern for ratio sheet
                    if 'ratio' in nm.lower():
                        # Pattern which matches anything except within.* or total
                        pat = re.compile(r'.*\'(within.*|total)\'.*', re.IGNORECASE)
                        num_format = ['Percent' if not pat.match(str(c)) else 'Comma [0]'
                                      for c in df.columns]
                    else:
                        pat = re.compile(r'.*(percent|\%|growth).*', re.IGNORECASE)
                        num_format = ['Percent' if pat.match(str(c)) else 'Comma [0]'
                                      for c in df.columns]
                    # Leave formatting alone for index columns and header rows
                    num_format = [None] * len(df.index.names) + num_format
                    # Update column formats
                    _excel_column_format(writer.sheets[nm], num_format, len(df.columns.names))

        print(f'\tSaved in: "{output_dir}"',
              f'\tDone in {time.perf_counter()-start:.0f}s',
              sep='\n')


##### FUNCTIONS #####
def _excel_column_format(sheet: Worksheet, formats: List[str], ignore_rows: int=0):
    """Updates the formats of all columns in given sheet based on list of formats.

    Parameters
    ----------
    sheet : Worksheet
        The sheet to update.
    formats : List[str]
        The names of the styles to use for each column in the sheet, should
        be the same length as the number of columns in the sheet.
    ignore_rows : int
        Number of header rows that shouldn't be formatted, default 0.
    """
    for i, col in enumerate(sheet.iter_cols()):
        # Get format for the current column if non given then move to next column
        try:
            form = formats[i]
        except IndexError:
            break
        if form is None or form.lower() == 'normal':
            continue
        # Update each cells style
        for r, cell in enumerate(col):
            # Ignore cells that are in header rows
            if r < ignore_rows:
                continue
            cell.style = form
