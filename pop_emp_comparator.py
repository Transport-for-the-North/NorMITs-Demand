# -*- coding: utf-8 -*-
"""
    Module to compare the inputs and outputs of the MSOA population and employment data.
"""

##### IMPORTS #####
# Standard imports
import re
import time
from pathlib import Path
from typing import List

# Third party imports
import pandas as pd
import numpy as np
from openpyxl.worksheet.worksheet import Worksheet

# Local imports
from demand_utilities import utils as du
from demand_utilities.sector_reporter_v2 import SectorReporter


##### CONSTANTS #####
EXCEL_MAX = (10000, 1000) # Maximum size of dataframe to be written to excel


##### CLASSES #####
class PopEmpComparator:
    """Class to read the input MSOA population (or employment) data and compare it to the output.

    Reads the input values, growth and constraints and compares these to the
    output population (or employment) data at various levels of aggregation.
    Excel file or CSVs are produced with the comparisons for checking.
    """
    ZONE_COL = 'model_zone_id'

    def __init__(self,
                 input_csv: str,
                 growth_csv: str,
                 constraint_csv: str,
                 output_csv: str,
                 data_type: {'population', 'employment'},
                 base_year: str,
                 msoa_lookup_file: str,
                 zone_system_name: str = 'msoa',
                 sector_grouping_file: str = None,
                 sector_system_name: str = 'tfn_sectors_zone',):
        """Reads and checks the csvs required for the comparisons.

        Parameters
        ----------
        input_csv : str
            Path to csv containing the input population (or employment) data for
            the base year at MSOA level.
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
        msoa_lookup_file : str
            Path to the lookup between MSOA model ID and MSOA code.
        zone_system_name : str, optional
            Name of the zone system being used for the msoa_lookup_file, default 'msoa'.
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
        # Function to provide information on reading inputs
        def read_print(csv, **kwargs):
            start = time.perf_counter()
            print(f'\tReading "{csv}"', end='')
            df = du.safe_read_csv(csv, kwargs)
            print(f' - Done in {time.perf_counter() - start:,.1f}s')
            return df

        # Check data_type is allowed and initialise variables for that type
        self.data_type = data_type.lower()
        if self.data_type == 'population':
            base_yr_col  = 'base_year_population'
        elif self.data_type == 'employment':
            base_yr_col = 'base_year_workers'
        else:
            raise ValueError('data_type parameter should be "population" or "employment" '
                             f'not "{data_type}"')
        print(f'Initialising {self.data_type.capitalize()} comparisons:')

        # Read the output data and extract years columns
        self.output = read_print(output_csv)
        pat = re.compile(r'\d{4}')
        self.years = [i for i in self.output.columns if pat.match(i.strip())]
        # Check base year is present in output
        self.base_year = base_year
        if str(self.base_year) not in self.years:
            raise ValueError(f'Base year ({self.base_year}) not found in the '
                             f'{self.data_type} output DataFrame.')

        # Read the required columns for input csvs
        self.input_data = read_print(input_csv, skipinitialspace=True)
        self.input_data.rename(columns={base_yr_col: str(self.base_year)}, inplace=True)
        cols = [self.ZONE_COL, *self.years]
        self.growth_data = read_print(growth_csv, skipinitialspace=True, usecols=cols)
        self.constraint_data = read_print(constraint_csv, skipinitialspace=True, usecols=cols)

        # Normalise the growth data against the base year
        self.growth_data[self.years] = self.growth_data[self.years].div(
            self.growth_data[str(self.base_year)], axis=0)

        # Create dictionary of sector reporter parameters
        self.msoa_lookup_file = msoa_lookup_file
        self.sector_params = {'sector_grouping_file': str(sector_grouping_file),
                              'sector_system_name': str(sector_system_name),
                              'zone_system_name': str(zone_system_name)}

    def compare_totals(self) -> pd.DataFrame:
        """Compares the input and output column totals and produces a summary of the differences.

        Returns
        -------
        pd.DataFrame
            DataFrame containing differences between the input and output column totals.
        """
        # Calculate totals for each input and output for each year column
        totals = pd.DataFrame({'input total': self.input_data.sum(),
                               'constraint total': self.constraint_data.sum(),
                               'mean growth input': self.growth_data.mean(),
                               'output total': self.output.sum()})
        drop_index = [i for i in totals.index if i not in self.years]
        totals = totals.drop(index=drop_index)
        totals.index.name = 'year'

        # Calculate comparison columns
        totals['constraint difference'] = totals['output total'] - totals['constraint total']
        totals['constraint % difference'] = (totals['output total']
                                             / totals['constraint total']) - 1
        totals['output growth'] = (totals['output total']
                                   / totals.loc[self.base_year, 'output total'])
        totals['growth difference'] = totals['output growth'] - totals['mean growth input']
        return totals

    def _compare_dataframes(self, index_col: str, input_: pd.DataFrame,
                            constraint: pd.DataFrame, growth: pd.DataFrame,
                            output: pd.DataFrame) -> pd.DataFrame:
        """Concatanates the given dataframes and calulcates comparison columns.

        All dataframes should include a column with the name `self.ZONE_COL` which
        will be set as the index for the concatenation. An additional level will be
        added to the column names to distinguish the source of the data.

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
        # Set index of dataframes to index_col for concat and rename columns with source
        concat = []
        for nm, df in (('input', input_), ('constraint', constraint), ('growth', growth),
                       ('output', output)):
            df = df.set_index(index_col)
            df.columns = pd.MultiIndex.from_tuples([(i, nm) for i in df.columns])
            concat.append(df)
        comp = pd.concat(concat, axis=1)

        # Calculate comparison columns
        for yr in self.years:
            comp[(yr, 'constraint difference')] = (comp[(yr, 'output')]
                                                   - comp[(yr, 'constraint')])
            comp[(yr, 'constraint % difference')] = (comp[(yr, 'output')]
                                                     / comp[(yr, 'constraint')]) - 1
            comp[(yr, 'output growth')] = (comp[(yr, 'output')]
                                           / comp[(self.base_year, 'output')])
            comp[(yr, 'growth difference')] = (comp[(yr, 'output growth')]
                                               - comp[(yr, 'growth')])
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
        # Convert from MSOA zone id to MSOA code
        msoa_lookup = pd.read_csv(self.msoa_lookup_file,
                                  usecols=['model_zone_code', 'model_zone_id'])
        msoa_coded_data = {}
        for nm, df in {'input': self.input_data,
                       'constraint': self.constraint_data,
                       'output': self.output,
                       'growth': self.growth_data}.items():
            df = msoa_lookup.merge(df, on='model_zone_id', validate='1:m')
            df = df.drop(columns='model_zone_id').rename(
                columns={'model_zone_code': 'model_zone_id'}
            )
            msoa_coded_data[nm] = df

        # Calculate sectors totals (or means) for the comparison data
        SPLIT_COL = 'overlap_msoa_split_factor'
        sector_rep = SectorReporter()
        sector_data = {}
        metric_cols = {}
        for (nm, df), met_cols in zip(msoa_coded_data.items(),
                                      [[self.base_year]] + [self.years] * 3):
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
        # Remove population/employment type columns from output data
        if self.data_type == 'population':
            sector_data['output'].drop(
                columns=['property_type_id', 'traveller_type_id'], inplace=True
                )
        else:
            sector_data['output'].drop(columns=['employment_class'], inplace=True)

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

    def write_comparisons(self, output_loc: str, output_as: str='excel', year_col: bool=False):
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
        return


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
