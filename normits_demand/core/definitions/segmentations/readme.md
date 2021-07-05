If you want to define a new segmentation, take the following steps:

1. Create a new folder, at this level, with the name of the segmentation.
   This will be the name used to refer to this segmentation in all areas of
   NorMITs Demand.
   
2. In this folder, two files need to be defined:
    * **unique_segments.pbz2** - A compressed pandas dataframe, compressed 
    using `normits_demand.utils.fileops.write_df()`.
    The column names are segment names, and each row is a unique combination
    of segment values, such that all rows describe every possible segment 
    when using this segmentation.
    If an example is needed, `normits_demand.utils.fileops.read_df()`
    can be used to decompress any segmentation files that already exist.
    * **naming_order.csv** - A csv file containing a list of segment names,
    with each segment name on its own line. The order of the segment names
    will be used when naming segments within normits_demand.
    
3. If no aggregation or multiplication methods need to be defined,
   you are done, otherwise:
    * **Multiplication** - Open multiply.csv. Add a row to this csv, where:
        * _a_ is this segmentation name.
        * _b_ is the name of the segmentation to multiply this one with.
        * _join_ is a semi-colon delimited list of segment names which _a_
        and _b_ have in common, and would be used in a join if using pandas.
        * _out_ the name of the segmentation that results of multiplying _a_
        and _b_ together, this is often either _a_ or _b_, depending on which
        has more segmentation.
    * **Aggregation** - Open aggregate.csv. Add a row to this csv, where:
        * _in_ is the name of the segmentation to start the aggregation with.
        * _out_ is the name of the segmentation to end the aggregation with.
        * _common_ is a semi-colon delimited list of segment names which _in_
        and _out_ have in common. This is often a list of all segment names that
        are in _out_.