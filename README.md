
![info](docs/TFN_Landscape_Colour_CMYK.png)

----

# NorMITs Demand

## What is it?
In short, NorMITs Demand is Transport for the North's (TfN) mainland GB
demand tools. These tools started out as Northern specific models, however
they are currently moving towards more flexible zoning to allow them to be applicable to 
elsewhere in mainland GB too.
 

## What does it produce?
Currently, the models are capable of building the following
(see [Demand Breakdown]() for how these outputs are used):
- **Northern Trip End Model** (NoTEM) - MSOA modelled trip ends (base and future year)
  based on the provided Land Use and NTS data. (Link to other repos)
- **Travel Market Synthesiser** (TMS) - Builds the synthetic base year matrices, by distributing
  the trip end data provided by NoTEM.
- **External Forecast System** (EFS) - Future Year forecast matrices, built on top of the 
  Post-Matrix-Estimation matrices provided, the trip ends from NoTEM, and the more accurate
  distributions from TMS.
- **NorMITs Matrix Tools** - A collection of tools for manipulating matrices that underpins much of 
  the work done by TMS and EFS.


## Demand Breakdown
NorMITs Demand is made up of a number of smaller sub-models. The links between them all
can be seen below:


Talk about DVector? Makes it fast!


## Required Data


## Quick Start Guide!


## Documentation
Current code documentation can only be found in the codebase. One of the 
planned improvements include getting this hosted online so it's easily
accessible!

## Planned Improvements