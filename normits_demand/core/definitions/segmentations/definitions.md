# Segment Definitions

Each segmentation should be defined here to allow ease of access when choosing
a segmentation level to use during development

## NoTEM Productions

### lu_pop
Equivalent to the segmentation the Land Use population data is received in.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`       | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand
The segmentation of NoTEM Production model pure_demand output.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `p`            | NTEM Purpose                                          | 1-8    |       |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`       | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand_reporting
A simplified reporting version of pure_demand segmentation.

| Segment Name | Description                                           | Values | Notes                   |
|--------------|-------------------------------------------------------|--------|-------------------------|
| `p`            | NTEM Purpose                                          | 1-8    |                         |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |                         |
| `tfn_agg_at`   | An Aggregated version of TfN Area Type                | 1-3    | 1: 1-3, 2: 4-6, 3: 7-8  |


## NoTEM Attractions


## General

### full_tfntt_tfnat
The segmentation of the production mode-time splits. This is the most detailed segmentation
available - and is usually too much for most operations when combined with a 
full zoning system. As Zones are usually directly linked to an area type, this 
segmentation contains a lot of redundant data; often `full_tfntt` is better used.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760    |       |
| `tfn_at`       | TfN Area Type - Extension of NTEM Area Type           | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


### full_tfntt
The lowest level of segmentation that NoTEM can handle when combined with zoning data.
Similar to `full_tfntt_tfnat`, but with the area types removed to remove redundant 0s.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760    |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


### hb_notem_output
The output segmentation of NoTEM models. Similar to `full_tfntt`, but with the unnecessary
household information removed. This will become the most detailed segmentation available
to other NorMITs Demand models.
Output of NoTEM treats all 6 time periods as total values, and NOT averages.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


