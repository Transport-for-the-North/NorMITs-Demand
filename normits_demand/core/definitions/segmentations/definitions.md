# Segment Definitions

Each segmentation should be defined here to allow ease of access when choosing
a segmentation level to use during development

## NoTEM HB Productions

### lu_pop
Equivalent to the segmentation the Land Use population data as received in.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand
The segmentation of NoTEM Production model pure_demand output.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `p`          | NTEM Purpose                                          | 1-8    |       |
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand_reporting
A simplified reporting version of pure_demand segmentation.

| Segment Name | Description                                           | Values | Notes                   |
|--------------|-------------------------------------------------------|--------|-------------------------|
| `p`          | NTEM Purpose                                          | 1-8    |                         |
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |                         |
| `tfn_agg_at` | An Aggregated version of TfN Area Type                | 1-3    | 1: 1-3, 2: 4-6, 3: 7-8  |


## NoTEM HB Attractions

### lu_emp
Equivalent to the segmentation the Land Use employment data as received in.

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `e_cat`      | TfN Traveller Type - Extension of NTEM Traveller Type | E01, E03-E14  | E07 is split into 4. E07A-E07D. 16 values in total. |
| `soc`        | Skill Level Proxy                                     | 1-3           |      |

### pure_attractions_ecat
Equivalent to the segmentation the attractions trip rates come in at. Similar to pure attractions,
but with e_category too

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `e_cat`      | TfN Traveller Type - Extension of NTEM Traveller Type | E01, E03-E14  | E07 is split into 4. E07A-E07D. 16 values in total. |
| `soc`        | Skill Level Proxy                                     | 1-3           |      |


### pure_attractions
The segmentation of NoTEM Attraction model pure_attractions output.

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `soc`        | Skill Level Proxy                                     | 1-3           |       |


### full_attraction
The segmentation of NoTEM Attraction model after mode split.

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `soc`        | Skill Level Proxy                                     | 1-3           |       |
| `m`          | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6      |       |


## NoTEM NHB Productions

### hb_notem_output_no_tp
The segmentation is similar to output of NoTEM models but without time periods. 

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |


### tfn_at
Equivalent to the segmentation containing only TfN area type (extracted from Land Use Population Data received in)

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8           |       |


### notem_hb_tfnat_p_m_g_soc_ns_ca
The segmentation is similar to hb_notem_output_no_tp with TfN area type. 
This segmentation contains NorMITs output segmentation but with TfN area type. 

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type           | 1-8      |       |
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |


### nhb_trip_rate
Equivalent to the segmentation the non home based trip rate data as received in.

| Segment Name | Description                                               | Values     | Notes |
|--------------|-----------------------------------------------------------|------------|-------|
| `nhb_p`      | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`      | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `p`          | NTEM Purpose                                              | 1-8        |       |
| `m`          | NTEM Mode. Modes 3 and 4 are combined to make mode 3      | 1-3, 5-6   |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |


### notem_hb_tfnat_p_m_g_soc_ns_ca_nhbp_nhbm
The segmentation of NoTEM NHB Production model pure_nhb_demand output. 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `p`            | NTEM Purpose                                              | 1-8        |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3      | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |


### pure_nhb_demand
The segmentation of pure_nhb_demand output 
with home based mode and purpose removed. 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |


### pure_nhb_demand_reporting
A simplified reporting version of pure_nhb_demand segmentation.

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_agg_at`   | An Aggregated version of TfN Area Type                    | 1-3        | 1: 1-3, 2: 4-6, 3: 7-8  |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |


### nhb_tfnat_p_m_tp
Equivalent to the segmentation the non home based time period split data as received in. 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |


### full_nhb_tfnat
The segmentation of fully segmented nhb_demand output. 
 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |


### full_nhb
The segmentation of fully segmented nhb_demand output with TfN area type removed.
 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |




## General

### hb_p_m
Segmentation containing just home based NTEM purposes and modes.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |

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


### nhb_notem_output
The output segmentation of NoTEM models. The segmentation is equivalent to full_nhb 
but with columns 'nhb_p' and 'nhb_m' renamed as 'p' and 'm' respectively.
 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `p`            | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `m`            | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |

