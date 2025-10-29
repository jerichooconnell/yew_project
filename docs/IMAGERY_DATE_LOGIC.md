# Earth Engine Imagery Date Selection Logic

## Overview

The extraction script now intelligently selects imagery dates based on whether a forest plot site is active or inactive.

## Site Classification

### Active Sites
Sites are considered **active** if either:
- `PSP_STATUS == 'A'` (Active status)
- `LAST_MSMT == 'N'` (Not the last measurement, indicating ongoing monitoring)

### Inactive Sites
Sites are considered **inactive** if:
- `PSP_STATUS == 'IA'` (Inactive status)
- `LAST_MSMT == 'Y'` (This was the last measurement)

## Imagery Date Selection Strategy

### For Active Sites
- **Date Range**: `2022-06-01` to `2024-08-31`
- **Rationale**: Use the latest available Sentinel-2 imagery to represent current environmental conditions
- **Growing Season**: June-August (BC growing season)
- **Result**: Provides most up-to-date satellite data for sites that are still being monitored

### For Inactive Sites
The script uses a temporal matching approach:

#### Case 1: Measurement Year ≥ 2015
- **Date Range**: `(year-1)-06-01` to `(year+1)-08-31`
- **Example**: 2018 measurement → 2017-06-01 to 2019-08-31
- **Rationale**: Use imagery temporally close to when the plot was measured

#### Case 2: Measurement Year < 2015
- **Date Range**: `2015-06-01` to `2017-08-31`
- **Rationale**: Sentinel-2 only began in 2015, so use earliest available imagery
- **Note**: These sites may have lower success rates due to limited early imagery

## Output Columns

The extraction results include these new columns:

| Column | Description |
|--------|-------------|
| `measurement_year` | Original field measurement year from `MEAS_YR` |
| `imagery_period` | Date range used for imagery extraction (e.g., "2022-2024") |
| `is_active_site` | Boolean indicating if site is active (True) or inactive (False) |

## Statistics from Dataset

Based on the BC sample data (32,125 plots):
- **Active sites (PSP_STATUS='A')**: 10,398 plots (32.4%)
- **Inactive sites (PSP_STATUS='IA')**: 10,288 plots (32.0%)
- **Sites with LAST_MSMT='N'**: 16,463 plots (51.2%)
- **Sites with LAST_MSMT='Y'**: 15,662 plots (48.8%)

## Benefits of This Approach

1. **Active Sites**: Most relevant and current environmental data for ongoing monitoring
2. **Inactive Sites**: Temporally matched imagery provides historical context
3. **Temporal Consistency**: Each site gets imagery appropriate to its monitoring status
4. **Better Success Rates**: Active sites benefit from recent, higher-quality Sentinel-2 data

## Example Results

From test extraction (10 plots):
```
Plot 4009209: 1990 measurement → Active → Using 2022-2024 imagery ✓
Plot 4045777: 1994 measurement → Inactive → Using 2015-2017 imagery ✗
Plot 4045217: 1996 measurement → Active → Using 2022-2024 imagery ✓
Plot 4042647: 1974 measurement → Active → Using 2022-2024 imagery ✓
```

Success rate: **60%** (6/10 plots successfully extracted)

## Usage

The logic is automatically applied in both extraction modes:
- `extract_point_data()`: Point-based extraction (mean values in 250m buffer)
- `extract_imagery_patch()`: Patch-based extraction (64x64 pixel images)

No configuration needed - the script automatically detects site status from the input CSV columns.

## Future Enhancements

Potential improvements:
1. Add fallback logic for inactive sites with no nearby imagery
2. Use Landsat data for pre-2015 measurements (Landsat goes back to 1984)
3. Add multi-temporal compositing for better cloud removal
4. Include inter-annual variability metrics for active sites

---

**Last Updated**: October 16, 2025  
**Related Scripts**: `scripts/preprocessing/extract_ee_imagery.py`
