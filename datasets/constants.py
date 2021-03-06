import datetime

NEAR_INFRARED_BAND = "B8"
VISIBLE_RED_BAND = "B4"
BANDS = ['B1', 'B2', 'B3', VISIBLE_RED_BAND, 'B5', 'B6', 'B7', NEAR_INFRARED_BAND,
       'B8A', 'B9', 'B10', 'B11', 'B12']

DATE_COLUMN_NAMES = ["YEAR", "MONTH", "DATE"]

STARTING_ACQUISITION_DATE = datetime.datetime(year=2018, month=1, day=1)

NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = 0

POINTS_TO_SAMPLE = 70
FRACTION_KEY_DATES_TRAINING_TIMES = 5