import datetime

# https://ris.utwente.nl/ws/files/267876295/Weikmann_2021_Timesencrop_a_million_labeled_sampl.pdf
# Tot. bands used: 9 -> B8 discarded

NEAR_INFRARED_BAND = "B8A"
VISIBLE_RED_BAND = "B4"

BANDS = ['B2', 'B3', VISIBLE_RED_BAND, 'B5', 'B6', 'B7', NEAR_INFRARED_BAND,
        'B11', 'B12']

DATE_COLUMN_NAMES = ["YEAR", "MONTH", "DATE"]

STARTING_ACQUISITION_DATE = datetime.datetime(year=2017, month=9, day=1)
STARTING_TEST_DATE = datetime.datetime(year=2018, month=9, day=1)

NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = 0

POINTS_TO_SAMPLE = 70
FRACTION_KEY_DATES_TRAINING_TIMES = 5