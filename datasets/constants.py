import datetime

NEAR_INFRARED_BAND = "B8"
VISIBLE_RED_BAND = "B4"
BAVCROPS_BANDS = ['B1', 'B2', 'B3', VISIBLE_RED_BAND, 'B5', 'B6', 'B7', NEAR_INFRARED_BAND,
       'B8A', 'B9', 'B10', 'B11', 'B12']

DENETHOR_BANDS = ['B1', 'B2', 'B3', VISIBLE_RED_BAND, 'B5', 'B6', 'B7', NEAR_INFRARED_BAND,
       'B8A', 'B9', 'B11', 'B12'] # B10 missing

DATE_COLUMN_NAMES = ["YEAR", "MONTH", "DATE"]

STARTING_ACQUISITION_DATE = datetime.datetime(year=2018, month=1, day=1)
STARTING_TEST_DATE = datetime.datetime(year=2019, month=1, day=1)

NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = 0

POINTS_TO_SAMPLE = 70
FRACTION_KEY_DATES_TRAINING_TIMES = 5

DENETHOR_CLASSES = ["Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]
DENETHOR_CROP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]