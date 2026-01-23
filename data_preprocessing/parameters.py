from pathlib import Path
# =====================================================
# ================== Main Parameters ==================
# =====================================================


# ============= Primary configs 
WATERCUT = 0.0  # Water cut percentage
STABILIZING_TIME = 20  # Stabilization time in seconds (time to wait before sampling)
SAMPLINGWINDOW = 60 # SAMPLINGWINDOW in Sconds (time to sample data after stabilization) 
CALIBRATION = True  # Enable/disable calibration mode
# ============= Excel (CSV) File path-
CSV_FILE_PATH = Path(r'/media/MLCanUser/Extreme SSD/pre-test/20251125_test7/WC0_PB_Part1.csv')
# ============= Saving logs directory 
LOG_DIRECTORY = Path(r'/media/MLCanUser/Extreme SSD/pre-test/20251125_test7/Position B') # Directory to save logs
# ============= GUI Configs
GUI_UPDATE_INTERVAL = 0.1  # Update GUI every 100ms


# ============= USER APP (main_user.py) Configs 
USER_LOG_DIRECTORY = Path(r'logs_user')  # Directory to save user logs
PREDICTION_WINDOW = 20 # Number of samples to predict Qo, Qg, Qw in seconds
PREDICTION_CALIBRATION = True  # Enable/disable data calibration before passing to the model
MODEL_PATH = Path(r'model/model_entire.pth')  # Path to the model file
# Prediciton ouput factors
Q_OIL_FACOTR = 1.0
Q_GAS_FACOTR = 1.0
Q_WATER_FACOTR = 1.0

# =====================================================
# ============= Calibration Parameters ================
# =====================================================

# ============= Error Removal Constants
CRL_ERROR_REMOVAL = 5 # Coriolis error removal in T/D
ALICAT_ERROR_REMOVAL = 1 # AliCat error removal in SLPM

# ============= INPUTS Calibration Curves Dictionary (Sensors 1-6)
# Naming here is like this:
# S#(number of sensor)_MIN(I_zero in mA)
# S#(number of sensor)_SPAN(Span in mA)
# S#(number of sensor)_MAX(P_max in PSI)

# SENSOR 5 (B1-I0) (0-5 PSI) 
S5_MIN = 3.998
S5_SPAN = 15.976
S5_MAX_PSI = 5.0

# SENSOR 3 (B1-I1) (0-5 PSI) 
S3_MIN_MA = 4.017
S3_SPAN_MA = 15.988
S3_MAX_PSI = 5.0

# SENSOR 1 (B1-I2) (0-600 PSI)
S1_MIN_MA = 4.00
S1_SPAN_MA = 16.0
S1_MAX_PSI = 100.0

# SENSOR 6 (B2-I0) (0-5 PSI)
S6_MIN_MA = 3.998
S6_SPAN_MA = 15.985
S6_MAX_PSI = 5.0

# SENSOR 4 (B2-I1) (0-5 PSI)
S4_MIN_MA = 4.003
S4_SPAN_MA = 15.997
S4_MAX_PSI = 5.0

# SENSOR 2 (B2-I2) (0-600 PSI)
S2_MIN_MA = 4.000
S2_SPAN_MA = 15.987
S2_MAX_PSI = 600.0

# ============= INPUTS Calibration Curves Dictionary (Coriolis and AliCat)
# Coriolis FM (B1-I3) (0-2000 T/D)
INPUT_CORIOLIS_MIN_SLMP = 0.0 # Minimum SLMP value
INPUT_CORIOLIS_MAX_SLMP = 2000.0 # Maximum SLMP value
INPUT_CORIOLIS_MIN_MA = 4.0 # Minimum mA value
INPUT_CORIOLIS_SPAN_MA = 16.0 # Span in mA value

# AliCat MFM (B2-I3) (0-500 SLPM)
INPUT_ALICAT_MIN_SLMP = 0.0 # Minimum SLMP value
INPUT_ALICAT_MAX_SLMP = 500.0 # Maximum SLMP value
INPUT_ALICAT_MIN_MA = 4.0 # Minimum mA value
INPUT_ALICAT_SPAN_MA = 16.0 # Span in mA value

# ============= OUTPUT Calibration Curves Dictionary (VFD and AliCat)
# AliCat (B3-I2) (0-500 SLPM)
OUTPUT_ALICAT_MIN_SLMP = 0.0 # Minimum SLMP value
OUTPUT_ALICAT_MAX_SLMP = 500.0 # Maximum SLMP value
OUTPUT_ALICAT_MIN_MA = 4.0 # Minimum mA value
OUTPUT_ALICAT_SPAN_MA = 16.0 # Span in mA value

# VFD (B3-I1) (0-60 Hz)
OUTPUT_VFD_MIN_SLMP = 0.0 # Minimum Hz value
OUTPUT_VFD_MAX_SLMP = 60.0 # Maximum Hz value
OUTPUT_VFD_MIN_MA = 4.0 # Minimum mA value
OUTPUT_VFD_SPAN_MA = 16.0 # Span in mA value



# =====================================================
# ================== Secondary Parameters ==================
# =====================================================

# ============= Mapping of channel identifiers to display names and units
CHANNEL_DISPLAY = {
    # Board 1 (Primary measurements)
    1: {
        'I0': {'name': 'Sensor5', 'unit': 'PSI'},
        'I1': {'name': 'Sensor3', 'unit': 'PSI'},
        'I2': {'name': 'Sensor1', 'unit': 'PSI'},
        'I3': {'name': 'CRL', 'unit': 'T/D'}
    },
    # Board 3 (Secondary measurements)
    3: {
        'I0': {'name': 'Sensor6', 'unit': 'PSI'},
        'I1': {'name': 'Sensor4', 'unit': 'PSI'},
        'I2': {'name': 'Sensor2', 'unit': 'PSI'},
        'I3': {'name': 'AliCat', 'unit': 'SLPM'}
    }
}

# ============= Basic ADC Configuration (ADC board configurations)  
BOARD_ADDRESSES = [1, 3]  # Two ADC boards
CHANNELS = ['I0', 'I1', 'I2', 'I3']  # Current channels
BLOCK_SIZE = 8192  # Total readings per block
SAMPLE_RATE = 16 # ADC sample rate setting
LOG_EN = False  # Enable/disable console logging to file
COMMAND_LOG_EN = False  # Enable/disable printing of ADC commands
BAR_WIDTH = 50  # Width of the bargraph in characters
BAR_UPDATE_INTERVAL = 3.0  # Update bargraph every second

# ============= Google Firebase Server Configuration 
# Firebase configuration
DATABASE_URL = 'https://mlcanapp-default-rtdb.firebaseio.com'
WELL_ID = "wellID_1"
FIREBASE_DATABASE_REFERENCE_CURRENT = f'wells/{WELL_ID}/current'
FIREBASE_DATABASE_REFERENCE_UPDATE = f'wells/{WELL_ID}/updates/update_'
FIREBASE_DATABASE_FIELDS = [
    "oilProduction",
    "gasProduction",
    "waterProduction",
]

# Sensor calibration map based on provided calibration certificates
# Real values from above constant variables
SENSOR_CALIBRATION_CURVE = {
    # Mapping of (board_id, channel) to (zero, span, p_max)
    # Format: (I_zero, Span, P_max)
    (1, 'I0'): (S5_MIN, S5_SPAN, S5_MAX_PSI),    #  sensor5 (B1-I0) (0-5 PSI) 
    (1, 'I1'): (S3_MIN_MA, S3_SPAN_MA , S3_MAX_PSI),   #  sensor3 (B1-I1) (0-5 PSI) 
    (1, 'I2'): (S1_MIN_MA, S1_SPAN_MA, S1_MAX_PSI),  #  sensor1 (B1-I2) (0-600 PSI)
    
    (3, 'I0'): (S6_MIN_MA, S6_SPAN_MA, S6_MAX_PSI),    #  sensor6 (B2-I0) (0-5 PSI)
    (3, 'I1'): (S4_MIN_MA, S4_SPAN_MA, S4_MAX_PSI),    #  sensor4 (B2-I1) (0-5 PSI)
    (3, 'I2'): (S2_MIN_MA, S2_SPAN_MA, S2_MAX_PSI),  #  sensor2 (B2-I2) (0-600 PSI)
}

SENSOR_CALIBRATION_LINEAR = {
    (1, 'I3'): (                    # Coriolis FM (B1-I3): 4-20mA maps to 0-2000 T/D
        INPUT_CORIOLIS_MIN_SLMP, 
        INPUT_CORIOLIS_MAX_SLMP, 
        INPUT_CORIOLIS_MIN_MA, 
        INPUT_CORIOLIS_SPAN_MA),     
    (3, 'I3'): (                    # AliCat MFM (B2-I3): 4-20mA maps to 0-500 SLPM
        INPUT_ALICAT_MIN_SLMP, 
        INPUT_ALICAT_MAX_SLMP, 
        INPUT_ALICAT_MIN_MA, 
        INPUT_ALICAT_SPAN_MA),     
}

# Output device calibration (unit_min, unit_max, mA_min, mA_span)
OUTPUT_CALIBRATION = {
    'AliCat': (                     # AliCat (B3-I2): 0-500 SLPM maps to 4-20mA 
        OUTPUT_ALICAT_MIN_SLMP, 
        OUTPUT_ALICAT_MAX_SLMP, 
        OUTPUT_ALICAT_MIN_MA, 
        OUTPUT_ALICAT_SPAN_MA
        ),  
    'VFD': (                        # VFD (B3-I1): 0-60 Hz maps to 4-20mA 
        OUTPUT_VFD_MIN_SLMP, 
        OUTPUT_VFD_MAX_SLMP, 
        OUTPUT_VFD_MIN_MA, 
        OUTPUT_VFD_SPAN_MA
        ),      
}
