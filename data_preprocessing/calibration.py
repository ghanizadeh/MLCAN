"""
Unified calibration module for processing and scaling raw ADC inputs to appropriate units.
This file consolidates:
1. Calibration data
2. Scaling methods
3. Input and output processors
"""
# Import all calibration parameters (channel displays, calibration values, etc.)
#from parameters import *
from data_preprocessing.parameters import *


# CRL_ERROR_REMOVAL = 5
# ALICAT_ERROR_REMOVAL = 1

# --------------------------------------------------------------------------
# Calibration data section
# --------------------------------------------------------------------------

# # Mapping of channel identifiers to display names and units
# CHANNEL_DISPLAY = {
#     # Board 1 (Primary measurements)
#     1: {
#         'I0': {'name': 'Sensor1', 'unit': 'PSI'},
#         'I1': {'name': 'Sensor2', 'unit': 'PSI'},
#         'I2': {'name': 'Sensor3', 'unit': 'PSI'},
#         'I3': {'name': 'CRL', 'unit': 'T/D'}
#     },
#     # Board 3 (Secondary measurements)
#     3: {
#         'I0': {'name': 'Sensor4', 'unit': 'PSI'},
#         'I1': {'name': 'Sensor5', 'unit': 'PSI'},
#         'I2': {'name': 'Sensor6', 'unit': 'PSI'},
#         'I3': {'name': 'AliCat', 'unit': 'SLPM'}
#     }
# }

# # Sensor calibration values based on provided calibration certificates
# SENSOR_CALIBRATION_CURVE = {
#     # Mapping of (board_id, channel) to (zero, span, p_max)
#     # Format: (I_zero, Span, P_max)
#     (1, 'I0'): (3.998, 15.976, 5.0),    #  sensor5 (0-5 PSI)
#     (1, 'I1'): (4.017, 15.988 , 5.0),   #  sensor3 (0-5 PSI) 
#     (1, 'I2'): (4.001, 15.991, 600.0),  #  sensor1 (0-600 PSI)
    
#     (3, 'I0'): (3.998, 15.985, 5.0),    #  sensor6 (0-5 PSI)
#     (3, 'I1'): (4.003, 15.997, 5.0),    #  sensor4 (0-5 PSI)
#     (3, 'I2'): (4.000, 15.987, 600.0),  #  sensor2 (0-600 PSI)
# }

# SENSOR_CALIBRATION_LINEAR = {
#     (1, 'I3'): (0.0, 2000.0, 4.0, 16.0),     # Coriolis FM (default values)
#     (3, 'I3'): (0.0, 1000.0, 4.0, 16.0),     # AliCat MFM (default values)
# }

# # Output device calibration (unit_min, unit_max, mA_min, mA_span)
# OUTPUT_CALIBRATION = {
#     'AliCat': (0.0, 500.0, 4.0, 16.0),  # 0-500 SLPM maps to 4-20mA
#     'VFD': (0.0, 60.0, 4.0, 16.0),      # 0-60 Hz maps to 4-20mA
# }

# --------------------------------------------------------------------------
# Input and Output Processor Classes
# --------------------------------------------------------------------------

class InputProcessor:
    def __init__(self):
        # Use the sensor calibration data directly
        self.sensor_calibration_curve = SENSOR_CALIBRATION_CURVE
        self.sensor_calibration_linear = SENSOR_CALIBRATION_LINEAR
    
    def scale_input(self, board_id, channel, mA_value, calibration=True):
        """
        Scale a raw input value to the appropriate unit using the simple formula:
        P = ((I - I_zero) / Span) * P_max
        
        Args:
            board_id: ADC board identifier
            channel: Channel identifier (e.g., 'I0')
            mA_value: Raw input in mA
            calibration: Boolean flag to enable/disable calibration
            
        Returns:
            tuple: (scaled_value, unit)
        """
        
        if not calibration:
            return mA_value, "mA"
        
        else:
            
            key = (board_id, channel)
            
            # When no signal coming from the sensor, return -1
            if abs(mA_value) < 3.8:
                return -1, "N/A"
            
            if key not in self.sensor_calibration_curve and key not in self.sensor_calibration_linear:
                # Return raw value if no calibration data is available
                return mA_value, "mA"
            
            if key in self.sensor_calibration_curve:
            
                i_zero, span, p_max = self.sensor_calibration_curve[key]
                
                # Apply the formula: P = ((I - I_zero) / Span) * P_max
                scaled_value = ((mA_value - i_zero) / span) * p_max
                
                # Handle negative values that might occur near zero
                scaled_value = max(0.0, scaled_value)
                
                unit = CHANNEL_DISPLAY[board_id][channel]['unit']
                
                return scaled_value, unit
            
            elif key in self.sensor_calibration_linear:
                
                unit_min, unit_max, mA_min, mA_span = self.sensor_calibration_linear[key]
                unit_range = unit_max - unit_min
                
                scaled_value = ((mA_value - mA_min) / mA_span) * unit_range + unit_min # linear scaling
                
                # Error removal
                if board_id == 1 and channel == 'I3':
                    scaled_value = max(0.0, scaled_value - CRL_ERROR_REMOVAL)
                elif board_id == 3 and channel == 'I3':
                    scaled_value = max(0.0, scaled_value - ALICAT_ERROR_REMOVAL)
                
                unit = CHANNEL_DISPLAY[board_id][channel]['unit']
                
                return scaled_value, unit

class OutputProcessor:
    def __init__(self):
        
        self.output_calibration = OUTPUT_CALIBRATION
    
    def scale_output(self, output_type, unit_value):
        """
        Scale a unit value to the appropriate mA output using the inverse formula:
        mA = mA_min + (unit_value / unit_max) * mA_span
        
        Args:
            output_type: Output device identifier (e.g., 'AliCat', 'VFD')
            unit_value: Value in units (e.g., SLPM, Hz)
            
        Returns:
            float: Corresponding mA value
        """
        if output_type not in self.output_calibration:
            # Return default 4-20mA range mapping if no calibration is available
            if output_type == 'AliCat':
                unit_min, unit_max = 0.0, 500.0
            elif output_type == 'VFD':
                unit_min, unit_max = 0.0, 60.0
            else:
                unit_min, unit_max = 0.0, 100.0
                
            mA_min, mA_span = 4.0, 16.0
        else:
            unit_min, unit_max, mA_min, mA_span = self.output_calibration[output_type]
        
        # Calculate the mA value (clamp unit_value between min and max)
        unit_value = max(unit_min, min(unit_value, unit_max))
        
        unit_range = unit_max - unit_min
        scaled_ma_value = ((unit_value - unit_min) / unit_range) * mA_span + mA_min # linear scaling
        
        return scaled_ma_value
