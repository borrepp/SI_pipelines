######################################################################################################################
# Combine Behavioral Data from *.nev (markers) & *.YAML files into a NWB file 
# file format follow YAULAB convention of the task : visual cues + 2 shakers + footbar + eye tracking + pupil diameter
# If Analog and Raw-Neural data is collected, it will be added to the NWBfile
######################################################################################################################
from yaulab_processing import get_filePaths_to_extract_raw, createNWB_raw


######################################################################################################################
# PATH/FILE INFORMATION:
# If "fileName = None" it will search for all the YAML files within the "folderName"
######################################################################################################################
folderName = 'Y:\Data_Albus\Albus_NeuralData\Albus-S1'
fileName = None

######################################################################################################################
# Parent folder where to search for expDAY_log.XLSX files that contain Receptive Field Information
# Set to "None" to not search for this information. 
# If the receptive field information has not been updates, set to NONE and wait until creating preproNWB
# Example path from S1 recordings: 
#       expLog_parentFolder = "Y:\\Data_Albus\\Albus_NeuralData\\Albus-S1_logs_xlsx"
######################################################################################################################
expLog_parentFolder = None

##########################################################################################################
# EXPERIMENTAL INFORMATION NEEDED TO CREATE .nwb.
# This information is not found in .yaml not in .nev/.ns5 files
##########################################################################################################
# DEFAULT FOR ALBUS:
Experiment_Description = "Amplitude categorization of unimanual (cued) stimuli with or without bimanual (distractor) stimuli. Decisions are reported by eye responses. \
    \nAmplitudes above than 28 microns are reported as high-category and matched to Red-visual cue locations.\
    Amplitudes lower than 28 microns are reported as low and matched to Yellow-visual cue locations.\
    Warm temperatures can be delivered through the mechanical probe\
    Stimulation can also be delivered by Intracortical MicroStimulation"

Stimulus_Notes = "The animal has both hands restrined with the palms facing up. Two small probes (~3 mm diamater each) were placed\
    on top of left & right finger tips, respectively. An accelerometer is attached to each probe. Command-voltage and accelerometer signals are recorded.\
    Thermal stimulation can be delivered through the mechanical probe, the thermistor signal is recorded.\
    The animal is looking at a monitor (viewing distance 49 cm) where visual cues are presented. The fixation cue\
    and onset of visual events are recorded using one photodiode signal for each.\
    Intracortical Microstimulation can be delivered by up to 4 electrode-contacts. Each microStimulation period (a.k.a. stim-epoch) consist of up to 3 consecutive trains of pulses"

Experimenters= [ "Chen, Jian", "Lin, Jing", "Rust, Dain", "Ung, Kevin", "Vergara de la Fuente, Jose", "Yau, Jeffrey M"]

KeywordExperiment = ['tactile', 'visual', 'rhesus monkey', 'vibration', 'amplitude categorization',
    'primary somatosensory cortex', 'single neuron', 'bimanual touch', 'ICMS'
  ] 

related_publications = None 

##########################################################################################################
# PARAMETERS TO GET AND PREPROCESS ANALOG SIGNALS FROM RIPPLE (.ns5):
##########################################################################################################
analogEye = {
    'exists_eyeHorizontal': True, # Whether or not Horizontal EyeData signal was recorded (set to False if you don't want to extract EyeData HorizontalPosition)
    'exists_eyePupil': True, # Whether or not Pupilometry EyeData signal was recorded (set to False if you don't want to extract Pupil diamater EyeData)
    'eyeTrackID': None, # extracted from YauLabInfo parameters, otherwise "Left" or "Right"
    'eyeXYComments': None, # extracted from YauLabInfo parameters, otherwise add "Text" comments            
    'pupilDiameterComments': None # extracted from YauLabInfo parameters, otherwise add "Text" comments 
}

analogAccl = {
    'exists': True, # Whether or not Accelerometer signal was recorded (set to False if you don't want to extract ShakerTimeStamps)
    'thresholdHigh_std': 15, # threshold in Standar Deviations (to process accelerometer the signal is Z-scored)
    'thresholdLow_std': 5, # In case High threshold didn't get a signal, it will use this one (units = Standar Deviations, signal is z-scored)
    'showPlot': False,
    'showPlot_lowThreshold': True, # PLOT THOSE STIM WHERE SIGNAL WAS DETECTED USING LOW THRESHOLD PARAMETER
    'showPlot_noDetected': True # PLOT THOSE STIM WHERE NO SIGNAL WAS DETECTED
}

analogFix = {
    'exists': True, # Whether or not Photodiode-Fixation signal was recorded (set to False if it doesn't want to be analyzed)
    'normThreshold': 0.2, # Threshold to detect Fixation ON, fixSignal is normalized between 0 - 1
    'showPlot': False
}

analogVisualEvents = {
    'exists': True, # Whether or not Photodiode-VisualEvents signal was recorded (set to False if it doesn't want to be analyzed)
    'normThreshold': 0.2, # Threshold to detect Fixation ON, fixSignal is normalized between 0 - 1
    'showPlot': False,
    'showWarningPlot': False # PLOT THOSE TRIALS WHERE IT WAS DETECTED MORE EVENTS THAN EXPECTED (possible artifacts in photodiode signal due to screen flash?)
}

analogFeet = {
    'exists': True, # Whether or not FootBar signal was recorded (set to False if you don't want to extract FootBar TimeStamps)
    'chunkSizeSecs':  60, # Seconds to chunk Feet signal to detect events.
    'showPlot': False
}

analogReward = {
    'exists': True, # Whether or not Reward signal was recorded (set to False if it doesn't want to be analyzed)
    'chunkSizeSecs':  60, # Seconds to chunk Reward signal to detect events.
    'showPlot': False
}

analogTemp = {
    'exists': True, # Whether or not Reward signal was recorded (set to False if it doesn't want to be analyzed)
    'thermistorIDs':  ['leftProbeTEMP'], # List of labels from thermistors. Making a list will help for future temperatures? e.g.: ['leftProbeTEMP', rightProbeTEMP]
}

##########################################################################################################
##########################################################################################################
if __name__ == '__main__':

    ##########################################################################################################
    #                                       GET THE YAML(s)-Path(s)
    ##########################################################################################################
    # if "updateFiles = True", it will overwrite the ones already created
    # Otherwise it will compile only the YAML files without an NWB-file
    yamlFiles2convert = get_filePaths_to_extract_raw(parentFolder=folderName, fileName=fileName, updateFiles = False)  

    ##########################################################################################################
    #                                       CREATE NWB
    ##########################################################################################################  
    for filePathYAML_in in yamlFiles2convert:

        createNWB_raw(
            filePathYAML=filePathYAML_in, 
            Stimulus_Notes=Stimulus_Notes, 
            KeywordExperiment=KeywordExperiment, 
            Experimenters=Experimenters, 
            Experiment_Description=Experiment_Description, 
            related_publications=related_publications,
            analogEye=analogEye,
            analogAccl=analogAccl, 
            analogFix=analogFix, 
            analogVisualEvents=analogVisualEvents, 
            analogFeet=analogFeet, 
            analogReward=analogReward,
            analogTemp=analogTemp,
            expLog_parentFolder=expLog_parentFolder,
            TimeZone=None, # extracted from YauLabInfo parameters
            process_INdisk = True, # it will copy the files into the same disk/partition as the NWBenv, It will help with I/O speed vs doing everything from the server
            raw_by_ElectrodeGroup = True, # It will concatenate channels from the same ElectrodeGroup/probe. If False, it will create an "Acquisition" container per channel
            verbose=True
        )
