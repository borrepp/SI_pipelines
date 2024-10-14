import spikeinterface.core as si
import spikeinterface.preprocessing as spre
from probeinterface.plotting import plot_probe
from spikeinterface.extractors import read_nwb

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion

from spikeinterface.sorters import run_sorter, read_sorter_folder
import spikeinterface.curation as scur
import spikeinterface.qualitymetrics as sqm
from spikeinterface.exporters import export_to_phy

import matplotlib.pyplot as plt
import SI_tools
import os
import shutil
import numpy
import pickle



####################################################################################################################################################################################
#                                                                    SOME DEFAULT PARAMETERS
####################################################################################################################################################################################
n_cpus = os.cpu_count()
n_jobs = n_cpus - 2

job_kwargs = dict(chunk_duration="1s", n_jobs=n_jobs, progress_bar=True)

# si.set_global_job_kwargs(**job_kwargs)

######################################################################################################
# PARAMETER FOR "LOCAL REFERENCING"
exclude_radius_chans_default = 1 # Number of neighbor channels to exclude because are too close to the reference channel
include_radius_chans_default = 4 # Number of neighbor channels delineates the outer boundary of the annulus whose role is to exclude channels that are too far away

noisy_freq_default = None

######################################################################################################
# GENERAL PARAMETER FOR "PEAKS LOCATIONS"
ms_before_default = 0.6
ms_after_default = 1.5
peak_sign_default = 'both' # (“neg” | “pos” | “both”)
nearest_chans_default = 3 # Number of neighbor channels to search for the same waveform

###########################################################
# DETECT PEAKS: 
peak_detect_threshold_default = 5 #  MAD: Median Amplitude Deviations

######################################################################################################
# MOTION ESTIMATION & INTERPOLATION
# monopolar triangulation + non-rigid + decentralized
motion_rigid_default = False
motion_options_default = {
    'method': 'dredge_ap', # Paninski Lab
    'method_kwargs' : {} 
}
interpolate_options_default = {
    'method': 'kriging', # Kilosort-like
    'border_mode': 'remove_channels' # ('remove_channels' | 'force_extrapolate' | 'force_zeros')
}

sorterName_default = 'kilosort4'



####################################################################################################################################################################################
####################################################################################################################################################################################
#                                                                   MAIN FUNCTIONS
####################################################################################################################################################################################
####################################################################################################################################################################################


####################################################################################################################################################################################
# Helper function to search for unique *NWB files from expDATES within a range of dates
####################################################################################################################################################################################
def get_expDay_in_range(parentRecordingFolder, year_start, month_start, day_start, year_stop, month_stop, day_stop):
    
    filesDate_log = []
    filesData_numList = []
    for _, _, files in os.walk(parentRecordingFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            if nameSplit[1]=='.nwb' and '-noNEV' not in nameSplit[0]:

                fileName = nameSplit[0]
                yearFile = int(fileName[3:7])
                monthFile = int(fileName[8:10])
                dayFile = int(fileName[11:13])

                fileIN = True
                if yearFile>=year_start and yearFile<=year_stop:
                    if yearFile==year_start:
                        if monthFile<month_start:
                            fileIN = False
                        elif monthFile==month_start and dayFile<day_start:
                            fileIN = False
                    elif yearFile==year_stop:
                        if monthFile>month_stop:
                            fileIN = False
                        elif monthFile==month_stop and dayFile>day_stop:
                            fileIN = False
                if fileIN:
                    file_label = '{}-{:02d}-{:02d}'.format(yearFile, monthFile, dayFile)
                    if file_label not in filesDate_log:
                        filesDate_log.append(file_label)
                        filesData_numList.append([yearFile, monthFile, dayFile])
    
    # Force Unique and increasing 
    dateSort = numpy.unique(numpy.array(filesData_numList), axis=0)

    return dateSort



####################################################################################################################################################################################
# RUN PREPROCESSING for all the Electrode/Probes and sessions from expDays within a range of dates
####################################################################################################################################################################################
def run_prepro_expDAY_in_range(parentRecordingFolder, parentPreproFolder,
        year_start, month_start, day_start, 
        year_stop, month_stop, day_stop,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = True
    ):

    dateSort = get_expDay_in_range(parentRecordingFolder, year_start, month_start, day_start, year_stop, month_stop, day_stop)

    for n in range(dateSort.shape[0]):

        sessionYear=dateSort[n, 0]
        sessionMonth=dateSort[n, 1]
        sessionDay=dateSort[n, 2]
        
        run_prepro_expDAY(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay,
            local_radius_chans = local_radius_chans, 
            noisy_freq = noisy_freq, 
            ms_before = ms_before, 
            ms_after = ms_after, 
            peak_sign = peak_sign,
            nearest_chans = nearest_chans, 
            peak_detect_threshold = peak_detect_threshold, 
            do_motion = do_motion,
            motion_rigid = motion_rigid, 
            motion_options = motion_options, 
            interpolate_options = interpolate_options,
            localProcess_NWB = localProcess_NWB,
            rewrite_prepro = rewrite_prepro
        )



####################################################################################################################################################################################
# RUN PREPROCESSING for all the Electrode/Probes (ALL sessions) for a given expDay
####################################################################################################################################################################################
def run_prepro_expDAY(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False, 
        rewrite_prepro = True
    ):

    print('Preprocessing exp{}-{:02d}-{:02d}'.format(sessionYear, sessionMonth, sessionDay))

    electrodeGroups = SI_tools.getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)

    for i in range(len(electrodeGroups)):

        if any([prefix in electrodeGroups[i]['probeInfo']['probeName'] for prefix in SI_tools.supported_probes_manufacturer]):
            
            electrodeGroup_sessions = SI_tools.select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name=electrodeGroups[i]['electrodeName'])

            run_prepro(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay, 
                electrodeGroup_sessions = electrodeGroup_sessions,
                local_radius_chans = local_radius_chans, 
                noisy_freq = noisy_freq, 
                ms_before = ms_before, 
                ms_after = ms_after, 
                peak_sign = peak_sign,
                nearest_chans = nearest_chans, 
                peak_detect_threshold = peak_detect_threshold, 
                do_motion = do_motion,
                motion_rigid = motion_rigid, 
                motion_options = motion_options, 
                interpolate_options = interpolate_options,
                localProcess_NWB = localProcess_NWB,
                rewrite_prepro = rewrite_prepro,
                return_recording = False 
            )

            print('\n\nexp{}-{:02d}-{:02d} ElectrodeGroup : {} was processed ¡¡¡\n\n'.format(sessionYear, sessionMonth, sessionDay, electrodeGroups[i]['electrodeName']))

        else:

            print('\nProbe: {} was not found as a valid Device\nFrom exp{}-{:02d}-{:02d}, ElectrodeGroup : "{}"\nIt will NOT be preprocessing\n\n'.format(
                electrodeGroups[i]['probeInfo']['probeName'], sessionYear, sessionMonth, sessionDay, electrodeGroups[i]['electrodeName']))



####################################################################################################################################################################################
# Function to set up parameters to run Sorter & to create Sorting_analyzer
####################################################################################################################################################################################
def get_sorting_params(sorterName, nChans, step_chan, sampling_frequency, ms_before, ms_after, peak_sign, nearest_chans, sorter_whitening, detect_threshold=None):

    # Validate Nearest channels relative to the number of channels in the recording object
    #"nearest_chans = 0" will perform similar to sorting single channels.
    if nChans==1:
        nearest_chans = 0 
    elif nearest_chans>nChans:
        nearest_chans = nChans

    if nearest_chans < 1:
        radius_um = step_chan/2
        location_method = "center_of_mass"
        unit_location_kwargs = {
            'feature': "ptp" #"ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        }

    else:
        radius_um = step_chan*nearest_chans
        location_method = "monopolar_triangulation"
        unit_location_kwargs = {
            'max_distance_um': step_chan*10,
            'optimizer': 'least_square'
        }

    snippet_T1 = int(numpy.ceil(ms_before * sampling_frequency / 1000.0))
    snippet_T2 = int(numpy.ceil(ms_after * sampling_frequency / 1000.0))
    
    ####################################################################################
    # Sparsity will be the first extension to be computed.
    estimate_sparsity_params = {
        'num_spikes_for_sparsity': 500, # How many spikes per units to compute the sparsity (default: int = 100) 
        'ms_before': ms_before, # Cut out in ms before spike time (default: float = 1.0)
        'ms_after':  ms_after, # Cut out in ms after spike time (default: float'= 2.5)
        'method': "radius", # ("radius" | "best_channels" | "amplitude" | "snr" | "by_property" | "ptp" (default: str = 'radius')
        'peak_sign': peak_sign, # Sign of the template to compute best channels (“neg” | “pos” | “both” (default: str = 'neg') 
        'radius_um': radius_um, # (default: float = 100.0) Radius in um for “radius” method
        'num_channels': nearest_chans, # Used for “best_channels” method (default: int = 5)
    }

    ####################################################################################
    # After sparsity is perfomed, then the rest of Postprocessing extensions can be listed:
    # All extensions.
    # Default parameters are listed as comments
    if nearest_chans>=5:
        include_multi_channel_metrics = True
        metric_names = ['peak_to_valley', 'peak_trough_ratio', 'halfwidth', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks',
                            'velocity_above', 'velocity_below', 'exp_decay', 'spread']
    else:
        include_multi_channel_metrics = False
        metric_names = []

    sorting_analyzer_params = {
            'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 500, 'margin_size': None}, # 'method': 'uniform' | 'all', 'max_spikes_per_unit': 500, 'margin_size': None
            'noise_levels': {},  # method : 'mad' | 'std', str default = 'mad' # it is not fully integrated to Extensions factory, it doesn't have the function "._set_params()" 
            'correlograms': {'window_ms' : 50.0, 'bin_ms': 1.0}, # 'window_ms' : 50 (if 50 ms, the correlations will be computed at lags -25 ms … 25 ms), 'bin_ms' : 1
            'isi_histograms': {'window_ms' : 50.0, 'bin_ms': 1.0}, # 'window_ms' : 50, 'bin_ms' : 1
            'waveforms': {'ms_before': ms_before, 'ms_after': ms_after},
            'principal_components': {'n_components': 5, 'mode': 'by_channel_local'}, # 'n_components': 5, 'mode': 'by_channel_local' | by_channel_global, default: by_channel_local
            'templates': {'operators': ["average"], 'ms_before': ms_before, 'ms_after': ms_after}, # The operators to compute. Can be "average", "std", "median", "percentile" , 'ms_before': 1, 'ms_after': 2
            'template_metrics': {'peak_sign': peak_sign, 
                                'upsampling_factor': 20, # The upsampling factor to upsample the templates, default: 10
                                'sparsity': None, # If None, template metrics are computed on the extremum channel only. If sparsity is given, template metrics are computed on all sparse channels of each unit. Default: None
                                'include_multi_channel_metrics': include_multi_channel_metrics, # Whether to compute multi-channel metrics (At least 10 channels shoulb be capturing the waveforms)
                                'metric_names': metric_names, # ['peak_to_valley', 'peak_trough_ratio', 'halfwidth', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks',
                                                    # 'velocity_above', 'velocity_below', 'exp_decay', 'spread'], # the following multi-channel metrics can be computed (when include_multi_channel_metrics=True)
                                'recovery_window_ms': 0.7, # the window in ms after the peak to compute the recovery_slope
                                'peak_relative_threshold': 0.2, #the relative threshold to detect positive and negative peaks, default: 0.2
                                'peak_width_ms': 0.1, # the width in samples to detect peaks, default: 0.1
                                'depth_direction': "y", # the direction to compute velocity above and below, default: "y" (see notes)
                                'min_channels_for_velocity': 5, #the minimum number of channels above or below to compute velocity, default: 5
                                'min_r2_velocity': 0.5, # the minimum r2 to accept the velocity fit, default: 0.5
                                'exp_peak_function': 'ptp', #the function to use to compute the peak amplitude for the exp decay, default: "ptp"
                                'min_r2_exp_decay': 0.5, # the minimum r2 to accept the exp decay fit, default: 0.5
                                'spread_threshold': 0.2, # the threshold to compute the spread, default: 0.2
                                'spread_smooth_um': step_chan, # the smoothing in um to compute the spread, default: 20
                                'column_range': None,   # the range in um in the horizontal direction to consider channels for velocity, default: None, If None, all channels all channels are considered, 
                                                        # If 0 or 1, only the "column" that includes the max channel is considered
                                                        # If > 1, only channels within range (+/-) um from the max channel horizontal position are used
                                    # Notes
                                    #    -----
                                    #    If any multi-channel metric is in the metric_names or include_multi_channel_metrics is True, sparsity must be None,
                                    #    so that one metric value will be computed per unit.
                                    #    For multi-channel metrics, 3D channel locations are not supported. By default, the depth direction is "y".    
                                },
            'template_similarity': {'method': 'cosine', 'max_lag_ms': 0.0}, # 'method': “cosine” | “l2” | “l1”, 'max_lag_ms': 0.0
            'amplitude_scalings': {'ms_before': ms_before, 'ms_after': ms_after, 'handle_collisions': True, 'delta_collision_ms': 2}, # 'handle_collisions': True, delta_collision_ms: 2
            'spike_amplitudes': {'peak_sign': peak_sign}, # ( “neg” | “pos” | “both”, default: str = 'neg') 
            'spike_locations': {
                'ms_before': ms_before, # ms_before : 0.5,
                'ms_after': ms_after, # ms_after : 0.5, 
                'method': location_method, # 'method': "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: 'center_of_mass'
                'spike_retriver_kwargs': {
                        'channel_from_template': False, # For each spike is the maximum channel computed from template or re estimated at every spikes. float, default: 50
                        'radius_um': radius_um, # In case channel_from_template=False, this is the radius to get the true peak. bool, default = True
                        'peak_sign': peak_sign, # In case channel_from_template=False, this is the peak sign. ( “neg” | “pos” | “both”, default: str = 'neg')
                    },
                },  
            'unit_locations': {
                'method': location_method,  #'method': "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: 'center_of_mass'
                'radius_um': radius_um, 
                **unit_location_kwargs
                } 
        }
    
    if 'both' in peak_sign:
        detect_sign = 0
    elif 'neg' in peak_sign:
        detect_sign = -1
    elif 'pos' in peak_sign:
        detect_sign = 1
        
    nt =  snippet_T1 + snippet_T2

    if nearest_chans>1:
        scheme2_detect_channel_radius = radius_um/2
    else:
        scheme2_detect_channel_radius = radius_um

    if sorterName=='mountainsort5':
        if detect_threshold is None:
            detect_threshold = 5.5 # ZCA-Whitening traces
        sorter_label = 'MS5ch{}th{}'.format(nearest_chans, detect_threshold).replace('.', '_')
        sorter_info = dict(
                sorter_name = 'mountainsort5',
                sorter_label = sorter_label,
                useDocker = False, 
                params2update = {
                        'scheme': '2', # Not searching for templates in chunks
                        'detect_threshold': detect_threshold, # ZCA-Whitening traces
                        'detect_sign': detect_sign,
                        'snippet_T1': snippet_T1,
                        'snippet_T2': snippet_T2,
                        'snippet_mask_radius': radius_um,
                        'scheme1_detect_channel_radius': radius_um,
                        'scheme2_phase1_detect_channel_radius': radius_um,
                        'scheme2_detect_channel_radius': scheme2_detect_channel_radius,
                        'scheme2_training_recording_sampling_mode': 'initial',
                        'filter': False,
                        'whiten': sorter_whitening,
                        **job_kwargs,
                },
                estimate_sparsity_params = estimate_sparsity_params,
                sorting_analyzer_params = sorting_analyzer_params
        )
    elif sorterName =='kilosort4':
        if detect_threshold is None:
            detect_threshold = 3  # Whitened Threshold (STD from Absolute traces)

        sorter_label = 'KS4ch{}th{}'.format(nearest_chans, detect_threshold).replace('.', '_')
        sorter_info= dict(
            sorter_name = 'kilosort4', 
            sorter_label = sorter_label,
            useDocker = False,
            params2update = {
                    'Th_universal': 8,
                    'Th_learned': 9,
                    'Th_single_ch': detect_threshold, # Whitened Threshold (STD from Absolute traces)
                    'nt': nt,
                    'nt0min': snippet_T1,
                    'dminx': 1,
                    'nearest_chans': nearest_chans+1,
                    'max_channel_distance': radius_um,
                    'do_CAR': False,
                    'do_correction': False,
                    'skip_kilosort_preprocessing': sorter_whitening==False,
                    'whitening_range': nearest_chans+1,
                    'keep_good_only': True, 
                    'save_extra_vars': True,
                    'torch_device': 'cuda'
            },
            estimate_sparsity_params = estimate_sparsity_params,
            sorting_analyzer_params = sorting_analyzer_params
        )
    
    else:
        raise Exception('Setting parameters to run sorter: "{}" is not supported by this pipeline version\nSupported sorters are:\n\t{}\n\t{}'.format(sorterName, 'mountainsort5', 'kilosort4'))
    
    return sorter_info



####################################################################################################################################################################################
#                                                              MAIN FUNCTION TO RUN PREPRO:
####################################################################################################################################################################################
#   1) Concatenate & Attach Probe (save Plots)
#   2) Filtering
#   3) Detect Bad Channels
#   4) CommonMedianReference (CMR)
#   5) PowerDensitySpectrum (PSD) (save Plots)
#   6) Peaks Locations | Amplitudes (save plots)
#
# If the number of channels is lower than "nearest_chans" it will NOT run MOTION steps
#
#   7) Motion Estimation (save Plots)
#   8) Motion Interpolation (save Plots: new PeaksLocations)
####################################################################################################################################################################################
def run_prepro(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_sessions=None, electrodeGroup_Name=None, session_index=None,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = True,
        return_recording = False     
    ):    
    
    if electrodeGroup_sessions is None:
        electrodeGroups = SI_tools.getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)
        electrodeGroup_sessions = SI_tools.select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=session_index)

    #######################################################################################################################
    # Get/Create FOLDER paths to save Figures
    #######################################################################################################################
    # Create General Session Folder to save Figures
    sessionFolder = os.path.join(os.path.abspath(parentPreproFolder), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
    if not os.path.isdir(sessionFolder):
        os.makedirs(sessionFolder)
    
    #############################################################################
    # Create Folder to save ElectrodeGroup-Session Preprocessing
    elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName'])
    elecGroupSessFolder = os.path.join(sessionFolder, elecGroupSessName)
    if not os.path.isdir(elecGroupSessFolder):
        os.mkdir(elecGroupSessFolder)

    # Check & clear SI temporal folder (it will be used to save peaks, peaks locations, and NWB files if localProcess_NWB=True)
    folder_temp = os.environ.get('SI_PROCESSOR_TEMPDIR')
    if folder_temp is None:
        SI_tools.set_SItempdir_environ()
        folder_temp = os.environ.get('SI_PROCESSOR_TEMPDIR')
    SI_tools.clear_SItempdir()
    

    #############################################################################################
    # Check if preprocessed Recording exists
    if os.path.isfile(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording.pkl')) and not rewrite_prepro:  

        print('\nPreprocessed recording was found... \nelectrodeGroup Session : {}\n'.format(elecGroupSessName))

        si_recording = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording.pkl'))

        step_chan = si_recording.get_annotation('y_contacts_distance_um')

        ###########################################################
        #   DETECT PEAKS options: 
        peaks_options = {
            'method': "locally_exclusive",
            'peak_sign': peak_sign, 
            'detect_threshold': peak_detect_threshold, #  MAD: Median Amplitude Deviations
            'radius_um': step_chan*nearest_chans,
        }

        ###########################################################
        #   LOCALIZE PEAKS options:
        locations_options = {     
            'ms_before': ms_before,
            'ms_after': ms_after,
            'location_method': 'monopolar_triangulation', # Paninski Lab
            'location_kwargs': {'max_distance_um': step_chan*(nearest_chans + 1), 'optimizer': 'least_square'}
        }
    
    else:

        #######################################################################################################################
        # Copy files if need it & get updated FILE paths to read
        #######################################################################################################################
        # Check if the NWB files exists in the same Disk as this "code". Otherwise copy NWB-files to the same disk for "speed"
        copyFiles = False
        if localProcess_NWB:
            for f in electrodeGroup_sessions['fileNamePaths']:
                filePath, _ = os.path.split(os.path.abspath(f))
                if filePath[0:2].lower()!=os.path.abspath(__file__)[0:2].lower():
                    copyFiles = True
        
        # If need it, copy NWBfiles. Update paths to read files
        for f in electrodeGroup_sessions['fileNamePaths']:
            fPath, fName = os.path.split(os.path.abspath(f))
            if copyFiles:
                print('..... copying file {}'.format(fName))
                shutil.copy2(os.path.join(fPath, fName), os.path.join(folder_temp, fName))
                electrodeGroup_sessions['fileNamePaths_read'].append(os.path.join(folder_temp, fName))
            else:
                electrodeGroup_sessions['fileNamePaths_read'].append(os.path.join(fPath, fName))

        ###########################################################
        # Save ElectrodeGroupInformation
        if not copyFiles:
            pickle.dump(electrodeGroup_sessions, open(os.path.join(elecGroupSessFolder, elecGroupSessName + '_electrodeGroupInfo.pkl'), 'wb' ))

        print('\nPreprocessing electrodeGroup Session : {}\n'.format(elecGroupSessName))

        ###################################################
        # LOAD session(s)
        ###################################################
        if len(electrodeGroup_sessions['fileNamePaths_read'])>1:

            si_recording_List = []
            for f in electrodeGroup_sessions['fileNamePaths_read']:

                print('Concatenating sessions :\n', f, '\n')
                
                si_recording_List.append(read_nwb(
                    file_path=f, 
                    electrical_series_path='acquisition/' + electrodeGroup_sessions['electrodePath'],
                    load_recording=True,
                    load_sorting=False,
                ))
                            
            si_recording_raw = si.concatenate_recordings(si_recording_List)

        else:

            f = electrodeGroup_sessions['fileNamePaths_read'][0]

            print('loading session : ', f)

            si_recording_raw = read_nwb(
                    file_path=f, 
                    electrical_series_path='acquisition/' + electrodeGroup_sessions['electrodePath'],
                    load_recording=True,
                    load_sorting=False,
            )
        
        if not si_recording_raw.has_probe():

            print('Attaching probe to the recording ......')
            probe_from_nwb = SI_tools.constructProbe_2d(si_recordingObj_nwb = si_recording_raw, probeInfo_dict = electrodeGroup_sessions['probeInfo'])

            si_recording_rawProbe = si_recording_raw.set_probe(probe_from_nwb)
        else:
            si_recording_rawProbe = si_recording_raw
        
        del si_recording_raw

        ###################################################
        # Order channels by location
        ###################################################
        print('Sorting channels by depth ......')

        chan_ordered_index, chan_orig_index = si.order_channels_by_depth(recording=si_recording_rawProbe, dimensions=("x", "y"))
        si_recording_ordered = si_recording_rawProbe.channel_slice(channel_ids=[si_recording_rawProbe.channel_ids[i] for i in chan_ordered_index])
        si_recording_ordered.set_property(key='to_raw_index', values = chan_orig_index)

        del si_recording_rawProbe

        # CONFIRM LOCATIONS ARE IN "um"
        if si_recording_ordered.has_probe():
            units_probe = numpy.unique(si_recording_ordered._properties['contact_vector']['si_units'])
            if len(units_probe)>1:
                raise Exception('more than one type of units were found in the location of the contacts within the probe : {}', units_probe)
            else:
                si_recording_ordered.annotate(location_units=units_probe[0])

        # Check if the recording Object has the location_units defined
        if 'location_units' in si_recording_ordered.get_annotation_keys():
            if si_recording_ordered.get_annotation('location_units') == 'mm':
                si_recording_ordered._properties['location'] *=1000
                si_recording_ordered._annotations['location_units'] = 'um'

            
            if si_recording_ordered.get_annotation('location_units') != 'mm' and si_recording_ordered.get_annotation('location_units') != 'um':
                raise Exception('Units of the Locations in the RecordingObject is not recognized: {}'.format(
                    si_recording_ordered.get_annotation('location_units') 
                    ))
        else:
            # If not Assume units in 'mm' (default units from NWB format)
            si_recording_ordered._properties['location'] *=1000
            si_recording_ordered.annotate(location_units='um')

        #################################################################################################################
        # GET CHANNEL-CONTACT SPACING ("y" coordinate)
        contact_locations = si_recording_ordered.get_property('location')
        if len(contact_locations)>1:
            step_chan = numpy.mean(numpy.absolute(numpy.diff(contact_locations[:, 1])))
        else:
            step_chan = 50 # Default from SipkeInterface

        si_recording_ordered.annotate(y_contacts_distance_um = step_chan)

        #################################################################################################################
        # ADD CONCATENATION INFORMATION
        nSessions = len(electrodeGroup_sessions['fileNames'])
        nConcatenations = nSessions-1
        if nConcatenations>0:
            concatenationSamples = numpy.array(electrodeGroup_sessions['fileSamples'][:-1]).cumsum()
            concatenationTimes = concatenationSamples/si_recording_ordered.sampling_frequency
        else:
            concatenationSamples = numpy.array([])
            concatenationTimes = numpy.array([])
        
        si_recording_ordered.annotate(nSessions = nSessions)
        si_recording_ordered.annotate(sessionSamples = numpy.array(electrodeGroup_sessions['fileSamples']))
        si_recording_ordered.annotate(nConcatenations = nConcatenations)
        si_recording_ordered.annotate(concatenationSamples = concatenationSamples)
        si_recording_ordered.annotate(concatenationTimes = concatenationTimes)
        si_recording_ordered.annotate(savePath = elecGroupSessFolder)
        si_recording_ordered.annotate(fileNamePaths = electrodeGroup_sessions['fileNamePaths'])

        #################################################################################################################
        #                                            PREPROCESSING
        #################################################################################################################

        #########################################################################################################
        # If there is noise at a specefic frequency, remove it with a notch filter
        if noisy_freq is not None:
            si_recording_denoise = spre.notch_filter(si_recording_ordered, freq=noisy_freq, q=10)
        else:
            si_recording_denoise = si_recording_ordered
        
        del si_recording_ordered
        
        ##########################################################
        # High-pass filter
        si_recording_filter = spre.bandpass_filter(recording=si_recording_denoise, freq_min=500) 

        ##########################################################
        # Median zero-center signal (per channel basis)
        si_recording_center_pre = spre.center(recording=si_recording_filter, mode='median')

        ##########################################################
        # Apply Common Median Reference
        if si_recording_denoise.get_num_channels()>1:
            si_recording_cmr_pre = spre.common_reference(recording=si_recording_center_pre, reference='local', operator='median', local_radius=(local_radius_chans[0] * step_chan, local_radius_chans[1] * step_chan ))
        else:
            si_recording_cmr_pre = si_recording_center_pre

        ##########################################################
        # Plot session's concatenation
        print('\nPlotting {} concatenations from electrodeGroup: {}\n'.format(si_recording_denoise.get_annotation('nConcatenations'), elecGroupSessName))
        SI_tools.plot_concatenations(si_recording_dict={'Raw': si_recording_denoise, 'Filter': si_recording_filter, 'CMR': si_recording_cmr_pre}, 
                                    plot_windows_secs=0.005, sampleChans=True, showPlots=False, savePlots=True, folderPlots=elecGroupSessFolder)

        del si_recording_filter, si_recording_cmr_pre

        #############################################################################################################
        #                       Detect bad channels
        #############################################################################################################

        # Use recObj centered (before CMR)
        print('\nDetecting Bad Channels ....... \n')
        bad_channel_ids, channel_labels = spre.detect_bad_channels(recording=si_recording_center_pre, method="coherence+psd")

        # add channel_labels:
        # 'coeherence+psd' : good/dead/noise/out 
        # 'std', 'mad', 'neighborhood_r2' : good/noise
        si_recording_center_pre.set_property(key='channel_labels', values=channel_labels)

        print('Channel labels : \n\t', channel_labels)
        print('\n{} bad channels detected\n\tBad channel Index: {}'.format(len(bad_channel_ids), bad_channel_ids))

        ##########################################################
        # Remove bad channels 
        si_recording_center_cleanChans = si_recording_center_pre.remove_channels(remove_channel_ids=bad_channel_ids)

        del si_recording_center_pre

        ######################################################################################################
        # PLOT THE PROBE & channel labels
        print('\nPlotting PROBE from electrodeGroup: {}\n'.format(elecGroupSessName))

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))
        ax[0].set_rasterized(True)
        plot_probe(probe=si_recording_denoise.get_probe(), ax = ax[0], with_contact_id=False, with_device_index=True)

        y_probe = si_recording_denoise.get_channel_locations(axes='y').flatten()
        ch_indx = si_recording_denoise.ids_to_indices()
        contact_ids = si_recording_denoise.get_channel_ids()
        for ch in ch_indx:
            ax[1].plot(0, y_probe[ch], marker="s", markeredgecolor=(0, 0, 1, 1), markerfacecolor=(0, 0, 1, 0.5), markeredgewidth = 1, markersize=6)
            ax[1].text(-0.5, y_probe[ch], 'ch={}'.format(contact_ids[ch]), horizontalalignment='left', color=(0, 0, 0, 1), fontsize=8)
            if channel_labels[ch]=='good':
                colorT = (0, 0, 1, 1)
                fweight = 'normal'
            else:
                colorT = (1, 0, 0, 1)
                fweight = 'bold'
            ax[1].text(0.5, y_probe[ch], channel_labels[ch], horizontalalignment='right', color=colorT, fontsize=8, fontweight=fweight)
        ax[1].set_xlim(-1, 1)
        ax[1].set_rasterized(True)
        fig.savefig(os.path.join(elecGroupSessFolder, elecGroupSessName + '_probe.eps'), dpi='figure', format='eps')
        plt.close(fig=fig)

        del bad_channel_ids, channel_labels, si_recording_denoise

        #####################################################################
        # PLOT Power Spectrum Density 
        print('\nPlotting PowerDensitySpectrums from electrodeGroup: {}\n'.format(elecGroupSessName))
        SI_tools.plotPSD_randomChunks(si_recording_center_cleanChans, compare_CMR=True, plot_by_channel=True, chan_radius=local_radius_chans, showPlots=False, savePlots=True, folderPlots=elecGroupSessFolder)

        #############################################################################################################
        #  Reference after removing bad channels
        #############################################################################################################
        if si_recording_center_cleanChans.get_num_channels()>1:
            si_recording = spre.common_reference(recording=si_recording_center_cleanChans, reference='local', operator='median', local_radius=(local_radius_chans[0] * step_chan, local_radius_chans[1] * step_chan ))
        else:
            si_recording = si_recording_center_cleanChans

        ###################################################
        # Add some annotations:
        si_recording.annotate(is_filtered=True)
        si_recording.annotate(is_centered=True)
        si_recording.annotate(centered_mode = 'median')
        si_recording.annotate(is_referenced=True)    
        si_recording.annotate(reference = 'local')
        si_recording.annotate(reference_mode = 'median')

        del si_recording_center_cleanChans

        ####################################################################################################################################
        #  If NWB files were not move from original location: Save Recording Object (LAZY without traces)
        if not copyFiles:
            si_recording.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording'))


        ########################################################################################################################
        #                                     PEAKS LOCATIONS AS A FUNCTION OF TIME
        ########################################################################################################################

        ###########################################################
        #   DETECT PEAKS options: 
        peaks_options = {
            'method': "locally_exclusive",
            'peak_sign': peak_sign, 
            'detect_threshold': peak_detect_threshold, #  MAD: Median Amplitude Deviations
            'radius_um': step_chan*nearest_chans,
        }

        ###########################################################
        #   LOCALIZE PEAKS options:
        locations_options = {     
            'ms_before': ms_before,
            'ms_after': ms_after,
            'location_method': 'monopolar_triangulation', # Paninski Lab
            'location_kwargs': {'max_distance_um': step_chan*(nearest_chans + 1), 'optimizer': 'least_square'}
        }

        ###########################################################
        # PLOT PEAKS LOCATIONS AS A FUNCTION OF TIME
        print('\nPlotting Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

        SI_tools.plot_peakLocations(
            si_recording = si_recording, 
            folderPeaks = folder_temp, 
            extra_recording_label = '', 
            peaks_options = peaks_options, 
            locations_options = locations_options, 
            rewrite = False, 
            locationsSubSampled = True,
            savePlots = True, 
            showPlots = False,
            folderPlots = elecGroupSessFolder
        )

    ########################################################################################################################
    # NOTE:
    # IF CHANNEL COUNT IS LOWER THAN "nearest_chans" IT WILL NOT RUN MOTION CORRECTION
    if si_recording.get_num_channels()>=nearest_chans and do_motion:

        ########################################################################################################################
        #                              MOTION ESTIMATION & INTERPOLATION
        ########################################################################################################################

        #####################################################################
        # Get peaks labels:
        if "by_channel" in peaks_options['method']:
            peaks_prefix = 'byCh'
        elif "locally_exclusive" in peaks_options['method']:
            peaks_prefix = 'loc'
        else:
            raise Exception('Peaks Detection method "{}" not recognized\nAvailable options: {}'.format(peaks_options['method'], ["by_channel", "locally_exclusive", "locally_exclusive_cl", 
                                                                                                                                 "by_channel_torch", "locally_exclusive_torch", "matched_filtering"]))

        peaks_label = '{}{}{}'.format(peaks_prefix, peaks_options['peak_sign'][0].upper(), peaks_options['detect_threshold']).replace('.', '_')

        #####################################################################
        # Get location labels:
        if locations_options['location_method']=='center_of_mass':
            loc_label = 'mass'
        elif locations_options['location_method']=='monopolar_triangulation':
            loc_label = 'mono'
        elif locations_options['location_method']=='grid_convolution':
            loc_label = 'grid'
        else:
            raise Exception('Peaks Location method "{}" not recognized\nAvailable options: {}'.format(peaks_options['method'], ['center_of_mass', 'monopolar_triangulation', 'grid_convolution']))
        
        peaksLoc_label = peaks_label + '_' + loc_label

        #####################################################################
        # Get motion labels:
        if motion_rigid:
            rigid_label = 'rigid'
            win_step_um = step_chan/2
        else:
            rigid_label = 'noRigid'
            win_step_um = step_chan            
        
        if motion_options['method']=='decentralized':
            motion_label = 'deCENTRAL' + rigid_label
        elif motion_options['method']=='iterative_template':
            motion_label = 'iterTEMP' + rigid_label
        elif motion_options['method']=='dredge_ap':
            motion_label = 'dredgeAP' + rigid_label
        elif motion_options['method']=='dredge_lfp':
            motion_label = 'dredgeLFP' + rigid_label
        else:
            raise Exception('Motion Estimation method "{}" not recognized\nAvailable options: {}'.format(motion_options['method'], ['decentralized', 'iterative_template', 'dredge_ap', 'dredge_lfp']))

        
        #####################################################################
        # Get interpolation labels:
        if interpolate_options['method']=='idw':
            interpolation_label = 'idw'
        elif interpolate_options['method']=='nearest':
            interpolation_label = 'near'
        elif interpolate_options['method']=='kriging':
            interpolation_label = 'krig'
        else:
            raise Exception('Interpolation method "{}" not recognized\nAvailable options: {}'.format(interpolate_options['method'], ['kriging', 'idw', 'nearest']))
        
        if interpolate_options['border_mode']=="remove_channels":
            border_label = 'Rmv'
        elif interpolate_options['border_mode']== "force_extrapolate":
            border_label = 'Extrap'
        elif interpolate_options['border_mode']=="force_zeros":
            border_label = 'Zeros'
        else:
            raise Exception('BorderMode method "{}" not recognized\nAvailable options: {}'.format(interpolate_options['border_mode'], ['remove_channels', 'force_extrapolate', 'force_zeros']))
        
        motion_interpolation_label = motion_label + '_' + interpolation_label + border_label

        si_recording_sufix = '_' + peaksLoc_label + '_' + motion_interpolation_label

        if os.path.isfile(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording' + '.pkl')) and not rewrite_prepro:

            print('\nMotion corrected recording was found... \nelectrodeGroup Session : {}\nPeaks {}: sign = {}, detectionTH = {}\nLocation Method: {}\nMotion Interpolation label : {}\n'.format(elecGroupSessName,
                                                                                    peaks_options['method'], peaks_options['peak_sign'], peaks_options['detect_threshold'], locations_options['location_method'], motion_interpolation_label))

            si_recording_motion = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording' + '.pkl'))

        else:

            ##################################################################
            #  Load PEAKS 
            if not os.path.exists(os.path.join(folder_temp, 'peaks_' + peaks_label + '.npy')):

                print('\nGetting Peaks from electrodeGroup: {}\n'.format(elecGroupSessName))

                noise_levels = si.get_noise_levels(si_recording, return_scaled=False)
                
                peaks = detect_peaks(recording=si_recording, 
                    noise_levels = noise_levels,
                    method=peaks_options['method'], 
                    gather_mode = 'memory', # gather_mode= 'npy', # 'npy'
                    folder = None, # folder = eg_dirs['motion'],
                    names = None, # names = ['peaks'],
                    peak_sign=peaks_options['peak_sign'], 
                    detect_threshold=peaks_options['detect_threshold'], 
                    radius_um=peaks_options['radius_um'],
                    **job_kwargs)
                        
                numpy.save(os.path.join(folder_temp, 'peaks_' + peaks_label + '.npy'), peaks)

                del peaks, noise_levels

            peaks = numpy.load(os.path.join(folder_temp, 'peaks_' + peaks_label + '.npy'))

            ##################################################################
            # Load all PEAKS LOCATIONS 
            if not os.path.exists(os.path.join(folder_temp, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy')):

                print('\nGetting Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

                peaks_locations = localize_peaks(
                    recording=si_recording,
                    peaks=peaks,
                    ms_before= locations_options['ms_before'],
                    ms_after= locations_options['ms_after'],
                    radius_um = peaks_options['radius_um'],
                    method= locations_options['location_method'],
                    **locations_options['location_kwargs'],
                    **job_kwargs
                    )

                numpy.save(os.path.join(folder_temp, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy'), peaks_locations)

                del peaks_locations   

            peaks_locations = numpy.load(os.path.join(folder_temp, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy'))

            ###############################################################################
            # MOTION ESTIMATION
            print('\nEstimating MOTION from electrodeGroup: {}\n'.format(elecGroupSessName))

            motion, extra_check = estimate_motion(
                # Parameters common to all motion Methods
                recording=si_recording,
                peaks=peaks,
                peak_locations=peaks_locations,
                direction="y", # **histogram section**
                rigid=motion_rigid, # **non-rigid section** # default False
                win_shape="gaussian", # default "gaussian"
                win_step_um = win_step_um, # default 50.0
                win_scale_um = win_step_um*3.0, # default 150.0
                win_margin_um = None, # default -win_scale_um/2
                method = motion_options['method'], # **method options**
                extra_outputs=True, # **extra options** 
                progress_bar=True, 
                verbose=False,

                # Parameters common to all motion Methods but defined as **method_kwargs
                bin_um = step_chan/4, # default 10.0
                bin_s = 10.0, # default 10.0 # bin_s=10.0, # default 10.0

                # Specific Method arguments
                **motion_options['method_kwargs'],
            )

            contact_locations = si_recording.get_property('location')
            minY_loc = -50
            maxY_loc = max(contact_locations[:, 1])+50

            SI_tools.plot_motion_outputs(
                peaks = peaks, 
                peaks_locations = peaks_locations, 
                sampling_frequency = si_recording.sampling_frequency, 
                motion = motion, 
                extra_check = extra_check,  
                peaks_label = peaks_label, 
                loc_label = loc_label,
                motion_label = motion_label,
                minY_loc = minY_loc, 
                maxY_loc = maxY_loc, 
                folderPlots = elecGroupSessFolder, 
                prefixRec = elecGroupSessName, 
                showPlots = False, 
                savePlots = True, 
                concatenationTimes = si_recording.get_annotation('concatenationTimes'), 
                verbose = True
            )

            ####################################################################################################################################
            #                                           Interpolate motion
            ####################################################################################################################################

            ##################################################################
            print('motion interpolaton : ...' + peaks_label + '-' + loc_label + '-' + motion_interpolation_label + '....')

            si_recording_motion = interpolate_motion(
                recording = si_recording,
                motion = motion,
                border_mode = interpolate_options['border_mode'],
                spatial_interpolation_method = interpolate_options['method'],
                sigma_um = step_chan, # Used in the "kriging" formula
                p=1, # Used in the "kriging" formula
                num_closest=3, # Number of closest channels used by "idw" method for interpolation
            )
            
            si_recording_motion.annotate(motion_corrected = True)
            si_recording_motion.annotate(motion_method_label = si_recording_sufix[1:])

            ###########################################################
            # PLOT NEW PEAKS LOCATIONS AS A FUNCTION OF TIME
            print('\nPlotting Motion Corrected Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

            SI_tools.plot_peakLocations(
                si_recording = si_recording_motion, 
                folderPeaks = folder_temp,  
                extra_recording_label = '_'+ motion_interpolation_label,
                peaks_options = peaks_options, 
                locations_options = locations_options, 
                rewrite = False, 
                locationsSubSampled = True,
                savePlots = True, 
                showPlots = False,
                folderPlots = elecGroupSessFolder
            )

            ####################################################################################################################################
            #  If NWB files were not move from original location: Save Recording Object (LAZY without traces)
            if not copyFiles:
                si_recording_motion.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording'))
    else:
        si_recording_sufix = ''

    if return_recording:
        if si_recording.get_num_channels()>nearest_chans and do_motion:
            return si_recording_motion, si_recording_sufix
        else:
            return si_recording, si_recording_sufix
    else:
        SI_tools.clear_SItempdir(verbose=False)
        


####################################################################################################################################################################################
#                                                              MAIN FUNCTION TO RUN SORTING (& PREPRO):
####################################################################################################################################################################################
#  It will run a sorting method (default Kilosort4 or Mountainsort5) 
#  In case there is no preprocessed SI-recording object to retrieve, it will run all the preprocessed steps (see "run_prepro" function for details about preprocessing steps)
#  "recording" & "sorting_analyzer" will be created in the parentSortingFolder: 
#           Note: if parentSortingFolder is not provided, then:
#                       if exporting "sorting_analyzer" to phy is True, both objects will be created temporary and deleted at the end
#                       if exporting "sorting_analyzer" to phy is False, both objects will be created in the same parentPreproFolder
####################################################################################################################################################################################
def run_prepro_and_sorting(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay, parentSortingFolder = None, parentPhyFolder = None, 
        electrodeGroup_sessions=None, electrodeGroup_Name=None, session_index=None,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = False,
        run_sorting = True,
        sorterName = sorterName_default,
        sorter_detect_threshold = None,
        sorter_nearest_chans = None,
        export2phy = True, 
        export2phy_with_recording = True
    ):    
    
    if electrodeGroup_sessions is None:
        electrodeGroups = SI_tools.getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)
        electrodeGroup_sessions = SI_tools.select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=session_index)

    elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName'])

    si_recording, si_recording_sufix = run_prepro(parentRecordingFolder, parentPreproFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_sessions=electrodeGroup_sessions,
                        local_radius_chans = local_radius_chans, 
                        noisy_freq = noisy_freq, 
                        ms_before = ms_before, 
                        ms_after = ms_after, 
                        peak_sign = peak_sign,
                        nearest_chans = nearest_chans, 
                        peak_detect_threshold = peak_detect_threshold, 
                        do_motion = do_motion,
                        motion_rigid = motion_rigid, 
                        motion_options = motion_options, 
                        interpolate_options = interpolate_options,
                        localProcess_NWB = localProcess_NWB,
                        rewrite_prepro = rewrite_prepro,
                        return_recording = True     
                    )
    
    if run_sorting:

        """
        ######################################################################################################################################################
        # Check for compatibility with sorter & number of channels:
        # mountainSort is recommended for single channel
        if si_recording.get_num_channels()==1 and sorterName =='kilosort4':
            continueVal = input('\nWARNING¡¡¡ The recording only contains 1 channel, it is recommended to run mountainsort5 instead of kilosort4\n\tDo you want to continue? (Y/N)')
            if continueVal.upper()=='N':
                raise Exception('Aborted by user, incompatibility between kilosort4 and the number of channels in the recording (n={})'.format(si_recording.get_num_channels()))
        """

        if si_recording.get_num_channels()==1 and sorterName =='kilosort4':
            print('\nWARNING¡¡¡\n\tThe recording contains 1 channel, it is recommended to run mountainsort5 instead of kilosort4\n')
        
        ######################################################################################################################################################
        # if parentSortingFolder is not provided, then:
        #       if exporting "sorting_analyzer" to phy is True, RECORDING & SORTING ANALYZER will be created temporary and deleted at the end
        #       if exporting "sorting_analyzer" to phy is False, RECORDING & SORTING ANALYZER will be created in the same parentPreproFolder
        #######################################################################################################################################################
        folder_temp = os.environ.get('SI_PROCESSOR_TEMPDIR')
        temp_analizer = False
        if parentSortingFolder is None:
            if export2phy:
                parentSortingFolder = folder_temp
                temp_analizer = True
            else:   
                parentSortingFolder = parentPreproFolder
        else:
            if not os.path.isdir(parentSortingFolder):
                os.makedirs(parentSortingFolder)

        #######################################################################################################################
        # Get/Create FOLDER paths to save RECORDING & SORTER & SORTING ANALYZER
        #######################################################################################################################
        # Create/Confirm General Session Folder exists
        sessionSortingFolder = os.path.join(os.path.abspath(parentSortingFolder), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
        if not os.path.isdir(sessionSortingFolder):
            os.makedirs(sessionSortingFolder)
        
        #############################################################################
        # Create/Confirm Folder path of the ElectrodeGroup-Session
        elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName'])
        elecGroupSessSortingFolder = os.path.join(sessionSortingFolder, elecGroupSessName)
        if not os.path.isdir(elecGroupSessSortingFolder):  
            os.mkdir(elecGroupSessSortingFolder)

        ###############################################################################################################################
        # Save Recording binary
        ###############################################################################################################################
        print('prepraring to write Recording into binary......\n')
        
        si_recording.save(folder=os.path.join(elecGroupSessSortingFolder, elecGroupSessName + si_recording_sufix + '_binary'), overwrite=True, **job_kwargs)

        del si_recording
        
        si_recording_loaded = si.load_extractor(os.path.join(elecGroupSessSortingFolder, elecGroupSessName + si_recording_sufix + '_binary'))

        print('reading Recording from binary successful¡......\n')

        #########################################################################################################
        # RUN SORTER 
        #########################################################################################################
        nChans = si_recording_loaded.get_num_channels()
        step_chan = si_recording_loaded.get_annotation('y_contacts_distance_um')
        sampling_frequency = si_recording_loaded.sampling_frequency

        #########################################################################################################
        # Whiten the recording
        recording_to_sort = spre.whiten(recording=si_recording_loaded, mode='local', radius_um = step_chan*3)
        recording_to_sort.annotate(is_whitened=True) 

        if sorter_nearest_chans is None:
            sorter_nearest_chans = nearest_chans

        sorter_info = get_sorting_params(sorterName, nChans, step_chan, sampling_frequency, ms_before, ms_after, peak_sign, sorter_nearest_chans, sorter_whitening=False, 
                                        detect_threshold=sorter_detect_threshold)
        
        folder_sorter = os.path.join(elecGroupSessSortingFolder, elecGroupSessName + si_recording_sufix + '_' + sorter_info['sorter_label'] + '_results')

        run_sorter(
            sorter_name = sorter_info['sorter_name'],
            recording = recording_to_sort,
            output_folder = folder_sorter,
            with_output = False,
            remove_existing_folder = True,
            delete_output_folder = False,
            docker_image = sorter_info['useDocker'],
            verbose = True,
            **sorter_info['params2update']
        )
            
        sorter_loaded = read_sorter_folder(folder_sorter, register_recording=False)

        #######################################################################################################################
        #                                    CREATE SORTER ANALYZER
        #######################################################################################################################

        print('\nCreating Sorting Analyzer......\n')

        # Returns a new sorting object which contains only units with at least one spike.
        sorting = sorter_loaded.remove_empty_units() 

        # Excess spikes are the ones exceeding a recording number of samples, for each segment
        sorting2 = scur.remove_excess_spikes(sorting, si_recording_loaded) 

        # Spikes are considered duplicated if they are less than x ms apart where x is the censored period.
        sorting3 = scur.remove_duplicated_spikes(sorting2, censored_period_ms=0.3, method="keep_first_iterative") 

        # Removes redundant or duplicate units by comparing the sorting output with itself
        sorting4 = scur.remove_redundant_units(sorting3, align=False, remove_strategy='max_spikes', peak_sign=peak_sign)

        #############################################################################
        # Create SORTING ANALYZER
        # When only one unit is detected, PHY can not create/load features and it will not launch
        # Therefore, the sorting analyzer and the recording object should not be deleted
        # Ensure saving SORTING ANALYZER when only one unit was found:
        if sorting4.get_num_units()<2 and temp_analizer:

            parentSortingAnalyzer = parentPreproFolder

            #######################################################################################################################
            # Get/Create FOLDER paths to save RECORDING & SORTING ANALYZER
            #######################################################################################################################
            # Create/Confirm General Session Folder exists
            sessionSortingAnalyzerFolder = os.path.join(os.path.abspath(parentSortingAnalyzer), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
            if not os.path.isdir(sessionSortingAnalyzerFolder):
                os.makedirs(sessionSortingAnalyzerFolder)
            
            #############################################################################
            # Create/Confirm Folder path of the ElectrodeGroup-Session
            elecGroupSessSortingAnalyzerFolder = os.path.join(sessionSortingAnalyzerFolder, elecGroupSessName)
            if not os.path.isdir(elecGroupSessSortingAnalyzerFolder):  
                os.mkdir(elecGroupSessSortingAnalyzerFolder)
            
            folder_analyzer = os.path.join(elecGroupSessSortingAnalyzerFolder, elecGroupSessName + si_recording_sufix + '_' + sorter_info['sorter_label'])

            print('WARINING¡ Only ONE UNIT was found¡¡\nSorting analyzer will be save instead of phy¡¡\nPrepraring to write Recording into Preprocessing folder......\n')
        
            si_recording_loaded.save(folder=os.path.join(elecGroupSessSortingAnalyzerFolder, elecGroupSessName + si_recording_sufix + '_binary'), overwrite=True, **job_kwargs)

            si_recording_reLoaded = si.load_extractor(os.path.join(elecGroupSessSortingAnalyzerFolder, elecGroupSessName + si_recording_sufix + '_binary'))

            sorting_analyzer = si.create_sorting_analyzer(
                sorting=sorting4, 
                recording=si_recording_reLoaded, 
                format='binary_folder', 
                folder=folder_analyzer, 
                overwrite=True,
                sparse=True, 
                **sorter_info['estimate_sparsity_params'],
                **job_kwargs
                )
            
            export2phy = False

        else:

            folder_analyzer = os.path.join(elecGroupSessSortingFolder, elecGroupSessName + si_recording_sufix + '_' + sorter_info['sorter_label'])

            sorting_analyzer = si.create_sorting_analyzer(
                sorting=sorting4, 
                recording=si_recording_loaded, 
                format='binary_folder', 
                folder=folder_analyzer, 
                overwrite=True,
                sparse=True, 
                **sorter_info['estimate_sparsity_params'],
                **job_kwargs
                )
            
        print('\nComputing extensions......\n')

        sorting_analyzer.compute(sorter_info['sorting_analyzer_params'], save=True, **job_kwargs)

        print('\nComputing quality metrics......\n')

        sqm.compute_quality_metrics(sorting_analyzer, save=True, skip_pc_metrics=False, peak_sign=peak_sign, **job_kwargs)

        #######################################################################################################################
        #                                    EXPORT TO PHY
        #######################################################################################################################
        if export2phy:

            print('\nExporting Sorting Analyzer to PHY......\n')

            if parentPhyFolder is None:
                parentPhyFolder = parentPreproFolder
            else:
                if not os.path.isdir(parentPhyFolder):
                    os.makedirs(parentPhyFolder)

            #######################################################################################################################
            # Get/Create FOLDER paths to save SORTING ANALYZER
            #######################################################################################################################
            # Create/Confirm General Phy Session Folder exists
            sessionPhyFolder = os.path.join(os.path.abspath(parentPhyFolder), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
            if not os.path.isdir(sessionPhyFolder):
                os.makedirs(sessionPhyFolder)
            
            #############################################################################
            # Create/Confirm Folder path of the Phy ElectrodeGroup-Session
            elecGroupSessPhyFolder = os.path.join(sessionPhyFolder, elecGroupSessName)
            if not os.path.isdir(elecGroupSessPhyFolder):  
                os.mkdir(elecGroupSessPhyFolder)

            #############################################################################
            # Create/Confirm Folder of the SORTING ANALYZER
            folder_analyzer_phy = os.path.join(elecGroupSessPhyFolder, elecGroupSessName + si_recording_sufix + '_' + sorter_info['sorter_label'] + '_phy')

            export_to_phy(
                sorting_analyzer, 
                output_folder= folder_analyzer_phy,
                remove_if_exists = True,
                copy_binary = export2phy_with_recording,
                verbose = True, 
                **job_kwargs
            )
                
        #######################################################################################################################
        # DELETE SORTING RESULTS
        #######################################################################################################################
        del sorter_loaded, sorting, sorting2, sorting3, sorting4, si_recording_loaded

        print('\nDeleting temporal sorting files .............\n')
        shutil.rmtree(folder_sorter, ignore_errors=True)
    
    SI_tools.clear_SItempdir(verbose=False)