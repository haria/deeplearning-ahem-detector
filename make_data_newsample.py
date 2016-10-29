import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import os, sys
import utils as ut

# Load sound files
path = '/archive1/ahem_data'
sound_file_paths = [os.path.join(path, "provocation_dirty.wav"),
                    os.path.join(path, "provocation_clean.wav"),
                   ]

sound_names = ["dirty", "clean"]
raw_sounds = ut.load_sound_files(sound_file_paths)

windowsize = 6000  # size of sliding window (22050 samples == 0.5 sec)  
step       = 3000
maxfiles   = 10000

dimx = 6
dimy = 5

# create positive samples
audiosamples = raw_sounds[0]
numsamples = audiosamples.shape[0]
numfiles = 0

for x in xrange(0, numsamples-windowsize, step):
    numfiles += 1 
    b = x               # begin 
    e = b+windowsize    # end 
    ut.printStuff('Creating spectrum image new samples [%d-%d] of %d file %d',(b,e, numsamples, numfiles))
    filename = os.path.join(path, '/archive1/ahem_data/new_sample/partial_spectrum_%d.png'%x)
    ut.specgram_frombuffer(audiosamples[b:e], dimx, dimy, fname=filename, dpi=180)
    
    #if x == maxfiles:
    #    break
 
        
print('\nbye!\n')        