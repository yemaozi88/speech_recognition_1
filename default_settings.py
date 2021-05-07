import os
#import pandas as pd


## direcotries.
home_dir = r'/home/takkan'
audio_corpora_dir = os.path.join(home_dir, 'experiments/taku_voice/wavs')
digit_dir = os.path.join(audio_corpora_dir, 'digit_trimmed')
vowel_dir = os.path.join(audio_corpora_dir, 'vowel_trimmed')
features_dir = os.path.join(home_dir, 'experiments/taku_voice/features')

# used to import other repositories.
# all related repos are expected to be under this directory.
repos_dir = os.path.join(home_dir, 'repos')

