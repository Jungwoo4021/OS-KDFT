import os
import ast
import pandas as pd
import soundfile as sf

from ._dataclass import MusicGenreClassificationData

class FMA:
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    def __init__(self, path, size):
        assert size in [self.SMALL, self.MEDIUM, self.LARGE]
        self.genre_dict = {}
        self.class_weight = []
        self.size = size
        
        # parse .csv
        f = os.path.join(path, 'fma_metadata/tracks.csv')
        tracks = self.load_tracks(f)
        partition_size = tracks['set', 'subset'] <= size

        # train_set
        train_tracks = tracks['set', 'split'] == 'training'
        train_tracks = tracks.loc[partition_size & train_tracks, ('track', 'genre_top')]
        self.train_set = self.make_item_list(path, train_tracks)
        
        # class weight of train_set
        self.class_weight = [0 for _ in range(len(self.genre_dict.keys()))]
        for item in self.train_set:
            # counting
            self.class_weight[item.label] += 1
        m = max(self.class_weight)
        for i in range(len(self.class_weight)):
            # scaling
            self.class_weight[i] = m / self.class_weight[i]
        
        # val_set
        val_tracks = tracks['set', 'split'] == 'validation'
        val_tracks = tracks.loc[partition_size & val_tracks, ('track', 'genre_top')]
        self.val_set = self.make_item_list(path, val_tracks)
        
        # test_set
        test_tracks = tracks['set', 'split'] == 'test'
        test_tracks = tracks.loc[partition_size & test_tracks, ('track', 'genre_top')]
        self.test_set = self.make_item_list(path, test_tracks)
        
    def make_item_list(self, path, tracks):
        items = []
        files = tracks.index.tolist()
        genres = tracks.values.tolist()
        
        for i in range(len(files)):
            if type(genres[i]) is not str: 
                continue
                 
            try: sf.read(os.path.join(path, self.size, f'{files[i] // 1000:03d}/{files[i]:06d}.wav'), start=0, stop=1)
            except: continue
                        
            # set genre label
            try: self.genre_dict[genres[i]]
            except: self.genre_dict[genres[i]] = len(self.genre_dict.keys())
            
            # append train_set
            items.append(
                MusicGenreClassificationData(
                    path=os.path.join(path, self.size, f'{files[i] // 1000:03d}/{files[i]:06d}.wav'),
                    genre=genres[i],
                    label=self.genre_dict[genres[i]]
                )
            )
            
        return items
               
    def load_tracks(self, path):
        tracks = pd.read_csv(path, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                    ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                    ('album', 'date_created'), ('album', 'date_released'),
                    ('artist', 'date_created'), ('artist', 'active_year_begin'),
                    ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = (self.SMALL, self.MEDIUM, self.LARGE)
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                        pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                    ('album', 'type'), ('album', 'information'),
                    ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks