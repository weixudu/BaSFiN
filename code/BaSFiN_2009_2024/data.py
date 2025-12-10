import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from collections import Counter

class Data:
    def __init__(self, path=None, symmetry=True, team_size=5, seed=None):
        self.seed = seed
        self.team_size = team_size
        self.symmetry = symmetry
        df = pd.read_csv(path)
        self.n = -1
        self.n_player = -1
        # Identify player columns
        player_cols = [f'player{i}' for i in range(1, 2 * self.team_size + 1)]
        assert all(col in df.columns for col in player_cols), "Missing player columns in data"

        # Count unique players (excluding -1) and create ID mapping
        all_players = pd.unique(df[player_cols].values.ravel())
        print("max allplayers",max(all_players))
        all_players = all_players[all_players != -1]
        self.n_player = len(all_players)
        self.n_individual = self.n_player
        self.max_player_id = int(max(all_players))
        self.player_id_to_index = {pid: idx for idx, pid in enumerate(sorted(all_players))}
        self.index_to_player_id = {idx: pid for pid, idx in self.player_id_to_index.items()}

        # Remap player IDs in the DataFrame
        for col in player_cols:
            df[col] = df[col].apply(lambda x: self.player_id_to_index.get(x, -1) if x != -1 else -1)

        # Verify mapping
        mapped_pids = pd.unique(df[player_cols].values.ravel())
        mapped_pids = mapped_pids[mapped_pids != -1]
        if mapped_pids.size > 0:
            min_mapped, max_mapped = mapped_pids.min(), mapped_pids.max()
            if max_mapped >= self.n_individual or min_mapped < 0:
                raise ValueError(f"Mapping failed: mapped IDs range {min_mapped} to {max_mapped}, expected 0 to {self.n_individual-1}")

        # Select relevant columns
        target_col = ['target'] if 'target' in df.columns else ['radiant_win']
        self.data = df.loc[:, ['year', 'id'] + player_cols + target_col].to_numpy()[:800000]
        print('Whole dataset size:', self.data.shape)
        print(f'Original player ID range: min={min(all_players)}, max={self.max_player_id}')
        print(f'Mapped player index range: 0 to {self.n_player-1}')
        self.split()
        self.train_valid = self.train  

    def split(self):
        # Define column names for DataFrame
        columns = ['year', 'id'] + [f'player{i}' for i in range(1, 2 * self.team_size + 1)] + ['target']
        df = pd.DataFrame(self.data, columns=columns)
        df = df.sort_values(by=['year', 'id']).reset_index(drop=True)

        # Get unique years and sort them
        years = sorted(df['year'].unique())
        if len(years) < 12:
            print(f"Warning: Found only {len(years)} years, expected at least 7")

        # Split based on years: first 5 for train, 6th for valid, 7th for test
        train_years = years[:10]
        valid_years = years[10:11]
        test_years = years[11:12]

        print(f"Train years: {train_years}")
        print(f"Valid year: {valid_years}")
        print(f"Test year: {test_years}")

        # Split data
        self.train = df[df['year'].isin(train_years)].to_numpy()
        self.valid = df[df['year'].isin(valid_years)].to_numpy()
        self.test = df[df['year'].isin(test_years)].to_numpy()

        # Verify splits
        for split, data in [('train', self.train), ('valid', self.valid), ('test', self.test)]:
            pids = data[:, 2:12].ravel()
            invalid_pids = pids[(pids != -1) & (pids >= self.n_individual)]
            if invalid_pids.size > 0:
                print(f"Invalid IDs in {split}: {np.unique(invalid_pids)}")
                raise ValueError(f"Found unmapped player IDs in {split}")

        # Calculate and print percentages
        total_samples = len(self.train) + len(self.valid) + len(self.test)
        train_percent = (len(self.train) / total_samples) * 100 if total_samples > 0 else 0
        val_percent = (len(self.valid) / total_samples) * 100 if total_samples > 0 else 0
        test_percent = (len(self.test) / total_samples) * 100 if total_samples > 0 else 0

        # Record player indices in each split
        self.train_players = set(self.select(self.train)[:, 1:].reshape(-1)) - {-1}
        valid_players = set(self.select(self.valid)[:, 1:].reshape(-1)) - {-1}
        test_players = set(self.select(self.test)[:, 1:].reshape(-1)) - {-1}

        all_players = self.train_players | valid_players | test_players
        valid_only = valid_players - self.train_players
        test_only = test_players - self.train_players
        valid_or_test_only = (valid_players | test_players) - self.train_players

        total_unique_players = len(all_players)
        valid_only_count = len(valid_only)
        test_only_count = len(test_only)
        valid_or_test_only_count = len(valid_or_test_only)

        print('Individual players in valid not in train:', valid_only_count)
        print('Individual players in test not in train:', test_only_count)
        print('Total unique players:', total_unique_players)
        print('Players only in valid or test (not in train):', valid_or_test_only_count)
        print('Percentage of players only in valid or test: {:.2f}%'.format(
            (valid_or_test_only_count / total_unique_players * 100) if total_unique_players > 0 else 0))
        
        print('Train shape:', self.train.shape)
        print('Valid shape:', self.valid.shape)
        print('Test shape:', self.test.shape)
        print(f'Train percentage: {train_percent:.2f}%')
        print(f'Validation percentage: {val_percent:.2f}%')
        print(f'Test percentage: {test_percent:.2f}%')

        train_cnt = Counter(self.select(self.train)[:, 1:].reshape(-1))
        valid_cnt = Counter(self.select(self.valid)[:, 1:].reshape(-1))
        test_cnt = Counter(self.select(self.test)[:, 1:].reshape(-1))
        self.player_cnt = train_cnt + valid_cnt + test_cnt

    def expand_training_data(self):
        if len(self.valid) == 0:
            print("Warning: Validation set is empty, no data to expand.")
            self.train_valid = self.train
            return
        self.train = np.vstack([self.train, self.valid])
        self.valid = np.array([])  # Clear validation set
        self.train_valid = self.train  # Update train_valid
        print(f"Expanded train shape: {self.train.shape}")
        print(f"Valid shape after expansion: {self.valid.shape}")
        print(f"Test shape: {self.test.shape}")
            
    def encode(self, data):
        t = self.team_size
        A = self.sparse(data[:, 1:1+t], self.n_player)
        B = self.sparse(data[:, 1+t:], self.n_player)

        if self.symmetry:
            return A + B * -1
        else:
            return hstack([A, B])

    def sparse(self, dense, n_individual):
        t = self.team_size
        n_match = len(dense)
        values = np.ones(n_match * t)
        rowptr = np.array([i * t for i in range(n_match + 1)])
        col_index = dense.reshape(-1)
        col_index = np.where(col_index == -1, self.n_player, col_index)
        A = csr_matrix((values, col_index, rowptr), shape=(n_match, n_individual + 1))[:, :n_individual]
        return A

    def get_all(self, type='train', encoding=False):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        elif type == 'train_valid':
            data = self.train_valid
        else:
            raise ValueError('Invalid type!')

        y = data[:, -1]
        data = self.select(data)

        if encoding:
            return self.encode(data), y
        else:
            return data, y

    def get_batch(self, batch_size=32, type='train', shuffle=False):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        elif type == 'train_valid':
            data = self.train_valid
        else:
            raise ValueError('Invalid type!')

        y = data[:, -1]
        data = self.select(data)
        length = len(data)
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = data[excerpt]
            yield X, y[excerpt]
            start_idx += batch_size

    def select(self, data):
        t = self.team_size
        data = data[:, 1:t*2+2]
        return data