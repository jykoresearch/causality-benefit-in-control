import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class LSTMOrdinalLogisticRegression(nn.Module):
    _N_ENV = 3

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_ratio, specific_label=None,
                 last_step_direct=False):
        """
        LSTM-based ordinal logistic regression model with class weighting.

        ``specific_label`` is the class that ``ordinal_loss`` weights differently from the rest.

        ``last_step_direct`` selects the "causal model + R_{t-1}" architecture (causal == 4):
        the LSTM digests ONLY the env triple (Ti, rh, mrt) over the whole window, and the
        most-recent override R (the last column of the 4-feature input) is concatenated to
        the LSTM hidden state at the FC head. Otherwise the LSTM sees all ``input_size``
        features and the head takes the hidden state alone.
        """
        super(LSTMOrdinalLogisticRegression, self).__init__()
        self.last_step_direct = last_step_direct
        if last_step_direct:
            self.lstm = nn.LSTM(self._N_ENV, hidden_size, num_layers, batch_first=True, bias=True)
            self.dropout = nn.Dropout(dropout_ratio)
            self.fc = nn.Linear(hidden_size + 1, 1, bias=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True)
            self.dropout = nn.Dropout(dropout_ratio)
            self.fc = nn.Linear(hidden_size, 1, bias=True)
        self.exponential = torch.exp

        self.raw_cutpoints = nn.Parameter(torch.randn(num_classes - 1))
        self.specific_label = specific_label


    def forward(self, x):
        """
        x: (batch_size, seq_length, input_size). Returns exp(linear predictor), shape
        (batch_size, 1) -- the exponential keeps it positive, matching the positive,
        increasing cutpoints used downstream.
        """
        if self.last_step_direct:
            out, _ = self.lstm(x[:, :, :self._N_ENV])
            h = self.dropout(out[:, -1, :])
            direct = x[:, -1, self._N_ENV:self._N_ENV + 1]
            logits = self.fc(torch.cat([h, direct], dim=1))
            return self.exponential(logits)

        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        logits = self.fc(out)
        result = self.exponential(logits)
        return result

    def ordinal_loss(self, logits, target, weight=None):
        """
        Ordinal logistic regression loss with proportional odds and per-class weighting.

        Cumulative probabilities, with cutpoints = cumsum(softplus(raw_cutpoints)):
            P(Y <= k) = sigmoid(cutpoint_k - logit)
        and hence class probabilities:
            P(Y = 0) = sigmoid(cutpoint_0 - logit)
            P(Y = k) = sigmoid(cutpoint_k - logit) - sigmoid(cutpoint_{k-1} - logit)   for 1 <= k <= num_classes-2
            P(Y = num_classes - 1) = 1 - sigmoid(cutpoint_{num_classes-2} - logit)

        ``logits`` is (batch_size, 1) as returned by ``forward``; ``target`` is (batch_size,)
        with labels in {0, ..., num_classes-1}. ``weight`` is a 2-element sequence indexed
        positionally: ``weight[0]`` applies where ``target == self.specific_label``,
        ``weight[1]`` everywhere else, and it is ignored unless ``specific_label`` is set.
        """
        batch_size = logits.size(0)
        cutpoints = torch.cumsum(F.softplus(self.raw_cutpoints), dim=0).view(1, -1)
        logits = logits.view(-1, 1)

        prob_cum = torch.sigmoid(cutpoints - logits)
        
        p0 = prob_cum[:, :1]
        if cutpoints.size(1) > 1:
            p_intermediate = prob_cum[:, 1:] - prob_cum[:, :-1]
        else:
            p_intermediate = torch.empty(batch_size, 0, device=logits.device)
        p_last = 1 - prob_cum[:, -1:]

        prob_classes = torch.cat([p0, p_intermediate, p_last], dim=1)

        target_one_hot = F.one_hot(target, num_classes=prob_classes.size(1)).float()
        nll = -torch.sum(target_one_hot * torch.log(prob_classes + 1e-8), dim=1)

        if weight is not None and self.specific_label is not None:
            weights_tensor = torch.where(target == self.specific_label, weight[0], weight[1])
        else:
            weights_tensor = torch.ones_like(nll)

        return (nll * weights_tensor).mean()

    def predict_proba(self, x):
        """
        Sample an ordinal class from the predicted distribution for x
        (batch_size, seq_length, input_size). Returns (y_pred, prob_classes): a stochastic
        inverse-CDF draw of shape (batch_size,) and the probabilities it was drawn from,
        shape (batch_size, num_classes). The draw is what makes the simulated occupant
        behave stochastically rather than always taking the modal response.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            cutpoints = torch.cumsum(F.softplus(self.raw_cutpoints), dim=0).view(1, -1)
            logits = logits.view(-1, 1)

            prob_cum = torch.sigmoid(cutpoints - logits)

            p0 = prob_cum[:, :1]
            if cutpoints.size(1) > 1:
                p_intermediate = prob_cum[:, 1:] - prob_cum[:, :-1]
            else:
                p_intermediate = torch.empty(logits.size(0), 0, device=logits.device)
            p_last = 1 - prob_cum[:, -1:]
            prob_classes = torch.cat([p0, p_intermediate, p_last], dim=1)
            
            
            prob_classes_cum = torch.cumsum(prob_classes, dim=1)
            p_rand = torch.rand(prob_classes_cum.shape[0])
            y_pred = torch.zeros(prob_classes_cum.shape[0])

            for i in range(prob_classes_cum.shape[0]):
                for j in range(prob_classes_cum.shape[1]):
                    if p_rand[i] < prob_classes_cum[i,j]:
                        y_pred[i] = j
                        break
                if p_rand[i] > prob_classes_cum[i,-1]:
                    y_pred[i] = prob_classes_cum.shape[1]

            return y_pred, prob_classes

def create_sequences(data, labels, seq_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data.iloc[i:i + seq_length].values
        label = labels.iloc[i + seq_length - 1].values
        sequences.append(seq)
        sequence_labels.append(label)
    return np.array(sequences), np.array(sequence_labels)

def concatenate_sequences(dfs, seq_length):

    for i in range(len(dfs)):
        sequences, sequence_labels = create_sequences(dfs[i][0], dfs[i][1], seq_length)
        if i == 0:
            concat_sequences = sequences
            concat_sequence_labels = sequence_labels
        else:
            concat_sequences = np.concatenate((concat_sequences, sequences), axis=0)
            concat_sequence_labels = np.concatenate((concat_sequence_labels, sequence_labels), axis=0)

    return concat_sequences, concat_sequence_labels

def process_dfs(dfs, causal = 0):
    processed_dfs = []
    for df in dfs:

        if causal == 0:
            selected_columns = ['time', 'day_of_year', 'day_of_week', 'override', 'dt_obs', 
                                'Ti', 'rh', 'mrt', 'outdoor_temp', 'outdoor_rh', 'direct_solar', 'diffuse_solar', 
                                'ground_solar', 'wind_speed', 'elec_hvac', 'cooling_setpoint', 
                                'hold_status', 'hold_timer']
        elif causal == 4:
            selected_columns = ['override', 'dt_obs',
                                'Ti', 'rh', 'mrt']
        else:
            raise ValueError(f"process_dfs: unsupported causal={causal!r}; expected 0 or 4")


        df_selected = df[selected_columns]
        df_selected = df_selected.copy()
        
        if causal == 0:
            df_selected['cooling_setpoint (prev)'] = df_selected['cooling_setpoint'].shift(1)
            df_selected['hold_timer (prev)'] = df_selected['hold_timer'].shift(1)
            df_selected['hold_status (prev)'] = df_selected['hold_status'].shift(1)
            df_selected['override (prev)'] = df_selected['override'].shift(1)
            df_selected.drop(columns=['cooling_setpoint', 'hold_status', 'hold_timer'], inplace=True)

            df_selected['time_sin'] = np.sin(2 * np.pi * df_selected['time'] / 24)
            df_selected['time_cos'] = np.cos(2 * np.pi * df_selected['time'] / 24)
            df_selected['day_of_year_sin'] = np.sin(2 * np.pi * df_selected['day_of_year'] / 365)
            df_selected['day_of_year_cos'] = np.cos(2 * np.pi * df_selected['day_of_year'] / 365)
            df_selected['day_of_week_sin'] = np.sin(2 * np.pi * df_selected['day_of_week'] / 7)
            df_selected['day_of_week_cos'] = np.cos(2 * np.pi * df_selected['day_of_week'] / 7)

            df_selected.drop(columns=['time', 'day_of_year', 'day_of_week'], inplace=True)
    
            assert df_selected.shape[1] == 22, f"Expected 22 columns, got {df_selected.shape[1]}"

        elif causal == 4:
            df_selected['override (prev)'] = df_selected['override'].shift(1)

        else:
            raise ValueError(f"process_dfs: unsupported causal={causal!r}; expected 0 or 4")

        df_selected.dropna(inplace=True)

        X = df_selected.drop(columns=['override','dt_obs'])

        
        bins = np.arange(-10, 5.5, 0.5)
        
        df_selected['dt_obs'] = pd.cut(df_selected['dt_obs'], 
                           bins=bins, 
                           labels=False,
                           include_lowest=True)
        

        y = df_selected[['dt_obs']]
        processed_dfs.append((X, y))
        
    return processed_dfs

def divide_datasets(train_sequences, train_sequence_labels):
    """
    Splits the dataset into training, validation and test sets (70/20/10).
    """
    train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(
        train_sequences, train_sequence_labels, test_size=0.1)

    train_sequences_split, vali_sequences, train_labels_split, vali_labels = train_test_split(
        train_val_sequences, train_val_labels, test_size=0.2/0.9
)
    
    return train_sequences_split, vali_sequences, test_sequences, train_labels_split, vali_labels, test_labels


def scale_sequences(train_sequences_split, vali_sequences, test_sequences, causal=4):
    scaler = StandardScaler()
    scaler.fit(train_sequences_split.reshape(-1, train_sequences_split.shape[-1]))
    scaled_train_sequences = scaler.transform(train_sequences_split.reshape(-1, train_sequences_split.shape[-1])).reshape(train_sequences_split.shape)
    scaled_vali_sequences = scaler.transform(vali_sequences.reshape(-1, vali_sequences.shape[-1])).reshape(vali_sequences.shape)
    scaled_test_sequences = scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)

    return scaled_train_sequences, scaled_vali_sequences, scaled_test_sequences, scaler


def create_dataloader(sequences, labels, batch_size):
    dataset = TensorDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def prep_data(dfs, causal, seq_length):

    train_sequences, train_sequence_labels = concatenate_sequences(dfs, seq_length)
    train_sequence_labels = train_sequence_labels[:, 0]  

    train_sequences_split, vali_sequences, test_sequences, train_labels_split, vali_labels, test_labels = divide_datasets(train_sequences, train_sequence_labels)

    if causal == 0:
        train_sequences_split[:, :, :12], vali_sequences[:, :, :12], test_sequences[:, :, :12], scaler = scale_sequences(train_sequences_split[:, :, :12], vali_sequences[:, :, :12], test_sequences[:, :, :12], causal=causal)

    elif causal == 4:
        train_sequences_split[:, :, :3], vali_sequences[:, :, :3], test_sequences[:, :, :3], scaler = scale_sequences(train_sequences_split[:, :, :3], vali_sequences[:, :, :3], test_sequences[:, :, :3], causal=causal)

    else:
        raise ValueError(f"prep_data: unsupported causal={causal!r}; expected 0 or 4")

    num_classes = np.arange(-10, 5.5, 0.5).shape[0]
    num_classes = num_classes - 1

    unique_classes, class_counts_per_label = np.unique(train_labels_split, return_counts=True)
    most_frequent_class = unique_classes[np.argmax(class_counts_per_label)]

    class_weights = np.array([1,1])
    class_weights_vali = np.array([1,1])
    class_weights_test = np.array([1,1])
    class_weights_shift = np.array([1,1])

    train_sequences_tensor = torch.tensor(train_sequences_split, dtype=torch.float32)
    train_sequence_labels_tensor = torch.tensor(train_labels_split, dtype=torch.long)

    vali_sequences_tensor = torch.tensor(vali_sequences, dtype=torch.float32)
    vali_sequence_labels_tensor = torch.tensor(vali_labels, dtype=torch.long)

    test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32)
    test_sequence_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    return (train_sequences_tensor, train_sequence_labels_tensor,
            vali_sequences_tensor, vali_sequence_labels_tensor,
            test_sequences_tensor, test_sequence_labels_tensor,
            scaler, num_classes, most_frequent_class,
            class_weights, class_weights_vali, class_weights_test, class_weights_shift)
