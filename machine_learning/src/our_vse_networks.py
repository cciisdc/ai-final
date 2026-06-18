import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

# Tire compounds used in the simulation
COMPOUNDS = ['A3', 'A4', 'A5', 'A6']
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

# FCY status encoding (from Heilmeier paper Section 3.2)
# 0 = no FCY, 1 = VSC started, 2 = VSC ongoing, 3 = SC started, 4 = SC ongoing
FCY_STATUSES = [0, 1, 2, 3, 4]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features_nn1(race_progress, tire_age_progress, remaining_stops,
                        fcy_status, position, close_ahead, used_2compounds,
                        rel_compound_num):
    """
    Build input feature vector for NN1 (pit stop decision network).

    Parameters:
        race_progress    : float, 0.0 to 1.0 (current lap / total laps)
        tire_age_progress: float, 0.0 to 1.0 (tire age / total laps)
        remaining_stops  : int,   0 to 3 (pit stops still to come)
        fcy_status       : int,   0-4 (safety car encoding)
        position         : int,   1 to 20 (current race position)
        close_ahead      : bool,  True if car ahead is within 1.5s
        used_2compounds  : bool,  True if driver already used 2 compound types
        rel_compound_num : int,   0-3 (relative compound index)
    """
    features = []

    # Numerical features (normalized 0-1)
    features.append(float(race_progress))        # how far into race
    features.append(float(tire_age_progress))    # how worn are tires

    # Bucketized: remaining pit stops → one-hot (4 classes: 0,1,2,3)
    remaining_onehot = [0.0] * 4
    remaining_onehot[min(int(remaining_stops), 3)] = 1.0
    features.extend(remaining_onehot)

    # Categorical: FCY status → one-hot (5 classes: 0,1,2,3,4)
    fcy_onehot = [0.0] * 5
    fcy_onehot[int(fcy_status)] = 1.0
    features.extend(fcy_onehot)

    # Numerical: normalized position (1-20 → 0.0-1.0)
    features.append(float(position) / 20.0)

    # Binary flags
    features.append(1.0 if close_ahead else 0.0)
    features.append(1.0 if used_2compounds else 0.0)

    # Categorical: relative compound → one-hot (4 classes)
    compound_onehot = [0.0] * 4
    compound_onehot[min(int(rel_compound_num), 3)] = 1.0
    features.extend(compound_onehot)

    return np.array(features, dtype=np.float32)


def build_features_nn2(race_progress, remaining_stops, fcy_status,
                        position, rel_compound_num, used_2compounds):
    """
    Build input feature vector for NN2 (compound choice network).

    Parameters:
        race_progress    : float, 0.0 to 1.0
        remaining_stops  : int,   0 to 3
        fcy_status       : int,   0-4
        position         : int,   1 to 20
        rel_compound_num : int,   0-3
        used_2compounds  : bool
    """
    features = []

    # Numerical
    features.append(float(race_progress))
    features.append(float(position) / 20.0)

    # Bucketized: remaining stops → one-hot
    remaining_onehot = [0.0] * 4
    remaining_onehot[min(int(remaining_stops), 3)] = 1.0
    features.extend(remaining_onehot)

    # Categorical: FCY status → one-hot
    fcy_onehot = [0.0] * 5
    fcy_onehot[int(fcy_status)] = 1.0
    features.extend(fcy_onehot)

    # Categorical: current compound → one-hot
    compound_onehot = [0.0] * 4
    compound_onehot[min(int(rel_compound_num), 3)] = 1.0
    features.extend(compound_onehot)

    # Binary
    features.append(1.0 if used_2compounds else 0.0)

    return np.array(features, dtype=np.float32)


# =============================================================================
# NN1: PIT STOP DECISION NETWORK
# Hybrid FFNN + LSTM (Table 9, Heilmeier et al. 2020)
# Binary classifier: pit (1) or don't pit (0)
# =============================================================================

def build_nn1(input_dim, sequence_length=5):

    # -----------------------------------------------------------------
    # PHASE 1: FFNN (Feedforward Neural Network)
    # -----------------------------------------------------------------
    ffnn_input = keras.Input(shape=(input_dim,), name='ffnn_input')

    # Hidden layer 1: 64 neurons, ReLU activation
    x = layers.Dense(64, activation='relu', name='ffnn_layer_1')(ffnn_input)

    # Hidden layer 2: 64 neurons, ReLU activation
    x = layers.Dense(64, activation='relu', name='ffnn_layer_2')(x)

    # Hidden layer 3: 64 neurons, ReLU activation
    x = layers.Dense(64, activation='relu', name='ffnn_layer_3')(x)

    ffnn_model = keras.Model(inputs=ffnn_input, outputs=x, name='ffnn_part')

    # -----------------------------------------------------------------
    # PHASE 2: LSTM (Long Short-Term Memory)
    # -----------------------------------------------------------------
    sequence_input = keras.Input(
        shape=(sequence_length, input_dim), name='sequence_input'
    )

    # Apply FFNN to each lap in the sequence
    ffnn_applied = layers.TimeDistributed(
        ffnn_model, name='ffnn_over_sequence'
    )(sequence_input)

    # LSTM layer: 1 neuron (as per Table 9)
    # return_sequences=False → only output from final timestep
    lstm_out = layers.LSTM(1, name='lstm_layer')(ffnn_applied)

    # Output layer: 1 neuron, sigmoid → probability of pitting
    output = layers.Dense(
        1, activation='sigmoid', name='pit_probability'
    )(lstm_out)

    model = keras.Model(
        inputs=sequence_input,
        outputs=output,
        name='NN1_PitStopDecision'
    )

    # Compile with binary crossentropy (binary classification problem)
    # Using class weights to handle imbalanced data (~3.1% pit laps)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================================
# NN2: TIRE COMPOUND CHOICE NETWORK
# =============================================================================

def build_nn2(input_dim):
    
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(input_dim,), name='compound_features'),

        # Hidden layer: 32 neurons, ReLU activation (from Table 14)
        layers.Dense(32, activation='relu', name='hidden_layer'),

        # Output layer: 4 neurons (A3, A4, A5, A6), softmax
        # Softmax converts raw scores to probabilities that sum to 1.0
        # e.g. [0.05, 0.70, 0.20, 0.05] → pick A4 (index 1, highest)
        layers.Dense(
            len(COMPOUNDS), activation='softmax', name='compound_probabilities'
        ),
    ], name='NN2_CompoundChoice')

    # Compile with categorical crossentropy (multi-class problem)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================================

# =============================================================================

def training_data(n_samples=5000, sequence_length=5,
                                      nn1_input_dim=18, nn2_input_dim=15):
    
    np.random.seed(42)

    # NN1 data: sequences of laps
    X_nn1 = np.random.rand(n_samples, sequence_length, nn1_input_dim).astype(np.float32)

    # Labels: ~3.1% pit stops (matching real F1 imbalance from paper)
    # Pit stops more likely when tire age progress > 0.6 (worn tires)
    pit_probability = np.where(
        X_nn1[:, -1, 1] > 0.6,   # tire age in last lap of sequence
        0.15,                      # 15% chance when tires worn
        0.01                       # 1% chance otherwise
    )
    y_nn1 = (np.random.rand(n_samples) < pit_probability).astype(np.float32)

    # NN2 data: single lap features
    X_nn2 = np.random.rand(n_samples, nn2_input_dim).astype(np.float32)

    # Compound labels: biased by race progress
    # Early race → softer compounds, late race → harder compounds
    race_progress = X_nn2[:, 0]
    compound_probs = np.column_stack([
        0.1 + 0.3 * race_progress,        # A3 (hard): more likely late
        0.3 * np.ones(n_samples),          # A4 (medium): always common
        0.4 - 0.2 * race_progress,         # A5 (soft): more likely early
        0.2 - 0.1 * race_progress,         # A6 (supersoft): early only
    ])
    compound_probs = compound_probs / compound_probs.sum(axis=1, keepdims=True)
    y_nn2 = np.array([
        np.random.choice(4, p=p) for p in compound_probs
    ])

    return X_nn1, y_nn1, X_nn2, y_nn2


# =============================================================================
# TRAINING
# =============================================================================

def train_nn1(model, X_train, y_train, X_val, y_val):

    # Calculate class weights (inverse of class frequency)
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    total = len(y_train)

    class_weight = {
        0: total / (2 * n_negative),   # weight for 'no pit'
        1: total / (2 * n_positive),   # weight for 'pit' (much higher)
    }

    print(f"  Class weights — no pit: {class_weight[0]:.2f}, "
          f"pit: {class_weight[1]:.2f}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[
            # Early stopping: stop if validation loss doesn't improve
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=0
    )

    return history


def train_nn2(model, X_train, y_train, X_val, y_val):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=0
    )
    return history


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_nn1(model, X_test, y_test, threshold=0.5):
    
    print("\n" + "="*50)
    print("NN1 EVALUATION — Pit Stop Decision Network")
    print("="*50)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > threshold).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 Score : {f1:.4f}  ← PRIMARY METRIC")
    print(f"  (Heilmeier reports F1 ≈ 0.59 on real F1 data)")
    print()
    print("  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['No Pit', 'Pit'],
        zero_division=0
    ))

    # Show probability distribution
    print(f"  Avg probability when true label = pit:    "
          f"{y_prob[y_test==1].mean():.3f}")
    print(f"  Avg probability when true label = no pit: "
          f"{y_prob[y_test==0].mean():.3f}")

    return f1


def evaluate_nn2(model, X_test, y_test):

    print("\n" + "="*50)
    print("NN2 EVALUATION — Tire Compound Choice Network")
    print("="*50)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)  ← PRIMARY METRIC")
    print(f"  (Heilmeier reports accuracy ≈ 0.77 on real F1 data)")
    print()
    print("  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=COMPOUNDS,
        zero_division=0
    ))

    return acc


# =============================================================================
# INFERENCE
# =============================================================================

class OurVSE:

    def __init__(self, nn1_model, nn2_model, threshold=0.5):
        self.nn1 = nn1_model    # pit stop decision network
        self.nn2 = nn2_model    # compound choice network
        self.threshold = threshold
        self.lap_history = []   # stores recent lap features for LSTM

    def reset(self):
        self.lap_history = []

    def make_decision(self, race_progress, tire_age_progress,
                       remaining_stops, fcy_status, position,
                       close_ahead, used_2compounds, rel_compound_num):

        # Build feature vector for this lap
        features_nn1 = build_features_nn1(
            race_progress, tire_age_progress, remaining_stops,
            fcy_status, position, close_ahead,
            used_2compounds, rel_compound_num
        )

        # Add to lap history for LSTM
        self.lap_history.append(features_nn1)
        if len(self.lap_history) > 5:
            self.lap_history.pop(0)

        # Need at least 5 laps of history for LSTM
        if len(self.lap_history) < 5:
            return None

        # ---------------------------------------------------------
        # NN1
        # ---------------------------------------------------------
        sequence = np.array(self.lap_history, dtype=np.float32)
        sequence = sequence[np.newaxis, :, :]  # shape: (1, 5, n_features)

        pit_probability = self.nn1.predict(sequence, verbose=0)[0][0]

        if pit_probability <= self.threshold:
            return None  # don't pit

        # ---------------------------------------------------------
        # NN2
        # ---------------------------------------------------------
        features_nn2 = build_features_nn2(
            race_progress, remaining_stops, fcy_status,
            position, rel_compound_num, used_2compounds
        )

        compound_probs = self.nn2.predict(
            features_nn2[np.newaxis, :], verbose=0
        )[0]

        chosen_idx = np.argmax(compound_probs)
        chosen_compound = COMPOUNDS[chosen_idx]

        return chosen_compound


# =============================================================================
# DEMO
# =============================================================================

def demo_race(vse, total_laps=71):

    print("\n" + "="*50)
    print("DEMO RACE — Our VSE Making Lap-by-Lap Decisions")
    print("="*50)

    strategy = [[0, 'A5']]  # start on soft tires
    current_compound = 'A5'
    tire_age = 0
    pit_count = 0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        race_progress = lap / total_laps
        tire_age_progress = tire_age / total_laps
        remaining_stops = max(0, 2 - pit_count)
        fcy_status = 3 if lap == 30 else 0  # safety car on lap 30
        position = max(1, 5 - pit_count)
        close_ahead = tire_age > 20
        used_2compounds = pit_count > 0
        rel_compound_num = COMPOUND_TO_IDX.get(current_compound, 0)

        decision = vse.make_decision(
            race_progress, tire_age_progress, remaining_stops,
            fcy_status, position, close_ahead,
            used_2compounds, rel_compound_num
        )

        if decision is not None:
            strategy.append([lap, decision])
            current_compound = decision
            tire_age = 0
            pit_count += 1

    print(f"\n  Final strategy: {strategy}")
    print(f"  Total pit stops: {pit_count}")
    return strategy


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # STEP 1:data
    # ------------------------------------------------------------------

    NN1_INPUT_DIM = 18
    NN2_INPUT_DIM = 15
    SEQUENCE_LENGTH = 5

    X_nn1, y_nn1, X_nn2, y_nn2 = training_data(
        n_samples=5000,
        sequence_length=SEQUENCE_LENGTH,
        nn1_input_dim=NN1_INPUT_DIM,
        nn2_input_dim=NN2_INPUT_DIM
    )

    # Train/val/test split (70/15/15)
    n = len(y_nn1)
    i1, i2 = int(0.7*n), int(0.85*n)

    X_nn1_train, X_nn1_val, X_nn1_test = X_nn1[:i1], X_nn1[i1:i2], X_nn1[i2:]
    y_nn1_train, y_nn1_val, y_nn1_test = y_nn1[:i1], y_nn1[i1:i2], y_nn1[i2:]
    X_nn2_train, X_nn2_val, X_nn2_test = X_nn2[:i1], X_nn2[i1:i2], X_nn2[i2:]
    y_nn2_train, y_nn2_val, y_nn2_test = y_nn2[:i1], y_nn2[i1:i2], y_nn2[i2:]

    # ------------------------------------------------------------------
    # STEP 2: Build networks
    # ------------------------------------------------------------------
    print("\n[2/5] Building neural network architectures...")

    nn1 = build_nn1(input_dim=NN1_INPUT_DIM, sequence_length=SEQUENCE_LENGTH)
    nn2 = build_nn2(input_dim=NN2_INPUT_DIM)

    print("\n  NN1 Summary:")
    nn1.summary()
    print("\n  NN2 Summary:")
    nn2.summary()

    # ------------------------------------------------------------------
    # STEP 3: Train networks
    # ------------------------------------------------------------------
    print("\n[3/5] Training NN1 (pit stop decision)...")
    train_nn1(nn1, X_nn1_train, y_nn1_train, X_nn1_val, y_nn1_val)
    print("      NN1 training complete.")

    print("\n      Training NN2 (compound choice)...")
    train_nn2(nn2, X_nn2_train, y_nn2_train, X_nn2_val, y_nn2_val)
    print("      NN2 training complete.")

    # ------------------------------------------------------------------
    # STEP 4: Evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating networks...")
    f1 = evaluate_nn1(nn1, X_nn1_test, y_nn1_test)
    acc = evaluate_nn2(nn2, X_nn2_test, y_nn2_test)

    # ------------------------------------------------------------------
    # STEP 5: Demo race
    # ------------------------------------------------------------------
    print("\n[5/5] Running demo race with our trained VSE...")
    our_vse = OurVSE(nn1, nn2, threshold=0.5)
    strategy = demo_race(our_vse)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"  NN1 F1 Score : {f1:.4f}")
    print(f"  NN2 Accuracy : {acc:.4f}")
