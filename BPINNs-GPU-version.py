# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# ==================== GPU setting ====================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"all {len(gpus)} GPU")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"error: {e}")
else:
    print("warning: NO GPU")

tf.keras.backend.set_floatx('float32')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

tfd = tfp.distributions


# ==================== 1. SimpleBayesianPINNModel ====================
class SimpleBayesianPINNModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=[64, 32], dropout_rate=0.1,
                 activation='relu', use_batch_norm=False, kl_weight=1e-5):
        super(SimpleBayesianPINNModel, self).__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.kl_weight = kl_weight

        self.layers_list = []
        dims = [input_dim] + hidden_units
        for i in range(len(dims) - 1):
            self.layers_list.append(
                tf.keras.layers.Dense(dims[i + 1], activation=None, name=f'dense_{i}')
            )
            if use_batch_norm:
                self.layers_list.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))

            if activation == 'relu':
                self.layers_list.append(tf.keras.layers.ReLU(name=f'activation_{i}'))
            elif activation == 'tanh':
                self.layers_list.append(tf.keras.layers.Activation('tanh', name=f'activation_{i}'))
            elif activation == 'swish':
                self.layers_list.append(tf.keras.layers.Activation(tf.nn.swish, name=f'activation_{i}'))

            self.layers_list.append(tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i}'))

        self.carotenoids_output = tfp.layers.DenseVariational(
            1, make_posterior_fn=self._make_simple_posterior_fn,
            make_prior_fn=self._make_simple_prior_fn,
            kl_weight=self.kl_weight, activation='relu',
            name='bayesian_carotenoids_output'
        )

        self.tree_high_output = tfp.layers.DenseVariational(
            1, make_posterior_fn=self._make_simple_posterior_fn,
            make_prior_fn=self._make_simple_prior_fn,
            kl_weight=self.kl_weight, activation='relu',
            name='bayesian_tree_high_output'
        )

        self.chl_output = tfp.layers.DenseVariational(
            1, make_posterior_fn=self._make_simple_posterior_fn,
            make_prior_fn=self._make_simple_prior_fn,
            kl_weight=self.kl_weight, activation='relu',
            name='bayesian_chl_output'
        )

        self.lma_output = tfp.layers.DenseVariational(
            1, make_posterior_fn=self._make_simple_posterior_fn,
            make_prior_fn=self._make_simple_prior_fn,
            kl_weight=self.kl_weight, activation='relu',
            name='bayesian_lma_output'
        )

    def _make_simple_posterior_fn(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-4 + 0.1 * tf.nn.softplus(t[..., n:])),
                reinterpreted_batch_ndims=1))
        ])

    def _make_simple_prior_fn(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=0.1),
                reinterpreted_batch_ndims=1))
        ])

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)

        return {
            'carotenoids': self.carotenoids_output(x),
            'tree_high': self.tree_high_output(x),
            'chl': self.chl_output(x),
            'lma': self.lma_output(x)
        }

    def predict_with_uncertainty(self, inputs, num_samples=100):
        predictions = {'carotenoids': [], 'tree_high': [], 'chl': [], 'lma': []}

        for _ in range(num_samples):
            pred_dict = self(inputs, training=True)
            for key in predictions:
                predictions[key].append(pred_dict[key].numpy())

        results = {}
        for key in predictions:
            preds_array = np.array(predictions[key])
            results[key] = {
                'mean': np.mean(preds_array, axis=0),
                'std': np.std(preds_array, axis=0)
            }

        return results


# ==================== 2. SimpleBayesianMSELoss ====================
class SimpleBayesianMSELoss:
    def __init__(self, lambda_accuracy=0.8, lambda_auxiliary=0.1, lambda_boundary=0.1,
                 num_train_samples=1, kl_weight_schedule=True):
        self.lambda_accuracy = lambda_accuracy
        self.lambda_auxiliary = lambda_auxiliary
        self.lambda_boundary = lambda_boundary
        self.num_train_samples = num_train_samples
        self.kl_weight_schedule = kl_weight_schedule
        self.current_epoch = 0

    def accuracy_loss(self, y_true, y_pred):
        huber_loss = tf.keras.losses.Huber(delta=0.1)
        return huber_loss(y_true, y_pred)

    def auxiliary_loss(self, tree_high_true, tree_high_pred,
                       chl_true, chl_pred, lma_true, lma_pred):
        tree_loss = tf.reduce_mean(tf.square(tree_high_true - tree_high_pred))
        chl_loss = tf.reduce_mean(tf.square(chl_true - chl_pred))
        lma_loss = tf.reduce_mean(tf.square(lma_true - lma_pred))
        return tree_loss + chl_loss + lma_loss

    def boundary_loss(self, y_pred):
        car_lower, car_upper = 0.05, 0.5
        car_violation = (tf.maximum(car_lower - y_pred, 0.0) +
                         tf.maximum(y_pred - car_upper, 0.0))
        return tf.reduce_mean(tf.square(car_violation))

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def total_loss(self, targets, predictions, model):
        acc_loss = self.accuracy_loss(targets['carotenoids'], predictions['carotenoids'])
        aux_loss = self.auxiliary_loss(
            targets['tree_high'], predictions['tree_high'],
            targets['chl'], predictions['chl'],
            targets['lma'], predictions['lma']
        )
        b_loss = self.boundary_loss(predictions['carotenoids'])
        kl_loss = sum(model.losses) / self.num_train_samples if model.losses else 0.0

        if self.kl_weight_schedule:
            kl_weight_factor = min(1.0, max(0.0, (self.current_epoch - 50) / 100.0))
            effective_kl_loss = kl_loss * kl_weight_factor
        else:
            effective_kl_loss = kl_loss

        total_loss_val = (self.lambda_accuracy * acc_loss +
                          self.lambda_auxiliary * aux_loss +
                          self.lambda_boundary * b_loss +
                          effective_kl_loss)

        return total_loss_val, {
            'mse_loss': acc_loss,
            'auxiliary_loss': aux_loss,
            'boundary_loss': b_loss,
            'kl_loss': kl_loss,
            'effective_kl_loss': effective_kl_loss,
            'total_loss': total_loss_val
        }


# ==================== 3. SimpleBayesianTrainer ====================
class SimpleBayesianTrainer:
    def __init__(self, model, loss_fn, optimizer_config):
        self.model = model
        self.loss_fn = loss_fn

        optimizer_type = optimizer_config['type']
        learning_rate = optimizer_config['learning_rate']

        if optimizer_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=optimizer_config.get('beta_1', 0.9),
                beta_2=optimizer_config.get('beta_2', 0.999)
            )
        elif optimizer_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=optimizer_config.get('momentum', 0.0)
            )
        elif optimizer_type == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.train_history = []

        self.best_predictions = None
        self.best_test_r2 = -np.inf

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            total_loss, loss_components = self.loss_fn.total_loss(targets, predictions, self.model)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        metrics = calculate_metrics(targets['carotenoids'], predictions['carotenoids'])
        return total_loss, loss_components, metrics

    def _save_best_predictions(self, train_dataset, test_dataset):
        """epoch"""
        # training prediction
        train_true, train_pred = [], []
        for batch in train_dataset:
            inputs, targets = batch
            predictions = self.model(inputs, training=False)
            train_true.extend(targets['carotenoids'].numpy().flatten())
            train_pred.extend(predictions['carotenoids'].numpy().flatten())

        # testing prediction
        test_true, test_pred = [], []
        for batch in test_dataset:
            inputs, targets = batch
            predictions = self.model(inputs, training=False)
            test_true.extend(targets['carotenoids'].numpy().flatten())
            test_pred.extend(predictions['carotenoids'].numpy().flatten())

        self.best_predictions = {
            'train_true': np.array(train_true),
            'train_pred': np.array(train_pred),
            'test_true': np.array(test_true),
            'test_pred': np.array(test_pred)
        }

    def train_with_detailed_progress(self, train_dataset, test_dataset, epochs=1000, show_detailed=False):
        if show_detailed:
            print("\n  【GPU auxiliary】")
            print("  10batch time...")
            import time
            start = time.time()
            for i, batch in enumerate(train_dataset.take(10)):
                inputs, targets = batch
                loss, _, _ = self.train_step(inputs, targets)
                if i == 0:
                    first_batch_time = time.time() - start
                    print(f"  use: {first_batch_time:.2f}s")
                    start = time.time()
            avg_time = (time.time() - start) / 9
            print(f"  {avg_time:.3f}s")
            if first_batch_time > avg_time * 3:
                print(f" {first_batch_time / avg_time:.1f}x")
            else:
                print(f"pass")
            print()

        if show_detailed:
            print(f"  process ({epochs}epochs):")
            print("  " + "-" * 100)
            print(
                f"  {'Epoch':>6} | {'Total':>8} | {'MSE':>8} | {'Aux':>8} | {'Bound':>8} | {'KL':>8} | {'EffKL':>8} | {'Train_R²':>8} | {'Test_R²':>8} | {'Best_R²':>8}")
            print("  " + "-" * 100)

        for epoch in range(epochs):
            self.loss_fn.update_epoch(epoch)
            epoch_losses = []
            epoch_loss_components = {
                'total_loss': [], 'mse_loss': [], 'auxiliary_loss': [],
                'boundary_loss': [], 'kl_loss': [], 'effective_kl_loss': []
            }
            epoch_metrics = {'r2': [], 'rmse': [], 'mae': []}

            for batch in train_dataset:
                inputs, targets = batch
                loss, components, metrics = self.train_step(inputs, targets)
                epoch_losses.append(loss.numpy())

                for key in epoch_loss_components:
                    if key in components:
                        epoch_loss_components[key].append(components[key].numpy())

                for key in epoch_metrics:
                    if not (np.isnan(metrics[key].numpy()) or np.isinf(metrics[key].numpy())):
                        epoch_metrics[key].append(metrics[key].numpy())

            train_loss = np.mean(epoch_losses)
            avg_mse_loss = np.mean(epoch_loss_components['mse_loss']) if epoch_loss_components['mse_loss'] else 0.0
            avg_aux_loss = np.mean(epoch_loss_components['auxiliary_loss']) if epoch_loss_components[
                'auxiliary_loss'] else 0.0
            avg_bound_loss = np.mean(epoch_loss_components['boundary_loss']) if epoch_loss_components[
                'boundary_loss'] else 0.0
            avg_kl_loss = np.mean(epoch_loss_components['kl_loss']) if epoch_loss_components['kl_loss'] else 0.0
            avg_eff_kl_loss = np.mean(epoch_loss_components['effective_kl_loss']) if epoch_loss_components[
                'effective_kl_loss'] else 0.0

            train_r2 = np.mean(epoch_metrics['r2']) if epoch_metrics['r2'] else -np.inf

            test_metrics = self.evaluate(test_dataset)
            test_r2 = test_metrics['r2']

            if test_r2 > self.best_test_r2:
                self.best_test_r2 = test_r2
                #  best epoch to save results
                self._save_best_predictions(train_dataset, test_dataset)

            if show_detailed:
                if epoch < 5 or epoch % 10 == 0 or epoch == epochs - 1:
                    print(
                        f"  {epoch + 1:6d} | {train_loss:8.5f} | {avg_mse_loss:8.5f} | {avg_aux_loss:8.5f} | {avg_bound_loss:8.5f} | {avg_kl_loss:8.5f} | {avg_eff_kl_loss:8.5f} | {train_r2:8.4f} | {test_r2:8.4f} | {self.best_test_r2:8.4f}")
                elif epoch % 5 == 0:
                    print(f"  {epoch + 1:6d} |  (best R²={self.best_test_r2:.4f})")
            else:
                progress_points = [int(epochs * 0.25) - 1, int(epochs * 0.5) - 1, int(epochs * 0.75) - 1, epochs - 1]
                if epoch in progress_points:
                    progress = int((epoch + 1) / epochs * 100)
                    print(f"{progress}%", end="...", flush=True)

        if show_detailed:
            print("  " + "-" * 100)
            print(f"  finish！bestR²: {self.best_test_r2:.6f}")

        return self.best_test_r2, self.model, self.best_predictions

    def evaluate(self, dataset):
        all_metrics = {'r2': [], 'rmse': [], 'mae': []}

        for batch in dataset:
            inputs, targets = batch
            predictions = self.model(inputs, training=False)
            batch_metrics = calculate_metrics(targets['carotenoids'], predictions['carotenoids'])

            for key in all_metrics:
                if not (np.isnan(batch_metrics[key].numpy()) or np.isinf(batch_metrics[key].numpy())):
                    all_metrics[key].append(batch_metrics[key].numpy())

        return {key: np.mean(values) if values else 0.0 for key, values in all_metrics.items()}


# ==================== 4. HyperparameterSearch ====================
class HyperparameterSearch:
    def __init__(self, train_dataset, test_dataset, input_dim):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.input_dim = input_dim
        self.results = []

    def define_search_space(self):
        search_space = {
            'hidden_units': [[32, 16], [64, 32], [128, 64], [64, 32, 16], [128, 64, 32]],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3],
            'activation': ['relu', 'tanh', 'swish'],
            'use_batch_norm': [False, True],
            'kl_weight': [1e-6, 1e-5, 1e-4],
            'optimizer_config': [
                {'type': 'adam', 'learning_rate': 1e-4},
                {'type': 'adam', 'learning_rate': 5e-4},
                {'type': 'adam', 'learning_rate': 1e-3},
                {'type': 'adam', 'learning_rate': 5e-3},
                {'type': 'sgd', 'learning_rate': 1e-3, 'momentum': 0.9},
                {'type': 'rmsprop', 'learning_rate': 1e-3},
            ],
            'epochs': [2000, 2500, 3000],
            'batch_size': [8, 16, 32]
        }
        return search_space

    def _process_epoch_predictions(self, epoch_predictions, model, train_dataset, test_dataset):
        """Uncertainty estimation"""


        train_true = epoch_predictions['train_true']
        train_pred_mean = epoch_predictions['train_pred']
        test_true = epoch_predictions['test_true']
        test_pred_mean = epoch_predictions['test_pred']


        print("\n", end="", flush=True)

        train_pred_std = []
        for batch in train_dataset:
            inputs, _ = batch
            uncertainty = model.predict_with_uncertainty(inputs, num_samples=100)
            train_pred_std.extend(uncertainty['carotenoids']['std'].flatten())
        train_pred_std = np.array(train_pred_std)

        test_pred_std = []
        for batch in test_dataset:
            inputs, _ = batch
            uncertainty = model.predict_with_uncertainty(inputs, num_samples=100)
            test_pred_std.extend(uncertainty['carotenoids']['std'].flatten())
        test_pred_std = np.array(test_pred_std)

        # metric
        train_r2 = r2_score(train_true, train_pred_mean)
        train_rmse = np.sqrt(np.mean((train_true - train_pred_mean) ** 2))
        train_mae = np.mean(np.abs(train_true - train_pred_mean))

        test_r2 = r2_score(test_true, test_pred_mean)
        test_rmse = np.sqrt(np.mean((test_true - test_pred_mean) ** 2))
        test_mae = np.mean(np.abs(test_true - test_pred_mean))

        # Coverage Rate (CR
        # 95%interval
        train_lower = train_pred_mean - 1.96 * train_pred_std
        train_upper = train_pred_mean + 1.96 * train_pred_std
        train_within_ci = ((train_true >= train_lower) & (train_true <= train_upper))
        train_cr = np.mean(train_within_ci)

        test_lower = test_pred_mean - 1.96 * test_pred_std
        test_upper = test_pred_mean + 1.96 * test_pred_std
        test_within_ci = ((test_true >= test_lower) & (test_true <= test_upper))
        test_cr = np.mean(test_within_ci)

        # DataFrame
        train_results_df = pd.DataFrame({
            'Sample_ID': range(1, len(train_true) + 1),
            'True_Carotenoids': train_true,
            'Predicted_Mean': train_pred_mean,
            'Predicted_Std': train_pred_std,
            'Confidence_Interval_Lower': train_lower,
            'Confidence_Interval_Upper': train_upper,
            'Within_CI': train_within_ci.astype(int),
            'Absolute_Error': np.abs(train_pred_mean - train_true),
            'Relative_Error_%': ((train_pred_mean - train_true) / train_true) * 100,
            'Dataset': 'Train'
        })

        test_results_df = pd.DataFrame({
            'Sample_ID': range(1, len(test_true) + 1),
            'True_Carotenoids': test_true,
            'Predicted_Mean': test_pred_mean,
            'Predicted_Std': test_pred_std,
            'Confidence_Interval_Lower': test_lower,
            'Confidence_Interval_Upper': test_upper,
            'Within_CI': test_within_ci.astype(int),
            'Absolute_Error': np.abs(test_pred_mean - test_true),
            'Relative_Error_%': ((test_pred_mean - test_true) / test_true) * 100,
            'Dataset': 'Test'
        })

        return {
            'train_r2': train_r2, 'train_rmse': train_rmse, 'train_mae': train_mae,
            'train_cr': train_cr,
            'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mae': test_mae,
            'test_cr': test_cr,
            'train_results_df': train_results_df,
            'test_results_df': test_results_df,
            'train_pred_std': train_pred_std,
            'test_pred_std': test_pred_std
        }

    def random_search(self, n_trials=30, seed=42):
        import time
        np.random.seed(seed)
        search_space = self.define_search_space()

        print(f" ({n_trials} test)")
        print("=" * 80)
        print("")
        print("-Carotenoids, Tree_high, Chl, LMA")
        print("- loss")
        print("- Carotenoids prediction")
        print("=" * 80)

        best_r2 = -np.inf
        best_config = None
        best_predictions = None

        for trial in range(n_trials):
            print(f"\n【Trial {trial + 1}/{n_trials} 】")

            start_time = time.time()

            config = {}
            config['hidden_units'] = search_space['hidden_units'][
                np.random.randint(0, len(search_space['hidden_units']))]
            config['dropout_rate'] = np.random.choice(search_space['dropout_rate'])
            config['activation'] = np.random.choice(search_space['activation'])
            config['use_batch_norm'] = np.random.choice(search_space['use_batch_norm'])
            config['kl_weight'] = np.random.choice(search_space['kl_weight'])
            config['optimizer_config'] = search_space['optimizer_config'][
                np.random.randint(0, len(search_space['optimizer_config']))]
            config['epochs'] = np.random.choice(search_space['epochs'])
            config['batch_size'] = np.random.choice(search_space['batch_size'])

            print(f"")
            print(f"  - {config['hidden_units']}")
            print(f"  - Dropout: {config['dropout_rate']}")
            print(f"  - {config['activation']}")
            print(f"  - {config['use_batch_norm']}")
            print(f"  - KL {config['kl_weight']}")
            print(f"  - {config['optimizer_config']['type']}")
            print(f"  - {config['optimizer_config']['learning_rate']}")
            print(f"  - {config['epochs']}")
            print(f"  - {config['batch_size']}")

            try:
                current_train_dataset = self.train_dataset.unbatch().batch(config['batch_size']).prefetch(
                    tf.data.AUTOTUNE)
                current_test_dataset = self.test_dataset.unbatch().batch(config['batch_size']).prefetch(
                    tf.data.AUTOTUNE)

                print(f"\n")

                model = SimpleBayesianPINNModel(
                    input_dim=self.input_dim,
                    hidden_units=config['hidden_units'],
                    dropout_rate=config['dropout_rate'],
                    activation=config['activation'],
                    use_batch_norm=config['use_batch_norm'],
                    kl_weight=config['kl_weight']
                )

                train_samples = sum(1 for _ in current_train_dataset) * config['batch_size']
                loss_fn = SimpleBayesianMSELoss(num_train_samples=train_samples)
                trainer = SimpleBayesianTrainer(model, loss_fn, config['optimizer_config'])

                print(f"\n")

                tf.random.set_seed(42)

                final_test_r2, trained_model, epoch_predictions = trainer.train_with_detailed_progress(
                    current_train_dataset,
                    current_test_dataset,
                    epochs=config['epochs'],
                    show_detailed=True
                )

                elapsed_time = time.time() - start_time

                result = {
                    'trial': trial + 1,
                    'config': config.copy(),
                    'test_r2': final_test_r2,
                    'training_time': elapsed_time
                }
                self.results.append(result)

                if final_test_r2 > best_r2:
                    best_r2 = final_test_r2
                    best_config = config.copy()

                    print(f"\n【Trial {trial + 1} 】")
                    print(f"  R²: {final_test_r2:.6f}")
                    print(f" : {elapsed_time:.1f}")
                    print(f"  ...", end="", flush=True)


                    best_predictions = self._process_epoch_predictions(
                        epoch_predictions,
                        trained_model,
                        current_train_dataset,
                        current_test_dataset
                    )
                    print(f" ✓")
                else:
                    print(f"\n【Trial {trial + 1} 】")
                    print(f"  R²: {final_test_r2:.6f}")
                    print(f"   {elapsed_time:.1f}")
                    print(f"  {best_r2:.6f}")

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n【Trial {trial + 1} 】")
                print(f"   {str(e)}")
                print(f"   {elapsed_time:.1f}")
                continue

        return best_config, best_r2, best_predictions

    def save_results(self, best_config, best_r2, species_name):
        results_data = []
        for result in self.results:
            row = {
                'trial': result['trial'],
                'test_r2': result['test_r2'],
                'hidden_units': str(result['config']['hidden_units']),
                'dropout_rate': result['config']['dropout_rate'],
                'activation': result['config']['activation'],
                'use_batch_norm': result['config']['use_batch_norm'],
                'kl_weight': result['config']['kl_weight'],
                'optimizer_type': result['config']['optimizer_config']['type'],
                'learning_rate': result['config']['optimizer_config']['learning_rate'],
                'epochs': result['config']['epochs'],
                'batch_size': result['config']['batch_size']
            }
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('test_r2', ascending=False)

        best_config_df = pd.DataFrame([
            ['Best_R2', best_r2],
            ['Hidden_Units', str(best_config['hidden_units'])],
            ['Dropout_Rate', best_config['dropout_rate']],
            ['Activation', best_config['activation']],
            ['Use_Batch_Norm', best_config['use_batch_norm']],
            ['KL_Weight', best_config['kl_weight']],
            ['Optimizer_Type', best_config['optimizer_config']['type']],
            ['Learning_Rate', best_config['optimizer_config']['learning_rate']],
            ['Epochs', best_config['epochs']],
            ['Batch_Size', best_config['batch_size']]
        ], columns=['Parameter', 'Value'])

        #
        folder_path = f"D:/1_SYJ-Carotenoides/VISSA-MCMR/{species_name}"
        save_path = os.path.join(folder_path, f"{species_name}_MultiTask_Hyperparameter_Search_Results.xlsx")

        try:

            os.makedirs(folder_path, exist_ok=True)

            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='All_Results', index=False)
                best_config_df.to_excel(writer, sheet_name='Best_Config', index=False)

            print(f"\n {save_path}")
        except Exception as e:
            print(f"{e}")

        return results_df

    def save_detailed_predictions(self, predictions, best_config, species_name):

        folder_path = f"D:/1_SYJ-Carotenoides/VISSA-MCMR/{species_name}"
        save_path = os.path.join(folder_path, f"{species_name}_MultiTask_Final_Results.xlsx")

        try:


            train_cr = predictions['train_cr']
            test_cr = predictions['test_cr']
            test_true = predictions['test_results_df']['True_Carotenoids'].values
            test_pred_mean = predictions['test_results_df']['Predicted_Mean'].values
            test_pred_std = predictions['test_results_df']['Predicted_Std'].values

            within_ci = ((test_true >= test_pred_mean - 1.96 * test_pred_std) &
                         (test_true <= test_pred_mean + 1.96 * test_pred_std))
            coverage_rate = np.mean(within_ci) * 100

            performance_df = pd.DataFrame({
                'Dataset': ['Training', 'Testing'],
                'Sample_Count': [len(predictions['train_results_df']), len(predictions['test_results_df'])],
                'R2': [predictions['train_r2'], predictions['test_r2']],
                'RMSE': [predictions['train_rmse'], predictions['test_rmse']],
                'MAE': [predictions['train_mae'], predictions['test_mae']],
                'Coverage_Rate': [train_cr, test_cr],
                'Mean_Uncertainty': [predictions['train_pred_std'].mean(), predictions['test_pred_std'].mean()],
                'Std_Uncertainty': [predictions['train_pred_std'].std(), predictions['test_pred_std'].std()]
            })

            config_df = pd.DataFrame([
                ['Test_R2', predictions['test_r2']],
                ['Train_R2', predictions['train_r2']],
                ['Test_Coverage_Rate', test_cr],
                ['Train_Coverage_Rate', train_cr],
                ['CI_Coverage_Rate_%', coverage_rate],
                ['Hidden_Units', str(best_config['hidden_units'])],
                ['Dropout_Rate', best_config['dropout_rate']],
                ['Activation', best_config['activation']],
                ['Use_Batch_Norm', best_config['use_batch_norm']],
                ['KL_Weight', best_config['kl_weight']],
                ['Optimizer_Type', best_config['optimizer_config']['type']],
                ['Learning_Rate', best_config['optimizer_config']['learning_rate']],
                ['Epochs', best_config['epochs']],
                ['Batch_Size', best_config['batch_size']]
            ], columns=['Parameter', 'Value'])

            os.makedirs(folder_path, exist_ok=True)

            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                predictions['train_results_df'].to_excel(writer, sheet_name='Train_Predictions', index=False)
                predictions['test_results_df'].to_excel(writer, sheet_name='Test_Predictions', index=False)
                performance_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)

            print(f"\n {save_path}")


            if abs(test_cr - 0.95) < 0.05:
                print(f"- ✓ ")
            elif test_cr < 0.90:
                print(f"- ⚠ ")
            else:
                print(f"- ⚠")

        except Exception as e:
            print(f" {e}")

@tf.function
def calculate_r2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2

@tf.function
def calculate_metrics(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    r2 = calculate_r2(y_true, y_pred)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def prepare_data(file_path, test_size=0.3, random_state=42):
    data = pd.read_excel(file_path)

    spectral_cols = [col for col in data.columns
                     if col not in ['Carotenoids', 'tree_high', 'Chl', 'LMA']]

    X_spectral = data[spectral_cols].values.astype(np.float32)
    X_physical = data[['tree_high', 'Chl', 'LMA']].values.astype(np.float32)
    X_combined = np.concatenate([X_spectral, X_physical], axis=1)

    y_carotenoid = data['Carotenoids'].values.astype(np.float32)
    y_tree_high = data['tree_high'].values.astype(np.float32)
    y_chl = data['Chl'].values.astype(np.float32)
    y_lma = data['LMA'].values.astype(np.float32)


    X_train, X_test, y_car_train, y_car_test, y_tree_train, y_tree_test, \
    y_chl_train, y_chl_test, y_lma_train, y_lma_test = train_test_split(
        X_combined, y_carotenoid, y_tree_high, y_chl, y_lma,
        test_size=test_size, random_state=random_state)

    def create_dataset(X, y_car, y_tree, y_chl, y_lma, batch_size=16):
        targets = {
            'carotenoids': y_car.reshape(-1, 1),
            'tree_high': y_tree.reshape(-1, 1),
            'chl': y_chl.reshape(-1, 1),
            'lma': y_lma.reshape(-1, 1)
        }
        dataset = tf.data.Dataset.from_tensor_slices((X, targets))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = create_dataset(X_train, y_car_train, y_tree_train, y_chl_train, y_lma_train)
    test_dataset = create_dataset(X_test, y_car_test, y_tree_test, y_chl_test, y_lma_test)


    return train_dataset, test_dataset, X_combined.shape[1], X_train, X_test, y_car_train, y_car_test


# ==================== 7. main====================
def main_hyperparameter_optimization(file_path, species_name="B.sexangular"):
    print("=" * 80)

    train_dataset, test_dataset, input_dim, X_train, X_test, y_train, y_test = prepare_data(
        file_path, random_state=42
    )

    print(f"\n")
    print(f"-  {input_dim}")
    print(f"-  {len(X_train)}")
    print(f"-  {len(X_test)}")

    search_engine = HyperparameterSearch(train_dataset, test_dataset, input_dim)
    best_config, best_r2, best_predictions = search_engine.random_search(n_trials=10, seed=42)

    print(f" {best_r2:.6f}")

    search_engine.save_results(best_config, best_r2, species_name)

    if best_predictions:
        search_engine.save_detailed_predictions(best_predictions, best_config, species_name)


    print(f"R²: {best_r2:.6f}")

    return {
        'best_config': best_config,
        'best_r2': best_r2,
        'predictions': best_predictions
    }


# ==================== example ====================
if __name__ == "__main__":

    species_name = "Allspecies-S2"
    # A.corniculatum, A.ilicifolius, A.marina, B.gymonrrhiza, B.sexangular, K.candel, R.stylosa, Allspecies
    # file_path = f"D:/1_SYJ-Carotenoides/VISSA-MCMR/{species_name}_vissa_optimal_correlation_features.xlsx"


    optimization_results = main_hyperparameter_optimization(file_path, species_name)

    print(f":{optimization_results['best_config']['hidden_units']}")
    print(f"Dropout: {optimization_results['best_config']['dropout_rate']}")
    print(f" {optimization_results['best_config']['activation']}")
    print(f" {optimization_results['best_config']['use_batch_norm']}")
    print(f"KL{optimization_results['best_config']['kl_weight']}")
    print(f"{optimization_results['best_config']['optimizer_config']['learning_rate']}")
    print(f"{optimization_results['best_config']['batch_size']}")
    print(f" {optimization_results['best_config']['epochs']}")
    print(f"\n {optimization_results['best_r2']:.6f}")


    if optimization_results['predictions']:
        predictions = optimization_results['predictions']
        print(f"\n：")
        print(f"R²: {predictions['test_r2']:.6f}")
        print(f"RMSE: {predictions['test_rmse']:.6f}")
        print(f"Coverage Rate: {predictions['test_cr']:.4f} ({predictions['test_cr'] * 100:.2f}%)")
        print(f": {predictions['test_pred_std'].mean():.6f}")

    print(f"\n")
    print(f"- (Carotenoids, Tree_high, Chl, LMA)")