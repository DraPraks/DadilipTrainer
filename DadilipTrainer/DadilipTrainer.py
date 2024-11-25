# DadilipTrainer.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, VGG16, MobileNetV2, InceptionV3
)
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from PIL import ImageFile

class ModelTrainer:
    """
    A class for training computer vision models using transfer learning with TensorFlow and Keras.

    Attributes:
        save_dir (str): Directory to save models and training history.
        history_filename (str): Filename for saving training history.
        weights_filename (str): Filename for saving the best model weights.
        pre_trained_model_name (str): Name of the pre-trained model to use (e.g., DenseNet121).
        num_layers_to_train (int): Number of layers to unfreeze for fine-tuning.
        train_dir (str): Directory path for training data.
        val_dir (str): Directory path for validation data.
        test_dir (str): Directory path for test data.
        batch_size (int): Batch size for training.
        task_type (str): Type of classification task ('binary' or 'multiclass').
        image_shape (tuple): Desired image shape (height, width).
        normalization_method (str): Method to normalize data ('rescale', 'standardize', etc.).
        use_l1 (bool): Whether to use L1 regularization.
        use_l2 (bool): Whether to use L2 regularization.
        l1_rate (float): L1 regularization rate.
        l2_rate (float): L2 regularization rate.
        use_dropout (bool): Whether to include dropout layers.
        dropout_rate (float): Dropout rate.
        use_lr_scheduler (bool): Whether to use a learning rate scheduler.
        initial_learning_rate (float): Initial learning rate for the optimizer.
    """
    def __init__(self, save_dir, history_filename='training_history.json', weights_filename='best_model.keras',
                 pre_trained_model_name='DenseNet121', num_layers_to_train=20, train_dir=None, val_dir=None,
                 test_dir=None, batch_size=16, task_type='binary', image_shape=(224, 224),
                 normalization_method='rescale', use_l1=False, use_l2=True, l1_rate=0.01, l2_rate=0.01,
                 use_dropout=True, dropout_rate=0.5, use_lr_scheduler=True, initial_learning_rate=1e-4):
        """
        Initializes the ModelTrainer with necessary paths and configurations.

        Args:
            save_dir (str): Directory to save models and history.
            history_filename (str): Filename for saving training history.
            weights_filename (str): Filename for saving model weights.
            pre_trained_model_name (str): Name of the pre-trained model to use.
            num_layers_to_train (int): Number of layers to unfreeze for training.
            train_dir (str): Directory path for training data.
            val_dir (str): Directory path for validation data.
            test_dir (str): Directory path for test data.
            batch_size (int): Batch size for training.
            task_type (str): 'binary' or 'multiclass' classification task.
            image_shape (tuple): Desired image shape (height, width).
            normalization_method (str): Method to normalize data ('rescale', 'standardize', etc.).
            use_l1 (bool): Whether to use L1 regularization.
            use_l2 (bool): Whether to use L2 regularization.
            l1_rate (float): L1 regularization rate.
            l2_rate (float): L2 regularization rate.
            use_dropout (bool): Whether to include dropout layers.
            dropout_rate (float): Dropout rate.
            use_lr_scheduler (bool): Whether to use learning rate scheduler.
            initial_learning_rate (float): Initial learning rate for the optimizer.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.history_path = os.path.join(self.save_dir, history_filename)
        self.weights_path = os.path.join(self.save_dir, weights_filename)

        self.pre_trained_model_name = pre_trained_model_name
        self.num_layers_to_train = num_layers_to_train
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.task_type = task_type
        self.image_shape = image_shape
        self.normalization_method = normalization_method
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_lr_scheduler = use_lr_scheduler
        self.initial_learning_rate = initial_learning_rate

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Configure mixed precision
        self.configure_mixed_precision()

        # Initialize placeholders
        self.history = None
        self.model = None
        self.initial_epoch = 0
        self.num_classes = None  # Will be set in prepare_data()
        self.class_indices = None  # Will be set in prepare_data()
        self.train_data = None
        self.val_data = None
        self.class_weights = None

    def configure_mixed_precision(self):
        """
        Enables mixed precision training for faster computation.
        """
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")

    def load_history(self):
        """
        Loads previous training history from a JSON file.
        Sets the initial epoch based on the loaded history.
        """
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
            self.initial_epoch = len(self.history.get('loss', []))
            print(f"Loaded training history from {self.history_path}. Starting from epoch {self.initial_epoch}.")
        else:
            print(f"No existing training history found at {self.history_path}. Starting fresh.")
            self.history = {}
            self.initial_epoch = 0

    def build_model(self):
        """
        Builds the model architecture based on the selected pre-trained model.
        Unfreezes the last num_layers_to_train layers for fine-tuning.
        Applies regularization and dropout as specified.
        """
        input_shape = self.image_shape + (3,)
        inputs = Input(shape=input_shape)

        # Dictionary mapping model names to functions
        model_dict = {
            'DenseNet121': DenseNet121,
            'ResNet50': ResNet50,
            'VGG16': VGG16,
            'MobileNetV2': MobileNetV2,
            'InceptionV3': InceptionV3
        }

        if self.pre_trained_model_name not in model_dict:
            raise ValueError(f"Model {self.pre_trained_model_name} is not supported.")

        print(f"Building model using {self.pre_trained_model_name}")
        base_model = model_dict[self.pre_trained_model_name](
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )

        # Build the model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Apply dropout if specified
        if self.use_dropout:
            x = Dropout(self.dropout_rate)(x)

        # Set up regularization
        if self.use_l1 and self.use_l2:
            regularizer = l1_l2(l1=self.l1_rate, l2=self.l2_rate)
        elif self.use_l1:
            regularizer = l1(self.l1_rate)
        elif self.use_l2:
            regularizer = l2(self.l2_rate)
        else:
            regularizer = None

        x = Dense(128, activation='relu', kernel_regularizer=regularizer)(x)

        if self.use_dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)

        if self.use_dropout:
            x = Dropout(self.dropout_rate)(x)

        if self.task_type == 'binary':
            output_units = 1
            activation = 'sigmoid'
        else:
            if self.num_classes is None:
                raise ValueError("Number of classes is not set. Please run prepare_data() before build_model().")
            output_units = self.num_classes
            activation = 'softmax'

        output = Dense(output_units, activation=activation, kernel_regularizer=regularizer, dtype='float32')(x)
        self.model = Model(inputs=base_model.input, outputs=output)

        # Unfreeze the last num_layers_to_train layers
        self.model.trainable = True
        if self.num_layers_to_train > 0:
            for layer in self.model.layers[:-self.num_layers_to_train]:
                layer.trainable = False
        else:
            # Freeze all layers
            for layer in self.model.layers:
                layer.trainable = False

        num_trainable_layers = sum([1 for layer in self.model.layers if layer.trainable])
        print(f"Number of trainable layers: {num_trainable_layers}")
        print("Model architecture ready.")

    def compile_model(self):
        """
        Compiles the model with Adam optimizer and appropriate loss.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        if self.task_type == 'binary':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        print("Model compiled with Adam optimizer.")

    def load_weights(self):
        """
        Loads saved model weights if the weights file exists.
        """
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"Weights loaded successfully from {self.weights_path}.")
        else:
            print(f"No weights file found at {self.weights_path}. Skipping weight loading.")

    def setup_callbacks(self):
        """
        Sets up training callbacks including learning rate scheduler,
        early stopping, and model checkpointing.

        Returns:
            list: A list of callback instances.
        """
        callbacks = []

        if self.use_lr_scheduler:
            lr_schedule = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-6
            )
            callbacks.append(lr_schedule)
            print("Learning rate scheduler added.")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            self.weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        callbacks.extend([early_stopping, model_checkpoint])
        print("Callbacks set up successfully.")
        return callbacks

    def prepare_data(self):
        """
        Prepares the training, validation, and test data pipelines using
        ImageDataGenerator.

        Returns:
            tuple: Training and validation generators.
        """
        if not self.train_dir or not self.val_dir:
            raise ValueError("Training and validation directories must be specified.")

        # Data normalization and augmentation
        if self.normalization_method == 'rescale':
            preprocessing_function = None
            rescale = 1. / 255
        elif self.normalization_method == 'standardize':
            # Use the preprocess_input function specific to the model
            model_preprocessing_functions = {
                'DenseNet121': tf.keras.applications.densenet.preprocess_input,
                'ResNet50': tf.keras.applications.resnet.preprocess_input,
                'VGG16': tf.keras.applications.vgg16.preprocess_input,
                'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
                'InceptionV3': tf.keras.applications.inception_v3.preprocess_input
            }
            preprocessing_function = model_preprocessing_functions.get(self.pre_trained_model_name)
            rescale = None
        else:
            preprocessing_function = None
            rescale = None

        # Define data augmentation and preprocessing for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rescale=rescale,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=[0.9, 1.25],
            horizontal_flip=True,
            fill_mode='reflect',
            brightness_range=[0.5, 1.5]
        )

        # Preprocessing for validation
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rescale=rescale
        )

        # class_mode
        if self.task_type == 'binary':
            class_mode = 'binary'
        else:
            class_mode = 'categorical'

        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_shape,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=True
        )

        # Create validation generator
        validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_shape,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=False
        )

        # Number of classes and class indices
        self.num_classes = train_generator.num_classes
        self.class_indices = train_generator.class_indices

        # Calculate class weights to handle class imbalance
        class_weights_values = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = {i: class_weights_values[i] for i in range(len(class_weights_values))}
        self.class_weights = class_weights
        print("Class weights calculated:", class_weights)

        # Set the generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator

        print("Data generators prepared successfully.")
        return train_generator, validation_generator

    def train(self, epochs=50):
        """
        Trains the model using the prepared data and callbacks.
        Resumes training from the initial_epoch if history is loaded.

        Args:
            epochs (int): Total number of epochs for training.
        """
        if self.model is None:
            raise ValueError("Model is not built yet. Please call build_model() before training.")
        if self.train_generator is None or self.validation_generator is None:
            raise ValueError("Data generators are not prepared. Please call prepare_data() before training.")

        callbacks = self.setup_callbacks()

        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            validation_steps=self.validation_generator.samples // self.batch_size,
            class_weight=self.class_weights,
            initial_epoch=self.initial_epoch
        )

        print("Training completed.")

    def save_history(self):
        """
        Saves the training history to a JSON file.
        """
        if self.history:
            with open(self.history_path, 'w') as f:
                json.dump(self.history.history, f)
            print(f"Training history saved to {self.history_path}.")
        else:
            print("No training history to save.")

    def plot_history(self):
        """
        Plots the training and validation accuracy and loss.
        """
        if not self.history:
            print("No training history available to plot.")
            return

        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(14, 5))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.get('accuracy', []), 'bo-', label='Training Accuracy')
        plt.plot(epochs, history.get('val_accuracy', []), 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.get('loss', []), 'bo-', label='Training Loss')
        plt.plot(epochs, history.get('val_loss', []), 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        print("Training history plotted successfully.")
