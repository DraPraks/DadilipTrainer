### README.md

# ModelTrainer: A Python Library for Fine-Tuning Pre-Trained Models

`ModelTrainer` is a Python library for fine-tuning pre-trained deep learning models (e.g., DenseNet121, ResNet50, VGG16, etc.) using TensorFlow. It simplifies the process of transfer learning, enabling users to customize regularization, dropout, and learning rate scheduling, with support for binary and multiclass classification tasks.

---

## Features

- Fine-tune popular pre-trained models.
- Customize regularization (L1/L2) and dropout.
- Support for learning rate schedulers.
- Handles large datasets with `tf.data` pipelines.
- Mixed-precision training for faster computation.

---

## Installation

1. Clone the repository or copy the library files:
   ```bash
   git clone https://github.com/yourusername/model_trainer.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
### OR

```pip install DadilipTrainer``` <== Is preferred

---

## Dataset Structure

The dataset directory should follow this structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── class2/
│       ├── image1.jpg
│       ├── image2.jpg
├── val/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── class2/
│       ├── image1.jpg
│       ├── image2.jpg
├── test/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── class2/
        ├── image1.jpg
        ├── image2.jpg
```

- Each subdirectory under `train/`, `val/`, and `test/` represents a class.
- Images in each subdirectory belong to the respective class.

---

## Quick Start Guide

### Example Usage

```python
from DadilipTrainer import ModelTrainer

# Initialize the trainer
trainer = ModelTrainer(
   save_dir='./models',
   pre_trained_model_name='DenseNet121',
   num_layers_to_train=20,
   train_dir='./dataset/train',
   val_dir='./dataset/val',
   test_dir='./dataset/test',
   batch_size=16,
   task_type='binary',  # Use 'multiclass' for multiple classes
   image_shape=(224, 224),
   normalization_method='rescale',  # Options: 'rescale', 'standardize'
   use_l1=True,
   use_l2=True,
   l1_rate=0.01,
   l2_rate=0.01,
   use_dropout=True,
   dropout_rate=0.5,
   use_lr_scheduler=True,
   initial_learning_rate=1e-4
)

# Load history if any
trainer.load_history()

# Build and compile the model
trainer.build_model()
trainer.compile_model()

# Prepare data
trainer.prepare_data()

# Train the model
trainer.train(epochs=50)

# Save and visualize history
trainer.save_history()
trainer.plot_history()
```

---

## Customization Options

### Pre-Trained Models
Supported models:
- DenseNet121
- ResNet50
- VGG16
- MobileNetV2
- InceptionV3

Set the model with `pre_trained_model_name`.

---

### Regularization and Dropout
- Enable L1/L2 regularization with `use_l1`, `use_l2`, `l1_rate`, and `l2_rate`.
- Use dropout with `use_dropout` and specify the rate with `dropout_rate`.

---

### Learning Rate Scheduling
Enable learning rate scheduling with `use_lr_scheduler`. Adjust the learning rate dynamically based on validation loss.

---

## Dataset Preparation

1. Organize your dataset into directories as shown in the **Dataset Structure** section.
2. Ensure all images are properly formatted and labeled.
3. Optionally, preprocess and clean the dataset to remove corrupted images.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub.

---

## License

### Creative Commons Attribution-NonCommercial 4.0 International License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).


---

## Author

- **Your Name**
- [GitHub](https://github.com/drapraks)
- [Email](mailto:prakosoadra@gmail.com)

---

Let me know if you'd like further adjustments!
