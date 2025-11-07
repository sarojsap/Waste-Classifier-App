import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

# Loads and prepares the training, validation, and test datasets
def create_datasets(data_dir):
    print("Creating training and validation datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123, # Using a seed ensures that the splits are reproducible
        image_size= (IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123, # Using a seed ensures that the splits are reproducible
        image_size= (IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Found Classes: {class_names}")

    print("Creating the test dataset...")
    val_batches = tf.data.experimental.cardinality(validation_ds)
    test_ds = validation_ds.take(val_batches // 2)
    validation_ds = validation_ds.skip(val_batches // 2)

    print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_ds)}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")

    # Define the data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])

    # --- Preprocessing and Performance Optimization ---
    # We define a function to apply augmentation and rescaling.
    # Rescaling normalizes pixel values from [0, 255] to [0, 1].

    rescale = tf.keras.layers.Rescaling(1./255)

    def prepare(ds, augment=False):
        ds = ds.map(lambda x,y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        # AUTOTUNE lets tf.data find the best performance settings dynamically.
        # .prefetch() overlaps data preprocessing and model execution

        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
     # Apply the preparation function to each dataset
    train_ds = prepare(train_ds, augment=True)
    validation_ds = prepare(validation_ds, augment=False)
    test_ds = prepare(test_ds, augment=False)

    return train_ds, validation_ds, test_ds, class_names