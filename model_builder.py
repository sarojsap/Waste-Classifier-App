import tensorflow as tf

IMG_SIZE = 224

# Builds a classification model using transfer learning with MobileNetV2.
def build_model(num_classes):

    # Define the input layer
    # This specifies the shape of the images the model will accept
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Load the pretrained base model (MobileNetV2)
    # We use MobileNetV2 pre-trained on the ImageNet dataset.
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,  # we are NOT including the final classification layer from the original model. We will add our own.
        weights='imagenet'  # `weights='imagenet'` specifies which pre-trained weights to load.
    )

    # Freeze the base model
    base_model.trainable = False

    # Connect the base model to the input
    # we pass our i/p layer to the base model
    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # create the final model
    model = tf.keras.Model(inputs, outputs)

    print('Model Built Successfully')
    model.summary()

    return model