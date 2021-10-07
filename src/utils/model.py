import tensorflow as tf

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28],name='inputlayer1'),
        tf.keras.layers.Dense(units=300, activation='relu', name='hiddenlayer1'),
        tf.keras.layers.Dense(units=100, activation='relu', name='hiddenlayer2'),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', name='outputlayer')
    ] 

    model_clf = tf.keras.models.Sequential(layers=LAYERS)

    model_clf.summary()

    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

    return model_clf ## <<< untrained model
    