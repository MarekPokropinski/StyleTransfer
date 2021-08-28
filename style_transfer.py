import tensorflow as tf
from imageio import imwrite


feature_models = {
    'vgg19' : tf.keras.applications.VGG19
}

layers = {
    'vgg19' : {
        'content': ['block5_conv2'],
        'style': ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    }
}

preprocessing = {
    'vgg19' : tf.keras.applications.vgg19.preprocess_input
}

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)



class StyleTransfer:
    def __init__(self, feature_extraction='vgg19') -> None:
        self.feature_extraction = feature_extraction
        self.feature_model = self.model(feature_extraction)
        self.optim = tf.keras.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)

    def model(self, feature_extraction):
        feature_model = feature_models[feature_extraction](include_top=False, weights='imagenet')
        feature_model.trainable = False

        layer_names = layers[feature_extraction]['content'] + layers[feature_extraction]['style']

        outputs = [feature_model.get_layer(layer_name).output for layer_name in layer_names]

        return tf.keras.Model([feature_model.input], outputs)

    def set_image(self, content, style):
        self.image = tf.Variable(content)

        content = content[tf.newaxis, ...]
        content = preprocessing[self.feature_extraction](content)

        idx = len(layers[self.feature_extraction]['content'])

        content_outputs = self.feature_model(content)
        self.content_targets = content_outputs[:idx]

        style = style[tf.newaxis, ...]
        style = preprocessing[self.feature_extraction](style)


        style_outputs = self.feature_model(style)        
        self.style_targets = [gram_matrix(output) for output in style_outputs[idx:]]

    def step(self):
        idx = len(layers[self.feature_extraction]['content'])

        with tf.GradientTape() as tape:
            image = self.image[tf.newaxis, ...]
            image = tf.clip_by_value(image, 0.0, 255.0)
            preprocessed = preprocessing[self.feature_extraction](self.image[tf.newaxis, ...])
            outputs = self.feature_model(preprocessed)

            content_outputs = outputs[:idx]
            style_outputs = [gram_matrix(output) for output in outputs[idx:]]


            content_loss = tf.reduce_mean([tf.reduce_mean((content_output-content_target)**2) for content_output, content_target in zip(content_outputs, self.content_targets)])
            style_loss = tf.reduce_mean([tf.reduce_mean((style_output-style_target)**2) for style_output, style_target in zip(style_outputs, self.style_targets)])

            # loss = 1e-2*style_loss + 1e4*content_loss
            loss = 1e-2*style_loss + 1e4*content_loss

        grad = tape.gradient(loss, self.image)
        self.optim.apply_gradients([(grad, self.image)])

    def get_image(self):
        return tf.clip_by_value(self.image, 0.0, 255.0).numpy().astype('uint8')


# content_path = tf.keras.utils.get_file('Green_Sea_Turtle_grazing_seagrass.jpg', 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg')
content_path = tf.keras.utils.get_file('tubingen.jpg', 'https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg')
# style_path = tf.keras.utils.get_file('munch_scream.jpg','https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_img = tf.keras.preprocessing.image.load_img(content_path)
style_img = tf.keras.preprocessing.image.load_img(style_path)

content_img = tf.image.resize(content_img, (512, 512), preserve_aspect_ratio=True)
style_img = tf.image.resize(style_img, (512, 512), preserve_aspect_ratio=True)

style_transfer = StyleTransfer()
style_transfer.set_image(content_img, style_img)

for i in range(10):
    for j in range(100):
        style_transfer.step()
    
    imwrite(f'outputs/{i:03d}.png', style_transfer.get_image())
