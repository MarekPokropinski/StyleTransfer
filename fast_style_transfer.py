import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

content_path = tf.keras.utils.get_file('tubingen.jpg', 'https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg')
# style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# content_path = 'pic.jpg'
# style_path = tf.keras.utils.get_file('Large_bonfire.jpg','https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg')
# style_path = tf.keras.utils.get_file('Derkovits_Gyula_Woman_head_1922.jpg','https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg')
style_path = tf.keras.utils.get_file('Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg','https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg')


content_img = tf.keras.preprocessing.image.load_img(content_path)
style_img = tf.keras.preprocessing.image.load_img(style_path)

content_img = tf.image.resize(content_img, (512, 512), preserve_aspect_ratio=True)/255.5

# cut center of image to make it square
style_img = np.array(style_img)
style_img_size = min(style_img.shape[:2])
x = (style_img.shape[0]-style_img_size)//2
y = (style_img.shape[1]-style_img_size)//2
style_img = style_img[x:x+style_img_size, y:y+style_img_size]

style_img = tf.image.resize(style_img, (256, 256))/255.5

outputs = model(content_img[tf.newaxis, ...], style_img[tf.newaxis, ...])

plt.imshow(outputs[0][0])
plt.show()
