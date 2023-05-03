import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

# Define the generator network
def generator():
         # Define the input layer
  input_layer = tf.keras.layers.Input(shape=(256, 256, 3))

  # Define the first 7x7 convolutional layer
  layer = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(input_layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  layer = tf.keras.layers.ReLU()(layer)

  # Define the next 3x3 convolutional layer
  layer = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  layer = tf.keras.layers.ReLU()(layer)

  # Define the next 3x3 convolutional layer
  layer = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  layer = tf.keras.layers.ReLU()(layer)

  # Define the residual blocks
  for i in range(9):
    residual = layer

    layer = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(layer)
    layer = tfa.layers.InstanceNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(layer)
    layer = tfa.layers.InstanceNormalization()(layer)

    layer = tf.keras.layers.add([layer, residual])
  
  # Define the next 3x3 fractional-strided convolutional layer
  layer = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  layer = tf.keras.layers.ReLU()(layer)

  # Define the next 3x3 fractional-strided convolutional layer
  layer = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  layer = tf.keras.layers.ReLU()(layer)

  # Define the last 7x7 convolutional layer
  layer = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(layer)
  layer = tfa.layers.InstanceNormalization()(layer)
  output_layer = tf.keras.layers.Activation('tanh')(layer)

  # Define the model
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

  return model

def discriminator():
    input_layer = tf.keras.layers.Input(shape=(256, 256, 3))
    
    layer = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(input_layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    
    layer = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(layer)
    layer = tfa.layers.InstanceNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    
    layer = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(layer)
    layer = tfa.layers.InstanceNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    
    layer = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(layer)
    layer = tfa.layers.InstanceNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    
    layer = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid')(layer)
    output_layer = tf.keras.layers.Activation('tanh')(layer)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def adversarial_loss(y_true, y_pred):
    loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return loss

LAMBDA = 10

# Define the cycle-consistency loss function
def cycle_consistency_loss(y_true, y_pred):
    loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return LAMBDA * loss

# Define the identity loss function
def identity_loss(y_true, y_pred):
    loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return LAMBDA * loss

BUFFER_SIZE = 1000
BATCH_SIZE = 1 #pg 5
IMG_WIDTH = 256
IMG_HEIGHT = 256

dataset_path = 'cycle_gan/maps'
dataset, metadata = tfds.load(dataset_path, with_info=True, as_supervised=True)
split_string = dataset_path.split('/')
data_name = split_string[1]

train_A, train_B = dataset['trainA'], dataset['trainB']
test_A, test_B = dataset['testA'], dataset['testB']  

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def preprocess_image_train(image, label):
  image = normalize(image)  
  return image   

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

AUTOTUNE = tf.data.AUTOTUNE 

train_A = train_A.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_B = train_B.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_A = test_A.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_B = test_B.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)


# normalize image for instancenormalization capability
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image


# Create the generator and discriminator networks
generator_AB = generator() #G
generator_BA = generator() #F

discriminator_A = discriminator() #X
discriminator_B = discriminator() #Y

sample_A = next(iter(train_A))
sample_B = next(iter(train_B))

to_B_from_A = generator_AB(sample_A) 
to_A_from_B = generator_BA(sample_B) 

# Define the optimizers for the generator and discriminator networks
optimizer_G = Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_F = Adam(learning_rate=0.0002, beta_1=0.5)

discriminator_A_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_B_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

  
# Compile the discriminator networks
#discriminator_A.compile(loss='binary_crossentropy', optimizer=optimizer_G)
#discriminator_B.compile(loss='binary_crossentropy', optimizer=optimizer_F)

# Define the input and output placeholders for the generator networks
#input_A = Input(shape=(256, 256, 3))
#input_B = Input(shape=(256, 256, 3))
#output_A = generator_BA(input_B)
#output_B = generator_AB(input_A)

# Define the cycle-consistency loss
#cycle_loss_A = cycle_consistency_loss(input_A, generator_BA(generator_AB(input_A)))
#cycle_loss_B = cycle_consistency_loss(input_B, generator_AB(generator_BA(input_B)))
#cycle_loss = cycle_loss_A + cycle_loss_B

# Define the identity loss
#identity_loss_A = identity_loss(input_A, generator_AB(input_A))
#identity_loss_B = identity_loss(input_B, generator_BA(input_B))
#identity_loss = identity_loss_A + identity_loss_B

#adversarial_loss_A = adversarial_loss(tf.ones_like(discriminator_A(output_A)), discriminator_A(output_A))
#adversarial_loss_B = adversarial_loss(tf.ones_like(discriminator_B(output_B)), discriminator_B(output_B))
#adversarial_loss = adversarial_loss_A + adversarial_loss_B

#generator_loss = cycle_loss + identity_loss + adversarial_loss

@tf.function
def train_step(input_A, input_B):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through the generators
        fake_y = generator_AB(input_A, training=True) 
        cycled_x = generator_BA(fake_y, training=True) 
        
        fake_x = generator_BA(input_B, training=True)
        cycled_y = generator_AB(fake_x, training=True)
        
    # Forward pass through the discriminators
        same_x = generator_BA(input_A, training=True)
        same_y = generator_AB(input_B, training=True)
        
        real_output_A = discriminator_A(input_A, training=True)
        real_output_B = discriminator_B(input_B, training=True)
        
        fake_output_A = discriminator_A(fake_x, training=True)
        fake_output_B = discriminator_B(fake_y, training=True)
    
        #calculate the loss
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gen_loss_g = loss(tf.ones_like(fake_output_B), fake_output_B)
        gen_loss_f = loss(tf.ones_like(fake_output_A), fake_output_A)
        
       
        # Define the cycle-consistency loss function
        def cycle_consistency_loss(y_true, y_pred):
            loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
            return LAMBDA * loss
        
        total_cycle_loss = cycle_consistency_loss(input_A, cycled_x) + cycle_consistency_loss(input_B, cycled_y)
        
        #generator loss 
        identity_loss_A = identity_loss(input_A, generator_AB(input_A))
        identity_loss_B = identity_loss(input_B, generator_BA(input_B))
        
        total_gen_loss_g = gen_loss_g + total_cycle_loss + identity_loss_B
        total_gen_loss_f = gen_loss_f + total_cycle_loss + identity_loss_A
        
        def discriminator_loss(real, generated):
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            real_loss = loss_obj(tf.ones_like(real), real)
            generated_loss = loss_obj(tf.zeros_like(generated), generated)
            total_disc_loss = real_loss + generated_loss
            return total_disc_loss * 0.5
        
        disc_A_loss = discriminator_loss(real_output_A, fake_output_A)
        disc_B_loss = discriminator_loss(real_output_B, fake_output_B)
        
    generator_g_gradients = tape.gradient(total_gen_loss_g, generator_AB.trainable_variables)
    
    generator_f_gradients = tape.gradient(total_gen_loss_f, generator_BA.trainable_variables)
    
    discriminator_A_gradients = tape.gradient(disc_A_loss, discriminator_A.trainable_variables)

    discriminator_B_gradients = tape.gradient(disc_B_loss, discriminator_B.trainable_variables)
    
    optimizer_G.apply_gradients(zip(generator_g_gradients, generator_AB.trainable_variables))
    optimizer_F.apply_gradients(zip(generator_f_gradients, generator_BA.trainable_variables))
    discriminator_A_optimizer.apply_gradients(zip(discriminator_A_gradients, discriminator_A.trainable_variables))
    discriminator_B_optimizer.apply_gradients(zip(discriminator_B_gradients, discriminator_B.trainable_variables))
    ########################################################3
        
    #    adversarial_loss_A = adversarial_loss(tf.ones_like(fake_output_A), fake_output_A)
    #    adversarial_loss_B = adversarial_loss(tf.ones_like(fake_output_B), fake_output_B)
    #    adversarial_loss = adversarial_loss_A + adversarial_loss_B
    
    # Compute the cycle-consistency loss
    #    cycle_loss_A = cycle_consistency_loss(input_A, generator_BA(generator_AB(input_A)))
    #    cycle_loss_B = cycle_consistency_loss(input_B, generator_AB(generator_BA(input_B)))
   #    cycle_loss = cycle_loss_A + cycle_loss_B
    
    # Compute the identity loss
    #    identity_loss_A = identity_loss(input_A, generator_AB(input_A))
    #    identity_loss_B = identity_loss(input_B, generator_BA(input_B))
    #    identity_loss = identity_loss_A + identity_loss_B
    
    # Compute the generator loss
    #    generator_loss = cycle_loss + identity_loss + adversarial_loss
    
    # Compute the gradients and update the generators
    
tf.keras.utils.plot_model(generator_AB, 'generatorAB.jpg', show_shapes = True, show_layer_names=False, dpi =64)     
tf.keras.utils.plot_model(discriminator_A, 'discriminatorA.jpg', show_shapes = True, show_layer_names=False, dpi =64)     

def generate_images(model, input, epoch, t, data_name):
  prediction = model(input)
    
  plt.figure(figsize=(10, 10))

  display_list = [input[0], prediction[0]]
  title = ['Input Image', 'Output Image']

  for a in range(2):
    plt.subplot(1, 2, a+1)
    plt.title(title[a])
    plt.imshow(display_list[a] * 0.5 + 0.5)
    plt.axis('off')
  filename = 'generated_image_' + str(epoch) + '_' + t + '_' + data_name + '.jpg'
  plt.savefig(filename)
  
epochs = 100
for epoch in range(epochs):
    for image_x, image_y in tf.data.Dataset.zip((train_A, train_B)):
        train_step(image_x,image_y)
        
    generate_images(generator_AB, sample_A, epoch, 'train', data_name)
    
epoch_test = 10
for e in range(epoch_test):
  for inp in test_A.take(5):
    generate_images(generator_AB, inp, e, 'testAB', data_name)
    
for e in range(epoch_test):
  for inp in test_B.take(5):
    generate_images(generator_BA, inp, e, 'testBA', data_name)
    
        


  


    