# Global Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

# Local Imports
import model
import input_data

NUM_EPOCH = 10000 + 1
BATCH_SIZE = 128
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

if __name__ == "__main__":

    # Global Step
    global_step = tf.Variable(0, trainable=False, dtype = tf.int32, name='global_step')

    # Placeholder for input images batch
    batch_input_images = tf.placeholder("float", [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='Input_Images')

    # Placeholder for colored images
    batch_images_color = tf.placeholder("float", [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='Colored_Images')

    # Placeholder for classes
    batch_classes = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='Classes')
    
    # Runnung the network model
    output_images, classification = model.network_model(batch_input_images)

    # Calculating the loss for current step
    total_loss = model.loss(output_images, batch_images_color, classification, batch_classes)

    # Performing training (backpropagation) by minimizing loss
    train_op = model.train(total_loss, global_step)

    # Object to save and restore model 
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Initializer all global variables in tensorflow
        sess.run(tf.global_variables_initializer())

        # Restore previous training if available
        checkpoint = tf.train.get_checkpoint_state("saved_checkpoint/")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # Increase Global Step Count
        global_step += 1
                
        # Evaluate the global step count
        step = global_step.eval(session=sess)
             
        for epoch_i in range(1,NUM_EPOCH+1):
            for batch_images_color_i, batch_images_bw_i, batch_classes_i in input_data.get_batches_for_train(BATCH_SIZE):

                # Run the Tensorflow Training operation
                sess.run(train_op, feed_dict={batch_input_images: batch_images_bw_i,
                                              batch_images_color: batch_images_color_i,
                                              batch_classes: batch_classes_i,
                                              global_step: step})

                # Evaluate, Print and Save the results at regular intervals of time
                if (step-1)%50 == 0:
                    loss_val = total_loss.eval(session=sess, feed_dict={batch_input_images: batch_images_bw,
                                                                        batch_images_color: batch_images_color_i,
                                                                        batch_classes: batch_classes_i})
                    
                    generated_images = output_images.eval(session=sess, feed_dict={batch_input_images: batch_images_bw_i})
                    identified_classes = classification.eval(session=sess, feed_dict={batch_input_images: batch_images_bw_i})
                    print('epoch: (%d/%d) , loss=%f' % (epoch_i,NUM_EPOCH+1,loss_val))
                    save_image_results(filename='epoch'+str(epoch_i)+'_step'+str(step)+'.jpg',
                                       input_images=batch_images_bw_i, output_images=generated_images,
                                       real_images=batch_images_color_i, num_images=2)
                    saver.save(sess,'saved_checkpoint/model.ckpt')

                # Increase Global Step Count
                global_step += 1


def test_model():
    return

def save_image_results(filename, input_images, output_images, real_images, num_images=1):
    """
    Saves the result images of the training. One image will have num_images rows and 3 columns
    (Input, Output, Truth)
    ARGS:
        filename : filename for saving the result image
        input_images : the batch of input images which were passed in the model (black and white pencil sketch)
        output_images : the batch of output images which were produced by the model (generated colored images)
        real_images : the batch of real images which are the actual results/ ground truth (colored images)
        num_images : (default=1) number of examples to show in the figure. Basically, number of rows.
    """
    
    fig = plt.figure()
    num_images = 10 if num_images > 10 else num_images
    indices = random.sample(range(len(input_images), num_images))
    
    
    nrows=num_images
    ncols=3
    plot_i = 1
    for i in indices:
        fig.add_subplot(nrows,ncols, plot_i).set_title('Input')
        plt.imshow(cv2.cvtColor(input_images[i], cv2.COLOR_LAB2BGR),'gray')
        plt.axis('off')
        plot_i += 1

        fig.add_subplot(nrows,ncols, plot_i).set_title('Output')
        plt.imshow(cv2.cvtColor(output_images[i], cv2.COLOR_LAB2BGR),'rgb')
        plt.axis('off')
        plot_i += 1

        fig.add_subplot(n_rows,ncols, plot_i).set_title('Truth')
        plt.imshow(cv2.cvtColor(real_images[i], cv2.COLOR_LAB2BGR), 'rgb')
        plt.axis('off')
        plot_i += 1

    fig.savefig('results/'+filename)   
