import tensorflow as tf
import os
import re



BATCH_SIZE = 128
IMAGE_SIZE = [244,244]
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.




def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.
    This is helpful when we train the model on multiple GPUs. Hence, all the
    common variables will be stored on CPU memory
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    #with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
 
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


# Network Model
def network_model(images):
    '''
    This is the whole Network Architecture.
    ARGS:
        images : a placeholder of input images (black and white pencil sketch)
    OUTPUT:
        output_layer : The final output after the colorization network. (a predicted colored image)
        classification_layer_2 : The output of the classification network. (a prediction of class of the image)
    '''
    
    #image = tf.placeholder("float", [None, 224, 224, 1])

    # low_level_conv1
    with tf.variable_scope('low_level_conv1') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 64],stddev=5e-2))
        bias = tf.Variable(tf.constant(0.0,shape=[64]))
        conv = tf.nn.conv2d(images, weight, [1, 2, 2, 1], padding="SAME")
        low_level_conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias))
        _activation_summary(low_level_conv1)

    
    # low_level_conv2
    with tf.variable_scope('low_level_conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv2)

    # low_level_conv3
    with tf.variable_scope('low_level_conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv2, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv3)

    # low_level_conv4
    with tf.variable_scope('low_level_conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv4)    

    # low_level_conv5
    with tf.variable_scope('low_level_conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv4, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv5)

    # low_level_conv6
    with tf.variable_scope('low_level_conv6') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv6 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv6)

    # mid_level_conv1
    with tf.variable_scope('mid_level_conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        mid_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(mid_level_conv1)

    # mid_level_conv2
    with tf.variable_scope('mid_level_conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(mid_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        mid_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        
        _activation_summary(mid_level_conv2)

    # global_level_conv1
    with tf.variable_scope('global_level_conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(low_level_conv6, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv1)

    # global_level_conv2
    with tf.variable_scope('global_level_conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(global_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv2)

    # global_level_conv3
    with tf.variable_scope('global_level_conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(global_level_conv2, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv3)

    # global_level_conv4
    with tf.variable_scope('global_level_conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(global_level_conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv4)

    # global_level_FC1
    with tf.variable_scope('global_level_FC1') as scope:
        
        reshape = tf.reshape(global_level_conv4, [-1, 7*7*512])
        global_level_FC1 = tf.layers.dense(inputs=reshape, units=1024,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
                                           bias_initializer=tf.constant_initializer(0.1),
                                           name=scope.name)
                                           
        '''

        reshape = tf.reshape(global_level_conv4, [BATCH_SIZE, -1])   #512*7*7 is from main paper diagram
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 1024], 
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        global_level_FC1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        '''
        _activation_summary(global_level_FC1)
    
    # global_level_FC2
    with tf.variable_scope('global_level_FC2') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[1024, 512],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        global_level_FC2 = tf.nn.relu(tf.matmul(global_level_FC1, weights) + biases, name=scope.name)
        _activation_summary(global_level_FC2)

    # global_level_FC3
    with tf.variable_scope('global_level_FC3') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[512, 256],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        global_level_FC3 = tf.nn.relu(tf.matmul(global_level_FC2, weights) + biases, name=scope.name)
        
        _activation_summary(global_level_FC3)

    # classification_network
    with tf.variable_scope('classification_network') as scope:
        weights1 = _variable_with_weight_decay('weights1',
                                              shape=[512, 256],
                                              stddev=0.04,
                                              wd=0.004)
        biases1 = _variable_on_cpu('biases1', [256], tf.constant_initializer(0.1))
        classification_layer_1 = tf.nn.sigmoid(tf.matmul(global_level_FC2, weights1) + biases1, name='classification_layer_1')
        _activation_summary(classification_layer_1)


        weights2 = _variable_with_weight_decay('weights2',
                                               shape=[256, NUM_CLASSES],
                                               stddev=0.04,
                                               wd=0.004)
        biases2 = _variable_on_cpu('biases2', [NUM_CLASSES], tf.constant_initializer(0.1))
        classification_layer_2 = tf.matmul(classification_layer_1, weights2) + biases2
        _activation_summary(classification_layer_2)

    
    # fusion_layer
    with tf.variable_scope('fusion_layer') as scope:
        print('global level Fc3 = %s' % global_level_FC3)
        print('mid level conv2 : %s' % mid_level_conv2)
        mid_level_conv2_reshaped = tf.reshape(mid_level_conv2,[-1, 28*28,256])
        print('mid_level_conv2_reshaped : %s' % mid_level_conv2_reshaped)
        mid_level_conv2_reshaped = tf.unstack(mid_level_conv2_reshaped,axis=1)
        print('mid_level_conv2_reshaped and unstacked : %d' % len(mid_level_conv2_reshaped))
        print('mid_level_conv2_reshaped and unstacked[0] : %s' % mid_level_conv2_reshaped[0])
        fusion_level = [tf.concat([see_mid,global_level_FC3],axis=1) for see_mid in mid_level_conv2_reshaped]
        print('After concatination of 2 layers')
        print('fusion_level length : %d ' % len(fusion_level))
        print('each fusion_level has : %s' % fusion_level[0])
        fusion_level = tf.stack(fusion_level,axis=1)
        print('fusion_level after stack : %s' % fusion_level)
        fusion_level = tf.reshape(fusion_level,[-1,28,28,512])
        
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 512, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(fusion_level, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        fusion_layer = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(fusion_layer)
    
    # colorization_level_conv1
    with tf.variable_scope('colorization_level_conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(fusion_layer, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        colorization_level_conv1_upsampled = tf.image.resize_images(images=colorization_level_conv1,
                                                                    size=tf.constant(value=[56,56], dtype=tf.int32),
                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _activation_summary(colorization_level_conv1_upsampled)
        
    # colorization_level_conv2
    with tf.variable_scope('colorization_level_conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(colorization_level_conv1_upsampled, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(colorization_level_conv2)

    # colorization_level_conv3
    with tf.variable_scope('colorization_level_conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(colorization_level_conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        colorization_level_conv3_upsampled = tf.image.resize_images(images=colorization_level_conv3,
                                                                    size=tf.constant(value=[112,112], dtype=tf.int32),
                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _activation_summary(colorization_level_conv3_upsampled)
        
    # colorization_level_conv4
    with tf.variable_scope('colorization_level_conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 32],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(colorization_level_conv3_upsampled, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(colorization_level_conv4)
        
    # output_layer
    with tf.variable_scope('output_layer') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(colorization_level_conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        output_layer = tf.nn.sigmoid(pre_activation, name=scope.name)
        _activation_summary(output_layer)


    with tf.variable_scope('final_colored_output') as scope:
        output_layer_upsampled = tf.image.resize_images(images=output_layer,
                                                        size=tf.constant(value=[224,224], dtype=tf.int32),
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        print('output_layer_upsampled : %s' % output_layer_upsampled)
        chrominance = (output_layer_upsampled*255)-127 # a,b value = range(-127,+128)
        luminance = images[:,:,:,0:1] # L value = range(0,100)

        final_colored_images = tf.concat([luminance, chrominance],axis=3) # in CL*a*b* format
        print('final_colored_images : %s' %final_colored_images)
        
    return final_colored_images, classification_layer_2


def loss(network_output, output, classification_output, classification_labels, alpha=1):
    """
    This returns the total loss of the network model.
    ARGS:
        network_output : The final output layer of the whole network i.e. output of colorization network - A Tensor
        output : The actual expected output of the layer (Labels). It should have the shape of network_output Tensor - A Tensor
        classification_output : The output of the classification layer i.e. an array of TOTAL_NUMBER_OF_CLASSES
        classification_labels : The actual labels of classes. its length must be same as classification_output
        alpha : (default = 1) The factor deciding what factor of the classification loss should be used in computing the total loss
    OUTPUT:
        Total Loss - A Scalar Tensor
    """
    
    loss_classification = tf.losses.sparse_softmax_cross_entropy(labels=classification_labels, logits=classification_output)
    loss_colorization = tf.losses.mean_squared_error(network_output, output)
    return loss_colorization + alpha * loss_classification

def train(total_loss, global_step):
    """
    This optimizes the loss function with Gradient Descent Optimizer with exponentially decaying learning rate.
    This also adds the histograms of various variables.
    ARGS:
        total_loss : the tensor returned by the loss funtion
        global_step : The step number of the training
    OUTPUT:
        an empty operation is returned (no use, just for namesake)
        but the weights are updated and optimization is applied
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

    
