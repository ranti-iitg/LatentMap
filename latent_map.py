import tensorflow as tf

from cell import ConvGRUCell, flatten, expand

batch_size = 1
timesteps = 5
width = 100
height = 100
channels = 512
filters = 3
kernel = [3, 3]
slice_dimension=3

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps, width, height, channels])
y_slice=tf.placeholder(tf.int32,[batch_size,timesteps,slice_dimension])
output_y=tf.placeholder(tf.int32,[batch_size,timesteps,width,height,channels])

# Flatten input because tf.nn.dynamic_rnn only accepts 3D input.
inputs = flatten(inputs)

# Add the ConvLSTM step.
cell = ConvGRUCell(height, width, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# Reshape outputs to videos again, because tf.nn.dynamic_rnn only accepts 3D input.
outputs = expand(outputs, height, width, filters)

for o_batch,y_batch in zip(outputs,y_slice):
	for o_time,y_time in zip(o_batch,y_batch):

unpack_outputs=tf.unstack(outputs)
unpack_y_slice=tf.unstack(y_slice)

out1=tf.slice(outputs[0][0],y_slice[0][0],[10,10,3])
out2=tf.slice(outputs[0][2],y_slice[0][2],[10,10,3])
out4=tf.slice(outputs[0][3],y_slice[0][3],[10,10,3])

out_now=[[tf.slice(o_time,y_time,[10,10,3]) for o_time,y_time in zip(tf.unstack(o_batch),tf.unstack(y_batch))] for o_batch,y_batch in zip(unpack_outputs,unpack_y_slice)]
out_stack=tf.stack(out_now)
samples_0, timesteps_0, height_0, width_0, filters_0 = out_stack.get_shape().as_list()
out_stack2=tf.reshape(out_stack,[samples_0*timesteps_0,height_0,width_0,filters_0])
#y = tf.unpack(tf.transpose(y, (1, 0, 2)))

conv_t1 = Conv2Dtranspose(out_stack2, (7, 7), 8, 8,(3, 3), activation='relu')
conv_t1_out = conv_t1.output()
conv_t2 = Conv2Dtranspose(conv_t1_out, (14, 14), 8, 8,(3, 3), activation='relu')
conv_t2_out = conv_t2.output()

conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 8, 16,(3, 3), activation='relu')
conv_t3_out = conv_t3.output()

conv_last = Convolution2D(conv_t3_out, (28, 28), 16, 1, (3, 3),activation='sigmoid')
decoded = conv_last.output()

decoded = tf.reshape(decoded, [-1, 784])
cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
loss = tf.reduce_mean(cross_entropy)



class Convolution2D(object):
    '''
      constructor's args:
          input     : input image (2D matrix)
          input_siz ; input image size
          in_ch     : number of incoming image channel
          out_ch    : number of outgoing image channel
          patch_siz : filter(patch) size
          weights   : (if input) (weights, bias)
    '''
    def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input      
        self.rows = input_siz[0]
        self.cols = input_siz[1]
        self.in_ch = in_ch
        self.activation = activation
        
        wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]
        
        w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
                            trainable=True)
        b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
                            trainable=True)
        
        self.w = w_cv
        self.b = b_cv
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [-1, self.rows, self.cols, self.in_ch]
        
        x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
        linout = tf.nn.conv2d(x_image, self.w, 
                  strides=[1, 1, 1, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout
        
        return self.output

# Max Pooling Layer   
class MaxPooling2D(object):
    '''
      constructor's args:
          input  : input image (2D matrix)
          ksize  : pooling patch size
    '''
    def __init__(self, input, ksize=None):
        self.input = input
        if ksize == None:
            ksize = [1, 2, 2, 1]
            self.ksize = ksize
    
    def output(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
                    strides=[1, 2, 2, 1], padding='SAME')
  
        return self.output

# Full-connected Layer   
class FullConnected(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
    
        w_h = tf.Variable(tf.truncated_normal([n_in,n_out],
                          mean=0.0, stddev=0.05), trainable=True)
        b_h = tf.Variable(tf.zeros([n_out]), trainable=True)
     
        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)
        
        return self.output


class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''
    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input      
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.out_ch = out_ch
        self.activation = activation
        
        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]    # note the arguments order
        
        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
                            trainable=True)
        b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
                            trainable=True)
        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]      
        linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                            strides=[1, 2, 2, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout
        
return self.output



def mk_nn_model(x, y_):
    # Encoding phase
    x_image = tf.reshape(x, [-1, 28, 28, 1])    
    conv1 = Convolution2D(x_image, (28, 28), 1, 16, 
                          (3, 3), activation='relu')
    conv1_out = conv1.output()
    
    pool1 = MaxPooling2D(conv1_out)
    pool1_out = pool1.output()
    
    conv2 = Convolution2D(pool1_out, (14, 14), 16, 8, 
                          (3, 3), activation='relu')
    conv2_out = conv2.output()
    
    pool2 = MaxPooling2D(conv2_out)
    pool2_out = pool2.output()

    conv3 = Convolution2D(pool2_out, (7, 7), 8, 8, (3, 3), activation='relu')
    conv3_out = conv3.output()

    pool3 = MaxPooling2D(conv3_out)
    pool3_out = pool3.output()
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    # Decoding phase
    conv_t1 = Conv2Dtranspose(pool3_out, (7, 7), 8, 8,
                         (3, 3), activation='relu')
    conv_t1_out = conv_t1.output()

    conv_t2 = Conv2Dtranspose(conv_t1_out, (14, 14), 8, 8,
                         (3, 3), activation='relu')
    conv_t2_out = conv_t2.output()

    conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 8, 16, 
                         (3, 3), activation='relu')
    conv_t3_out = conv_t3.output()

    conv_last = Convolution2D(conv_t3_out, (28, 28), 16, 1, (3, 3),
                         activation='sigmoid')
    decoded = conv_last.output()

    decoded = tf.reshape(decoded, [-1, 784])
    cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)
       
    return loss, decoded

