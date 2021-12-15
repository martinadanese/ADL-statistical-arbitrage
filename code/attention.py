import tensorflow as tf

class Bahdanau(tf.keras.layers.Layer):
    """
    Class implemnting Bahdanau attention (Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv:1409.0473.)
    Members

     -  w, u are determined using a single dense layer of number_of_units units (hyperparameter). 
     -  v is determined by a single dense layer of 1 unit
     -  The context is calculated in call() method, after weights determination
    """

    #   ----------------------------------------------------- 
    def __init__(self, number_of_units):
    #   ----------------------------------------------------- 
        super(Bahdanau, self).__init__()
        self.number_of_units = number_of_units
        self.w = tf.keras.layers.Dense(number_of_units)
        self.u = tf.keras.layers.Dense(number_of_units)
        self.v = tf.keras.layers.Dense(1)    
    

    #   ----------------------------------------------------- 
    def get_config(self):
    #   ----------------------------------------------------- 
        """
        mandatory method to get the input arguments
        """
        config = super().get_config()#.copy()
        config.update({
            'number_of_units': self.number_of_units,
        })
        return config
    


    #   ----------------------------------------------------- 
    def call(self, all_hidden_states, last_hidden_state):
    #   ----------------------------------------------------- 
        """
        Context calculator:
         - expand the last_hidden_state to (?,1,128) for the addition that follows later along the time axis
         - apply soft attention score 
         - compute soft attention alignment model and determine w,u and v
         - pass to soft max to obtain the attention weights
         - attention_weights are multiplied with all_hidden_states and summed up to return the context
         
         ------
         input : last_hidden_state and all_hidden_states of lstm
         output: the weighted context
        
        """
        
        last_hidden_state = tf.expand_dims(last_hidden_state, 1)
        alignment_model = self.v(tf.nn.tanh(self.w(last_hidden_state) + self.u(all_hidden_states)))
        attention_weights = tf.nn.softmax(alignment_model, axis=1)
        context = attention_weights*all_hidden_states
        context = tf.reduce_sum(context, axis=1)        
        return context







class Luong(tf.keras.layers.Layer):
    """
     Implements Luong general attention (Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.)
     -  w is determined using a single dense layer of number_of_units units (hyperparameter). 
    """

    #   ----------------------------------------------------- 
    def __init__(self, number_of_units):
    #   ----------------------------------------------------- 
        super(Luong, self).__init__()
        self.number_of_units = number_of_units
        self.w = tf.keras.layers.Dense(number_of_units)
    

    #   ----------------------------------------------------- 
    def get_config(self):
    #   ----------------------------------------------------- 
        """
        mandatory method to get the input arguments
        """
        config = super().get_config()#.copy()
        config.update({
            'number_of_units': self.number_of_units,
        })
        return config
    


    #   ----------------------------------------------------- 
    def call(self, all_hidden_states):
    #   ----------------------------------------------------- 
        """
        Context calculator:
         - compute the score multiplying the outcome of the lstm
         - apply softmax and determine w
         - pass to soft max to obtain the attention weights
         - attention_weights are multiplied with all_hidden_states and summed up to return the context
         
         ------
         input : last_hidden_state and all_hidden_states of lstm
         output: the weighted context
        
        """
        
        score = all_hidden_states*all_hidden_states
        attention_weights = tf.nn.softmax(self.w(score), axis=1)
        context = attention_weights*all_hidden_states
        context = tf.reduce_sum(context, axis=1)        
        return context
