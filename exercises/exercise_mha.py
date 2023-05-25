import tensorflow as tf
def exercise_multihead_attention():
    '''Understand the inner workings of tf.keras.layers.MultiHeadAttention.
    
    Vaswani et al - 2017 - Attention is all you need
    
      Instead of performing a single attention function with d model -dimensional
      keys, values and queries, we found it beneﬁcial to linearly project the
      queries, keys and values h times with different, learned linear projections
      to d k , d k and d v dimensions, respectively. On each of these projected
      versions of queries, keys and values we then perform the attention function
      in parallel, yielding d v -dimensional output values. These are
      concatenated and once again projected, resulting in the ﬁnal values, as
      depicted in Figure 2.

      Multi-head attention allows the model to jointly attend to information from
      different representation subspaces at different positions. With a single
      attention head, averaging inhibits this.

      MultiHead(Q, K, V ) = Concat(head_1 , ..., head_h ) W^O 
                   head_i = Attention(QW_i^ Q , KW_i^K , V W_i^V)
    '''

    tf.random.set_seed(1234)
    mha = tf.keras.layers.MultiHeadAttention(key_dim=3, num_heads=5)

    query = tf.random.uniform((2,  7, 11)) # (batch_size, query_elements, query_depth)
    key   = tf.random.uniform((2,  9, 13)) # (batch_size, key_elements,   key_depth)
    value = tf.random.uniform((2,  9, 17)) # (batch_size, key_elements,   value_depth)

    tf_result, tf_scores = mha(query, value, key, return_attention_scores=True) 
    query_kernel, query_bias, key_kernel, key_bias, value_kernel, value_bias, projection_kernel, projection_bias = mha.weights

    tf.debugging.assert_near( projection_bias, tf.zeros_like(projection_bias) ) # Per default, bias initializer is zero.
    assert query_kernel.shape == (11,5,3)
    assert key_kernel.shape   == (13,5,3)
    assert value_kernel.shape == (17,5,3)
    # assert value_kernel.shape == 
    assert projection_bias.shape == (11,) # 'the query input last dimension if output_shape is None'
    # query ~ (2,7,11), query_kernel ~ (11,5,3) => q ~ (2,7,5,3)
    q = tf.einsum('????', query_kernel, query)
    assert q.shape == (2, 7, 5, 3)
    # key ~ (2,9,13), key_kernel ~ (13,5,3) => k ~ (2,9,5,3)
    k = tf.einsum('????', key_kernel, key)
    assert k.shape == (2, 9, 5, 3)
    # value ~ (2,9,17), value_kernel ~ (17,5,3) => v ~ (2,9,5,3)
    v = tf.einsum('????', value_kernel, value)
    assert v.shape == (2, 9, 5, 3)
    inner = TODO
    inner /= tf.sqrt(3.)
    sm = tf.nn.softmax( inner )
    multi_head_output = TODO
    tmp = tf.einsum("...NHK, ...HKO -> ...NO", multi_head_output, projection_kernel)
    tf.debugging.assert_near( tmp, tf_result )

def test_attention():
    '''Understand the inner workings of tf.keras.layers.Attention.'''
    inputA = tf.keras.Input(shape=(3,4))
    inputB = tf.keras.Input(shape=(3,4))
    inputC = tf.keras.Input(shape=(3,4))
    ##
    # key: Optional key Tensor of shape [batch_size, Tv, dim]. If not given, will use value for both key and value, which is the most common case.
    ##
    at = tf.keras.layers.Attention()([inputA, inputB, inputC])
    model = tf.keras.Model(inputs=[inputA,inputB,inputC], outputs=at, name="ATTTT")

    tf.random.set_seed(123)
    queries = tf.random.uniform( (1,3,4) )
    x = [1.,100.,100000.]
    y = [10.,10.,10.,10.]
    values = tf.tensordot(x,y,axes=0)[None,:,:]
    print(values, values.shape)
    keys   = tf.random.uniform( (1,3,4) )

    by_hand = TODO

    tf.debugging.assert_equal( by_hand, model([queries,values,keys]) )

exercise_multihead_attention()