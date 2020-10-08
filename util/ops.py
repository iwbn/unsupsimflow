import tensorflow as tf

def per_batch_op(inp, func, out_type=tf.float32):
    l = tf.shape(inp)[0]
    i = tf.constant(0)
    a = tf.TensorArray(out_type, size=l, dynamic_size=False)
    c = lambda i, a: tf.less(i, l)
    def b(i, a):
        v = func(inp[i])
        a = a.write(i, v)
        i = i + 1
        return i, a
    i, a = tf.while_loop(c, b, [i, a], parallel_iterations=True)
    r = a.stack()
    return r