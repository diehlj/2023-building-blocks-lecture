import tensorflow as tf

# Create a 3x4 tensor
tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=tf.float32)

# Exercise: Extract the second row of the tensor
second_row = TODO
tf.debugging.assert_equal(second_row, tf.constant([5, 6, 7, 8], dtype=tf.float32))

# Exercise: Extract the third column of the tensor
third_column = TODO
tf.debugging.assert_equal(third_column, tf.constant([3, 7, 11], dtype=tf.float32))

# Exercise: Extract the submatrix from rows 1 to 2 and columns 1 to 3
submatrix = TODO
tf.debugging.assert_equal(submatrix, tf.constant([[6, 7, 8], [10, 11, 12]], dtype=tf.float32))


# Exercise: Create two 3x3 tensors
tensor1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tensor2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Exercise: Multiply the two tensors element-wise
elementwise_product = TODO
tf.debugging.assert_equal(elementwise_product, tf.constant([[5, 12], [21, 32]], dtype=tf.float32))

# Exercise: Perform matrix multiplication of the two tensors
matrix_product = TODO
tf.debugging.assert_equal(matrix_product, tf.constant([[19, 22], [43, 50]], dtype=tf.float32))

###########

tensor = tf.constant([[1], [2], [3]], dtype=tf.float32)

# Exercise: Remove the singleton dimension using squeeze
squeezed_tensor = TODO
expected_squeezed_tensor = tf.constant([1, 2, 3], dtype=tf.float32)
tf.debugging.assert_near(squeezed_tensor, expected_squeezed_tensor)

# Exercise: Add a singleton dimension using expand_dims (unsqueeze equivalent)
unsqueezed_tensor = TODO
tf.debugging.assert_near(unsqueezed_tensor, tensor)


##############

tensor1 = tf.random.normal([2, 3, 4], dtype=tf.float32)
tensor2 = tf.random.normal([2, 3, 4], dtype=tf.float32)

# Exercise: Use einsum to multiply the two tensors element-wise:
result = TODO
expected_result = tf.multiply(tensor1, tensor2)

tf.debugging.assert_near(result, expected_result)


##############

tensor = tf.random.normal([2, 3, 4], dtype=tf.float32)

# Exercise: Use einsum to sum over the last axis
result = TODO
expected_result = tf.reduce_sum(tensor, axis=-1)

# Exercise: Put the correct shape:
assert result.shape == TODO
tf.debugging.assert_near(result, expected_result)

##############

tensor1 = tf.random.normal([3, 4, 2], dtype=tf.float32)
tensor2 = tf.random.normal([3, 4, 2], dtype=tf.float32)

# Exercise: Use einsum to contract the last axis of tensor1 with the last axis of tensor2.
result = TODO
expected_result = tf.reduce_sum(tensor1 * tensor2, axis=-1)
print(tensor1)
print(tensor2)
print(result)
# Exercise: Put the correct shape:
assert result.shape == TODO
tf.debugging.assert_near(result, expected_result)

##############

# Exercise: What does * do?
tensor1 = tf.random.normal([3, 4, 2], dtype=tf.float32)
tensor2 = tf.random.normal([3, 4, 2], dtype=tf.float32)

einsum_result = TODO

print(tensor1 * tensor2)
tf.debugging.assert_near(einsum_result, tensor1 * tensor2)

# Exercise: What does @ do?
tensor1 = tf.random.normal([3, 4, 2], dtype=tf.float32)
tensor2 = tf.random.normal([2, 5], dtype=tf.float32)
print(tensor1 @ tensor2)
einsum_result = TODO
tf.debugging.assert_near(einsum_result, tensor1 @ tensor2)


############
# Matrix multiplication of tensors with arbitrary dimensions:

tensor1 = tf.random.normal([2, 3, 4], dtype=tf.float32)
tensor2 = tf.random.normal([2, 4, 5], dtype=tf.float32)

result = TODO
expected_result = tf.matmul(tensor1, tensor2)

tf.debugging.assert_near(result, expected_result)

#############
# Batched outer product of tensors with arbitrary dimensions:
tensor1 = tf.random.normal([3, 4, 2], dtype=tf.float32)
tensor2 = tf.random.normal([3, 4, 2], dtype=tf.float32)

result = TODO
expected_result = tf.einsum('abj,abk->abjk', tensor1, tensor2)

tf.debugging.assert_near(result, expected_result)