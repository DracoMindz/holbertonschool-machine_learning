# 0x11. Attention
Specialization Machine Learning
Supervised Learning

##Learning Objectives

At the end of this project, you are expected
to be able to explain to anyone, without the
help of Google:

###**General**
```
    What is the attention mechanism?
    How to apply attention to RNNs
    What is a transformer?
    How to create an encoder-decoder transformer model
    What is GPT?
    What is BERT?
    What is self-supervised learning?
    How to use BERT for specific NLP tasks
    What is SQuAD? GLUE?
```

Update Tensorflow to 1.15 for this project.

##Tasks

**0. RNN Encoder**

Create a class RNNEncoder that inherits from 
tensorflow.keras.layers.Layer to encode for 
machine translation.
___
**1. Self Attention**

Create a class SelfAttention that inherits from
tensorflow.keras.layers.Layer to calculate the
attention for machine translation.
___
**2. RNN Decoder**

Create a class RNNDecoder that inherits from
tensorflow.keras.layers.Layer to decode for 
machine translation.
___
**3. Positional Encoding**

Write the function def positional_encoding(max_seq_len, dm):
that calculates the positional encoding for a transformer.
___
**4. Scaled Dot Product Attention**

Write the function def sdp_attention(Q, K, V, mask=None)
that calculates the scaled dot product attention.
___
**5. Multi Head Attention**

 Create a class MultiHeadAttention that inherits from
 tensorflow.keras.layers.Layer to perform multi head
 attention.
___
**6. Transformer Encoder Block**
 
Create a class EncoderBlock that inherits from
tensorflow.keras.layers.Layer to create an encoder
block for a transformer.
___
**7. Transformer Decoder Block**

Create a class DecoderBlock that
inherits from tensorflow.keras.layers.Layer 
to create an encoder block for a transformer.
___
**8. Transformer Encoder**

Create a class Encoder that inherits from
tensorflow.keras.layers.Layer to create the
encoder for a transformer.
___
**9. Transformer Decoder**

Create a class Decoder that inherits from 
tensorflow.keras.layers.Layer to create the 
decoder for a transformer.
___
**10. Transformer Network**

Create a class Transformer that inherits
from tensorflow.keras.Model to create a
transformer network.
___

