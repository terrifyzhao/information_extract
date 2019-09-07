from keras.layers import *
from keras.models import Model
from keras.optimizers import *
import keras.backend as K


def position_embed(x):
    # [batch_size, seq_len]
    # [seq_len,]
    p = K.arange(K.shape(x)[1])
    # [1, seq_len]
    p = K.expand_dims(p, 0)
    # [batch_size, seq_len]
    p = K.tile(p, [K.shape(x)[0], 1])

    return p


def dilated_gated_conv1d(x, dilation_rate, mask):
    dim = K.int_shape(x)[-1]
    conv = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(x)

    def _gate(x):
        x, conv = x
        conv1 = conv[:, :, :dim]
        conv2 = conv[:, :, dim:]
        conv2_sig = K.sigmoid(conv2)
        result = x * (1 - conv2_sig) + conv1 * conv2_sig
        return result

    result = Lambda(_gate)([x, conv])
    result = Lambda(lambda x: x[0] * x[1])([result, mask])
    return result


def max_pool(x):
    x, mask = x
    x -= (1 - mask) * 1e10
    return K.max(x, 1, keepdims=True)


class SelfAttention(Layer):
    def __init__(self, n_head, head_size, **kwargs):
        self.n_head = n_head
        self.head_size = head_size
        self.output_dim = n_head * head_size
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        q_dim = input_shape[0][-1]
        k_dim = input_shape[1][-1]
        self.v_dim = input_shape[2][-1]
        self.matrix_q = self.add_weight(name='matrix_q',
                                        shape=(q_dim, self.output_dim),
                                        initializer='glorot_normal')
        self.matrix_k = self.add_weight(name='matrix_k',
                                        shape=(k_dim, self.output_dim),
                                        initializer='glorot_normal')
        self.matrix_v = self.add_weight(name='matrix_v',
                                        shape=(self.v_dim, self.output_dim),
                                        initializer='glorot_normal')
        self.matrix_scale = self.add_weight(name='matrix_scale',
                                            shape=(self.output_dim, self.v_dim),
                                            initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10

    def call(self, inputs, **kwargs):
        # [batch_size, seq_len, embed_size]  [batch_size, seq_len, 1]
        q, k, v, mask = inputs
        # [batch_size, seq_len, output_dim]
        qw = K.dot(q, self.matrix_q)
        kw = K.dot(k, self.matrix_k)
        vw = K.dot(v, self.matrix_v)
        # [batch_size, seq_len, n_head, head_size]
        qw = K.reshape(qw, [-1, K.shape(qw)[1], self.n_head, self.head_size])
        kw = K.reshape(kw, [-1, K.shape(kw)[1], self.n_head, self.head_size])
        vw = K.reshape(vw, [-1, K.shape(vw)[1], self.n_head, self.head_size])
        # [batch_size, n_head, seq_len, head_size]
        qw = K.permute_dimensions(qw, [0, 2, 1, 3])
        kw = K.permute_dimensions(kw, [0, 2, 1, 3])
        vw = K.permute_dimensions(vw, [0, 2, 1, 3])
        # [batch_size, n_head, seq_len, seq_len]
        out = K.batch_dot(qw, kw, [3, 3]) / (self.head_size ** 0.5)

        # [batch_size, seq_len, seq_len, n_head]
        out = K.permute_dimensions(out, [0, 3, 2, 1])
        # [batch_size, seq_len, seq_len, n_head]
        out = self.mask(out, mask, mode='add')
        # [batch_size, seq_len, seq_len, n_head]
        out = K.permute_dimensions(out, [0, 3, 2, 1])

        # [batch_size, n_head, seq_len, seq_len]
        weight = K.softmax(out)
        # [batch_size, n_head, seq_len, head_size]
        out = K.batch_dot(weight, vw, [3, 2])
        # [batch_size, seq_len, n_head, head_size]
        out = K.permute_dimensions(out, (0, 2, 1, 3))
        # [batch_size, seq_len, head*head_size]
        out = K.reshape(out, (-1, K.shape(out)[1], self.output_dim))
        out = K.dot(out, self.matrix_scale)
        out = self.mask(out, mask, 'mul')
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.v_dim)


def model(char_len, seq_len, num_class, drop_rate=0.2):
    char_size = 128
    position_size = 128
    n_head = 8
    head_size = 64
    kernel_size = 3

    x_char = Input(shape=(None,))
    x_word = Input(shape=(None, 300))
    s_index = Input(shape=(None,))
    s_star = Input(shape=(None,))
    s_end = Input(shape=(None,))
    p_o_star = Input(shape=(None, num_class))
    p_o_end = Input(shape=(None, num_class))
    pres_s = Input(shape=(None, 2))
    pres_po = Input(shape=(None, num_class * 2))

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x_char)

    x_position = Lambda(position_embed)(x_char)
    x_position = Embedding(seq_len, position_size)(x_position)

    x_char_embed = Embedding(char_len + 2, char_size)(x_char)

    x_word_embed = Dense(char_size, use_bias=False)(x_word)

    x = Add()([x_char_embed, x_position, x_word_embed])
    x = Dropout(drop_rate)(x)

    x = Lambda(lambda x: x[0] * x[1])([x, mask])
    x = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=2, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=4, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=2, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=4, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=2, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=4, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    x = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    cnn_out = dilated_gated_conv1d(x, dilation_rate=1, mask=mask)
    cnn_out = Dropout(drop_rate)(cnn_out)
    cnn_dim = K.int_shape(cnn_out)[-1]

    # 全局实体识别
    pn1 = Dense(char_size, activation='relu')(cnn_out)
    pn1 = Dense(1, activation='sigmoid')(pn1)
    pn2 = Dense(char_size, activation='relu')(cnn_out)
    pn2 = Dense(1, activation='sigmoid')(pn2)

    # tm encoder
    x = SelfAttention(n_head, head_size)([x, x, x, mask])
    # x = Add()([x, cnn_out])
    # x = BatchNormalization()(x)

    x = Concatenate()([x, cnn_out, pres_s])
    x = Conv1D(char_size, kernel_size, padding='same', activation='relu')(x)

    s_star_out = Dense(1, activation='sigmoid')(x)
    s_end_out = Dense(1, activation='sigmoid')(x)

    s_star_out = Lambda(lambda x: x[0] * x[1])([s_star_out, pn1])
    s_end_out = Lambda(lambda x: x[0] * x[1])([s_end_out, pn2])

    subject_model = Model([x_char, x_word, pres_s], [s_star_out, s_end_out])

    # object--------------------------------------------------
    # 全局关系存在性判断模块
    t_max = Lambda(max_pool)([cnn_out, mask])
    pc = Dense(char_size, activation='relu')(t_max)
    pc = Dense(num_class, activation='sigmoid')(pc)

    s_embed = Embedding(char_len + 2, char_size)(s_index)
    # s_out = Bidirectional(CuDNNLSTM(cnn_dim))(s_embed)
    s_out = Bidirectional(LSTM(cnn_dim))(s_embed)

    s_star_embed = Embedding(seq_len, position_size)(s_star)
    s_end_embed = Embedding(seq_len, position_size)(s_end)
    s_position_embed = Concatenate()([s_star_embed, s_end_embed])
    s_out = Lambda(lambda x: K.expand_dims(x, 1))(s_out)
    s_out = Add()([s_out, s_position_embed])

    x = SelfAttention(n_head, head_size)([x, x, x, mask])
    # x = Add()([x, cnn_out])
    # x = BatchNormalization()(x)

    x = Concatenate()([x, cnn_out, s_out, pres_s, pres_po])
    x = Conv1D(char_size, kernel_size, padding='same', activation='relu')(x)

    # 全局关系检测模块
    po = Dense(1, activation='sigmoid')(x)

    p_o_star_out = Dense(num_class, activation='sigmoid')(x)
    p_o_end_out = Dense(num_class, activation='sigmoid')(x)
    p_o_star_out = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, p_o_star_out, pc, pn1])
    p_o_end_out = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, p_o_end_out, pc, pn2])

    # p_o_star_out = Dense(num_class, activation='sigmoid')(p_o_star_out)
    # p_o_end_out = Dense(num_class, activation='sigmoid')(p_o_end_out)

    object_model = Model([x_char, x_word, s_index, s_star, s_end, pres_s, pres_po],
                         [p_o_star_out, p_o_end_out])

    train_model = Model([x_char, x_word, s_index, s_star, s_end, p_o_star, p_o_end, pres_s, pres_po],
                        [s_star_out, s_end_out, p_o_star_out, p_o_end_out])

    s_star = K.expand_dims(s_star, axis=-1)
    s_end = K.expand_dims(s_end, axis=-1)

    s_star_loss = K.binary_crossentropy(s_star, s_star_out)
    s_star_loss = K.sum(s_star_loss * mask) / K.sum(mask)
    s_end_loss = K.binary_crossentropy(s_end, s_end_out)
    s_end_loss = K.sum(s_end_loss * mask) / K.sum(mask)

    p_o_star_loss = K.binary_crossentropy(p_o_star, p_o_star_out)
    p_o_star_loss = K.sum(p_o_star_loss * mask) / K.sum(mask)
    p_o_end_loss = K.binary_crossentropy(p_o_end, p_o_end_out)
    p_o_end_loss = K.sum(p_o_end_loss * mask) / K.sum(mask)

    loss = s_star_loss + s_end_loss + p_o_star_loss + p_o_end_loss

    train_model.add_loss(loss)
    train_model.compile(Adam(1e-3))
    train_model.summary()

    return train_model, subject_model, object_model


if __name__ == '__main__':
    model(500, 50, 100)
